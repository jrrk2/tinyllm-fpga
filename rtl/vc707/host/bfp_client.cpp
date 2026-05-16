// bfp_client.cpp — pure C++ host for the VC707 BFP autoregress bitstream.
//
// No python, no torch, no transformers — just POSIX sockets + the
// FastTrans wire format eth_ctrl.sv parses (FT_REG_WRITE / FT_REG_READ
// / FT_DDR_WRITE / FT_ACK).
//
// Subcommands (all assume PEER = 192.168.1.42:19783; override with -p):
//   discover          look up the FPGA's IP from its hardcoded MAC by
//                     scanning the host's ARP cache (and seeding it
//                     with a one-shot subnet probe if not yet cached)
//   upload  <bin>     bulk-write the weight image to DDR3 base 0
//   verify  <bin>     restart, capture the streamer's first AXI read,
//                     compare 64 bytes of DDR3 against the file
//   restart           pulse 0x1F1[0] and poll 0x1F0 for done
//   tokens            read 10 result words from 0x1D0, print 19 raw IDs
//                     (and decoded text if -v <vocab.bin> is supplied)
//   all     <bin>     upload → verify → restart → tokens
//
// Build:
//   g++ -std=c++17 -O2 -Wall -Wextra -o bfp_client bfp_client.cpp
//
// Wire-format sources of truth (do not edit without re-syncing):
//   rtl/vc707/src/microgpt_eth_ctrl.sv  — frame parser + FT_* opcodes
//   rtl/vc707/src/vc707_microgpt_eth.sv — register map (line 1500+)
//
// The FT_REG_WRITE entry layout has a 5-byte header pad + 16-bit
// trailing pad per entry; entries start at packet byte 8 (see
// memory note feedback_ft_reg_write_format).

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr const char* DEFAULT_PEER_IP   = "192.168.1.42";
constexpr uint16_t    DEFAULT_PEER_PORT = 19783;

// Hardcoded MAC of the VC707 BFP bitstream — locally-administered (02:),
// payload bytes encode "MG1" = MicroGPT v1.  See README + framing_top
// for where this gets baked into the wire.
constexpr const char* FPGA_MAC = "02:00:00:4d:47:31";

constexpr uint8_t FT_REG_WRITE = 0x01;
constexpr uint8_t FT_REG_READ  = 0x02;
constexpr uint8_t FT_REG_RSP   = 0x03;
constexpr uint8_t FT_HEARTBEAT = 0x04;
constexpr uint8_t FT_ACK       = 0x06;
constexpr uint8_t FT_DDR_WRITE = 0x0A;

// Register addresses (10-bit, see vc707_microgpt_eth.sv ~line 1500).
constexpr uint16_t REG_DDR_LOAD_TOG  = 0x010;   // bit 0 toggles DDR-write swap (legacy)
constexpr uint16_t REG_DBG_STATUS    = 0x012;   // {dbg_snap_done, dbg_first_r_seen,
                                                //  dbg_first_ar_seen, init_calib_complete}
constexpr uint16_t REG_DBG_ARADDR    = 0x013;   // first AR addr captured (low 30 bits)
constexpr uint16_t REG_DBG_RDATA0    = 0x014;   // first R data: 16 words = 512 bits
constexpr uint16_t REG_ETH_RING      = 0x018;   // {15'd0, lastbuf[4:0], nextbuf[4:0], firstbuf[4:0]}
constexpr uint16_t REG_DDR_WR_RX     = 0x019;   // FT_DDR_WRITE frames accepted
constexpr uint16_t REG_DDR_WR_DONE   = 0x01A;   // ddr_wr_req toggles (writes dispatched)
constexpr uint16_t REG_DDR_WR_ACK    = 0x01B;   // MIG ack toggles seen
constexpr uint16_t REG_DDR_WR_TX     = 0x01C;   // FT_ACK frames dispatched back
constexpr uint16_t REG_HAS_RUN       = 0x049;   // bit 0: single-shot run complete
constexpr uint16_t REG_RDATA_CRC     = 0x04A;   // CRC32 over BFP master's R beats
constexpr uint16_t REG_BUILD_VERSION = 0x10F;
constexpr uint16_t REG_RESULT        = 0x1D0;
constexpr uint16_t REG_DONE          = 0x1F0;   // {30'd0, lay_done_latched, lay_done}
constexpr uint16_t REG_RESTART       = 0x1F1;   // bit 0: write 1 to pulse restart

constexpr int N_STEPS_DEFAULT = 19;   // matches LBFP_FULL_NPROMPT+NGEN baked
constexpr int N_STEPS_MAX     = 64;   // regmap 0x1D0..0x1EF = 32 words = 64 × 16-bit

constexpr int      CHUNK            = 64;            // bytes per FT_DDR_WRITE
constexpr int      DEFAULT_INFLIGHT = 32;
constexpr int      RECV_TIMEOUT_MS  = 50;

// ---------------------------------------------------------------------
// MAC → IP discovery via /proc/net/arp.
// ---------------------------------------------------------------------
// /proc/net/arp format (after a one-line header):
//   IP address       HW type     Flags       HW address            Mask     Device
//   192.168.1.42     0x1         0x2         02:00:00:4d:47:31     *        eno1
// Flags 0x2 = ATF_COM (entry is complete / resolved).  We accept any
// non-zero-MAC entry that matches the target MAC, case-insensitive.

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string lookup_ip_for_mac(const std::string& target_mac) {
    std::ifstream f("/proc/net/arp");
    if (!f) return {};
    std::string line;
    std::getline(f, line);  // discard header
    std::string want = lower(target_mac);
    while (std::getline(f, line)) {
        char ip[64], hw_type[16], flags[16], hw_addr[64], mask[16], dev[64];
        int n = std::sscanf(line.c_str(), "%63s %15s %15s %63s %15s %63s",
                            ip, hw_type, flags, hw_addr, mask, dev);
        if (n < 4) continue;
        if (lower(hw_addr) == want) return std::string(ip);
    }
    return {};
}

// Fire one UDP packet at every host in a /24 around the seed IP.  The
// kernel sends ARP requests as a side effect, populating /proc/net/arp.
// We don't care if any of these reach a real listener — most won't.
void seed_arp_subnet(const std::string& seed_ip, uint16_t port = 7) {
    in_addr seed{};
    if (::inet_pton(AF_INET, seed_ip.c_str(), &seed) != 1) return;
    uint32_t base = ntohl(seed.s_addr) & 0xffffff00u;
    int s = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0) return;
    int bcast = 1;
    ::setsockopt(s, SOL_SOCKET, SO_BROADCAST, &bcast, sizeof(bcast));
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port   = htons(port);
    const char probe[1] = {0};
    for (uint32_t i = 1; i < 255; ++i) {
        dst.sin_addr.s_addr = htonl(base | i);
        ::sendto(s, probe, sizeof(probe), 0, (sockaddr*)&dst, sizeof(dst));
    }
    ::close(s);
    // Give the kernel a moment to resolve.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

class Udp {
public:
    Udp(const std::string& peer_ip, uint16_t peer_port) {
        fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (fd_ < 0) throw std::runtime_error("socket() failed");
        std::memset(&peer_, 0, sizeof(peer_));
        peer_.sin_family = AF_INET;
        peer_.sin_port   = htons(peer_port);
        if (::inet_pton(AF_INET, peer_ip.c_str(), &peer_.sin_addr) != 1)
            throw std::runtime_error("bad peer IP");
        set_recv_timeout(RECV_TIMEOUT_MS);
        // Bind to ephemeral so heartbeats etc. can flow back.
        sockaddr_in local{};
        local.sin_family = AF_INET;
        local.sin_port   = 0;
        local.sin_addr.s_addr = htonl(INADDR_ANY);
        if (::bind(fd_, (sockaddr*)&local, sizeof(local)) < 0)
            throw std::runtime_error("bind() failed");
    }
    ~Udp() { if (fd_ >= 0) ::close(fd_); }
    Udp(const Udp&) = delete;
    Udp& operator=(const Udp&) = delete;

    void set_recv_timeout(int ms) {
        timeval tv{ms / 1000, (ms % 1000) * 1000};
        ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }

    void send(const void* buf, size_t n) {
        ::sendto(fd_, buf, n, 0, (sockaddr*)&peer_, sizeof(peer_));
    }

    // Returns true if a frame was received before SO_RCVTIMEO fires.
    bool recv(uint8_t* buf, size_t cap, size_t* out_n) {
        ssize_t n = ::recvfrom(fd_, buf, cap, 0, nullptr, nullptr);
        if (n < 0) return false;
        if (out_n) *out_n = static_cast<size_t>(n);
        return true;
    }

    // Genuine non-blocking recv (regardless of SO_RCVTIMEO, which on
    // Linux interprets {0,0} as "block forever", not "don't block").
    bool try_recv(uint8_t* buf, size_t cap, size_t* out_n) {
        ssize_t n = ::recvfrom(fd_, buf, cap, MSG_DONTWAIT, nullptr, nullptr);
        if (n < 0) return false;
        if (out_n) *out_n = static_cast<size_t>(n);
        return true;
    }

    // Drop any heartbeats that have piled up before we issue a query.
    void drain() {
        set_recv_timeout(20);
        uint8_t buf[2048];
        for (int i = 0; i < 32; ++i) {
            ssize_t n = ::recvfrom(fd_, buf, sizeof(buf), 0, nullptr, nullptr);
            if (n < 0) break;
        }
        set_recv_timeout(RECV_TIMEOUT_MS);
    }

private:
    int          fd_   = -1;
    sockaddr_in  peer_ = {};
};

// Encode FT_REG_READ frame: [type, seq, addr_lo, addr_hi, nwords, 0,0,0]
void encode_reg_read(uint8_t* out, uint8_t seq, uint16_t addr, uint8_t nwords) {
    out[0] = FT_REG_READ;
    out[1] = seq;
    out[2] = static_cast<uint8_t>(addr & 0xff);
    out[3] = static_cast<uint8_t>((addr >> 8) & 0xff);
    out[4] = nwords;
    out[5] = 0;
    out[6] = 0;
    out[7] = 0;
}

// Encode FT_REG_WRITE for a single (addr, data) pair.
// Header: [type=0x01, seq, n_writes=1, 0, 0, 0, 0, 0]
// Entry : [addr_lo, addr_hi, d0, d1, d2, d3, pad_lo, pad_hi]
void encode_reg_write(uint8_t* out, uint8_t seq, uint16_t addr, uint32_t data) {
    out[0]  = FT_REG_WRITE;
    out[1]  = seq;
    out[2]  = 1;            // n_writes
    out[3]  = 0;
    out[4]  = 0;
    out[5]  = 0;
    out[6]  = 0;
    out[7]  = 0;
    out[8]  = static_cast<uint8_t>(addr & 0xff);
    out[9]  = static_cast<uint8_t>((addr >> 8) & 0xff);
    out[10] = static_cast<uint8_t>(data & 0xff);
    out[11] = static_cast<uint8_t>((data >> 8) & 0xff);
    out[12] = static_cast<uint8_t>((data >> 16) & 0xff);
    out[13] = static_cast<uint8_t>((data >> 24) & 0xff);
    out[14] = 0;
    out[15] = 0;
}

// Encode FT_DDR_WRITE: [type, seq, addr0..addr3, data...]
// 6-byte header (NOT 8 — eth_ctrl's parser slices data starting at
// offset 6; the two extra pad bytes I had earlier shifted every chunk
// by 2 bytes and silently corrupted every weight).  Matches Python's
// struct.pack("<BBI", FT_DDR_WRITE, seq, addr) + data.
void encode_ddr_write(uint8_t* out, uint8_t seq, uint32_t addr, const uint8_t* data) {
    out[0] = FT_DDR_WRITE;
    out[1] = seq;
    out[2] = static_cast<uint8_t>(addr & 0xff);
    out[3] = static_cast<uint8_t>((addr >> 8) & 0xff);
    out[4] = static_cast<uint8_t>((addr >> 16) & 0xff);
    out[5] = static_cast<uint8_t>((addr >> 24) & 0xff);
    std::memcpy(out + 6, data, CHUNK);
}

std::vector<uint32_t> reg_read(Udp& u, uint16_t addr, uint8_t nwords = 1, uint8_t seq = 1) {
    uint8_t tx[8];
    encode_reg_read(tx, seq, addr, nwords);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    for (int attempt = 0; attempt < 5; ++attempt) {
        u.send(tx, sizeof(tx));
        while (std::chrono::steady_clock::now() < deadline) {
            uint8_t rx[2048];
            size_t n = 0;
            if (!u.recv(rx, sizeof(rx), &n)) break;
            if (n < 8 || rx[0] != FT_REG_RSP || rx[1] != seq) continue;
            uint8_t got = rx[4];
            if (got > nwords) got = nwords;
            std::vector<uint32_t> v(got);
            for (uint8_t i = 0; i < got; ++i) {
                std::memcpy(&v[i], rx + 8 + 4 * i, 4);
            }
            return v;
        }
    }
    throw std::runtime_error("REG_RSP timeout for addr 0x" + std::to_string(addr));
}

void reg_write(Udp& u, uint16_t addr, uint32_t data, uint8_t seq = 1) {
    uint8_t tx[16];
    encode_reg_write(tx, seq, addr, data);
    u.send(tx, sizeof(tx));
}

// ---------------------------------------------------------------------
// Subcommand: upload
// ---------------------------------------------------------------------
// Mirrors host/ddr_write.py: stream FT_DDR_WRITE chunks back-to-back
// with a sliding window of in-flight requests.  Exit the moment every
// chunk is on the wire — eth_ctrl's ACK queue is best-effort and
// occasionally drops the trailing handful under bursty load.
struct UploadStats {
    size_t bytes_sent  = 0;
    int    retries     = 0;
    int    acked       = 0;
    double seconds     = 0.0;
};

UploadStats upload(Udp& u, const std::string& peer_ip, uint16_t peer_port,
                   const std::string& path,
                   uint32_t base = 0, int max_passes = 4,
                   int lead_limit = 1024, int poll_ms = 20,
                   double target_mbps = 2.0,
                   int backlog_limit = 32) {
    (void)max_passes;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::streamsize n_bytes = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> blob(static_cast<size_t>(n_bytes));
    if (!f.read(reinterpret_cast<char*>(blob.data()), n_bytes))
        throw std::runtime_error("read failed");

    int n_real = static_cast<int>((n_bytes + CHUNK - 1) / CHUNK);
    blob.resize(static_cast<size_t>(n_real) * CHUNK, 0);  // zero-pad tail
    std::printf("[upload] file=%s  bytes=%lld  chunks=%d  passes(max)=%d\n",
                path.c_str(), (long long)n_bytes, n_real, max_passes);

    // Sanity check: dump the bytes that would be in the first frame
    // (chunk_idx=0).  Cross-check against Python's encode_frame() for
    // the same file — they MUST be byte-identical, or we have a wire-
    // format bug masquerading as a rate issue.
    {
        uint8_t probe[6 + CHUNK];
        encode_ddr_write(probe, /*seq=*/0, /*addr=*/base, blob.data());
        std::printf("[upload] first-frame probe (70 bytes, hex):\n  hdr:");
        for (int i = 0; i < 6;  ++i) std::printf(" %02x", probe[i]);
        std::printf("\n  data:");
        for (int i = 0; i < CHUNK; ++i) {
            std::printf(" %02x", probe[6 + i]);
            if ((i & 0xf) == 0xf && i + 1 < CHUNK) std::printf("\n       ");
        }
        std::printf("\n");
    }

    // Per-chunk acked tracking (ground truth for which chunks need
    // re-sending in pass 2+).  ACKs only carry an 8-bit seq byte, so
    // every 256 chunks we cycle the seq space — we use one deque per
    // seq holding the chunk indices currently in flight with that seq,
    // and pop the front when an ACK with matching seq arrives.  As
    // long as the network mostly preserves order (true for local Gig-E
    // when not flooded), this produces a near-correct per-chunk map.
    std::vector<uint8_t> chunk_acked(n_real, 0);
    std::atomic<int> acked_total{0};

    auto read_u32 = [&](uint16_t addr) -> uint32_t {
        return reg_read(u, addr, 1, 250)[0];
    };
    uint32_t pre_rx   = read_u32(REG_DDR_WR_RX);
    uint32_t pre_done = read_u32(REG_DDR_WR_DONE);

    // Adaptive rate control: a second UDP socket polls the FPGA's
    // ddr_wr_done_count every `poll_ms` and the sender holds back if
    // global_sent gets more than `lead_limit` chunks ahead of actual
    // MIG-write progress.  Effective ceiling ≈ lead_limit/poll_ms
    // chunks/ms.  Tune via -L (lead_limit) and -P (poll_ms).
    // Steady pacing: target_mbps gives an inter-chunk period that we
    // busy-wait toward (kernel sleep_for rounds sub-ms up to scheduler
    // tick, ruining the cadence).  Mirrors Python's structurally-slow
    // one-chunk-at-a-time loop — bursts seemed to be what was tripping
    // the eth_ctrl parser→MIG handshake even at safe average rates.
    auto chunk_period = std::chrono::nanoseconds(
        static_cast<long long>(1e9 * CHUNK / (target_mbps * 1e6)));
    std::printf("[upload] steady pace: target %.2f MB/s "
                "(%.1f µs/chunk).  Burst-safety net: -L %d -P %d.\n",
                target_mbps,
                std::chrono::duration<double, std::micro>(chunk_period).count(),
                lead_limit, poll_ms);
    Udp u_poll(peer_ip, peer_port);
    std::atomic<uint32_t> fpga_done_delta{0};
    std::atomic<int>      fpga_backlog{0};        // rx_count - done_count (parser→MIG)
    std::atomic<int>      fpga_backlog_peak{0};
    std::atomic<int>      eth_ring{0};            // (nextbuf - firstbuf) & 31 (MAC→parser)
    std::atomic<int>      eth_ring_peak{0};
    std::atomic<int>      eth_ring_max{0};        // lastbuf
    std::atomic<bool>     poll_stop{false};
    std::thread poller([&]() {
        while (!poll_stop.load(std::memory_order_relaxed)) {
            try {
                // 0x018 (eth ring) + 0x019..0x01A (rx/done counters):
                // consecutive registers, fetch in one reg_read (nwords=3).
                auto v = reg_read(u_poll, REG_ETH_RING, 3, 251);
                uint32_t ring     = v[0];
                uint32_t rx_now   = v[1];
                uint32_t done_now = v[2];
                fpga_done_delta.store(done_now - pre_done, std::memory_order_relaxed);
                int backlog = static_cast<int>(rx_now - done_now);
                fpga_backlog.store(backlog, std::memory_order_relaxed);
                int peak = fpga_backlog_peak.load(std::memory_order_relaxed);
                if (backlog > peak)
                    fpga_backlog_peak.store(backlog, std::memory_order_relaxed);
                int firstbuf =  ring        & 0x1f;
                int nextbuf  = (ring >>  5) & 0x1f;
                int lastbuf  = (ring >> 10) & 0x1f;
                int depth    = (nextbuf - firstbuf) & 0x1f;
                eth_ring.store(depth, std::memory_order_relaxed);
                eth_ring_max.store(lastbuf, std::memory_order_relaxed);
                int rpeak = eth_ring_peak.load(std::memory_order_relaxed);
                if (depth > rpeak)
                    eth_ring_peak.store(depth, std::memory_order_relaxed);
            } catch (...) { /* ignore transient timeouts */ }
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
        }
    });

    auto t0 = std::chrono::steady_clock::now();
    int retries_total = 0;
    uint32_t global_sent = 0;  // total chunks sent across all passes

    for (int pass = 1; pass <= max_passes; ++pass) {
        // Build the list of chunks still needing a send for this pass.
        std::vector<int> work;
        if (pass == 1) {
            work.resize(n_real);
            for (int i = 0; i < n_real; ++i) work[i] = i;
        } else {
            for (int i = 0; i < n_real; ++i)
                if (!chunk_acked[i]) work.push_back(i);
            if (work.empty()) break;
            ++retries_total;
        }
        std::printf("[pass %d] sending %zu chunks\n", pass, work.size());
        std::fflush(stdout);

        std::deque<int> seq_q[256];
        std::mutex      seq_mu;
        std::atomic<bool> drain_stop{false};

        // ACK-drain thread.  For each ACK[seq] arriving, pop the
        // oldest chunk_index queued under that seq.  acked_total +
        // chunk_acked[] are the persistent state across passes.
        std::thread drainer([&]() {
            u.set_recv_timeout(20);
            uint8_t rx[2048];
            size_t  n = 0;
            while (!drain_stop.load(std::memory_order_relaxed)) {
                if (!u.recv(rx, sizeof(rx), &n)) continue;
                if (n < 1 || rx[0] != FT_ACK) continue;
                uint8_t s = (n >= 2) ? rx[1] : 0xff;
                if (s == 0xff) continue;
                std::lock_guard<std::mutex> g(seq_mu);
                if (seq_q[s].empty()) continue;
                int chunk_idx = seq_q[s].front();
                seq_q[s].pop_front();
                if (!chunk_acked[chunk_idx]) {
                    chunk_acked[chunk_idx] = 1;
                    acked_total.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });

        auto pass_start = std::chrono::steady_clock::now();
        auto last_print = pass_start;
        int  acked_at_start = acked_total.load(std::memory_order_relaxed);

        // Strict pacing reference time — busy-wait keeps us close to
        // the requested rate without rounding-up to scheduler ticks.
        auto next_send_at = std::chrono::steady_clock::now();
        uint8_t seq = 0;
        for (size_t i = 0; i < work.size(); ++i) {
            // Per-chunk pacing (primary throttle): wait until the
            // scheduled send time for this chunk.
            while (std::chrono::steady_clock::now() < next_send_at)
                std::this_thread::yield();
            next_send_at += chunk_period;

            // FPGA-side flow control (backstop): hold off whenever EITHER
            //   (a) cumulative send is too far ahead of MIG-write progress
            //       (lead_limit), or
            //   (b) the parser→MIG backlog (rx - done, live) is too deep —
            //       autoregress reads can starve ddr_wr_master, causing
            //       the FIFO to grow even at a "safe" average rate.
            while (true) {
                uint32_t fdone = fpga_done_delta.load(std::memory_order_relaxed);
                int      backlog = fpga_backlog.load(std::memory_order_relaxed);
                bool ahead   = (global_sent > fdone + (uint32_t)lead_limit);
                bool stuffed = (backlog > backlog_limit);
                if (!ahead && !stuffed) break;
                std::this_thread::yield();
            }

            int chunk_idx = work[i];
            uint32_t addr = base + static_cast<uint32_t>(chunk_idx) * CHUNK;
            uint8_t  tx[6 + CHUNK];
            encode_ddr_write(tx, seq, addr, blob.data() + (size_t)chunk_idx * CHUNK);
            u.send(tx, sizeof(tx));
            {
                std::lock_guard<std::mutex> g(seq_mu);
                seq_q[seq].push_back(chunk_idx);
            }
            ++seq;
            ++global_sent;

            // Light progress; no per-iter sleep on purpose.
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_print).count() > 1.0) {
                double secs = std::chrono::duration<double>(now - pass_start).count();
                int    a    = acked_total.load(std::memory_order_relaxed);
                double sent_mb   = (i * (double)CHUNK) / 1e6;
                double mbps_send = sent_mb / secs;
                double pct_sent  = 100.0 * (double)i / (double)work.size();
                double pct_acked = 100.0 * (double)a / (double)n_real;
                int    bl   = fpga_backlog.load(std::memory_order_relaxed);
                int    blpk = fpga_backlog_peak.load(std::memory_order_relaxed);
                int    er   = eth_ring.load(std::memory_order_relaxed);
                int    erpk = eth_ring_peak.load(std::memory_order_relaxed);
                int    ermx = eth_ring_max.load(std::memory_order_relaxed);
                std::printf("  pass%d sent=%zu/%zu (%5.1f%%, %4.1f MB) "
                            "acked=%d (%5.1f%%)  %5.2f MB/s  "
                            "parser->mig=%d/peak%d  eth_ring=%d/peak%d/max%d\n",
                            pass, i, work.size(), pct_sent, sent_mb,
                            a, pct_acked, mbps_send,
                            bl, blpk, er, erpk, ermx);
                std::fflush(stdout);
                last_print = now;
            }
        }

        // Tail: wait for ACKs to drain (cap so we don't spin forever).
        std::this_thread::sleep_for(std::chrono::seconds(1));
        for (int waited = 0; waited < 30; ++waited) {
            int prev = acked_total.load(std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (acked_total.load(std::memory_order_relaxed) == prev) break;
        }

        drain_stop.store(true, std::memory_order_relaxed);
        drainer.join();

        int newly_acked = acked_total.load(std::memory_order_relaxed) - acked_at_start;
        int still_missing = 0;
        for (int i = 0; i < n_real; ++i) if (!chunk_acked[i]) ++still_missing;

        uint32_t rx_now   = read_u32(REG_DDR_WR_RX);
        uint32_t done_now = read_u32(REG_DDR_WR_DONE);
        uint32_t rx_delta   = rx_now   - pre_rx;
        uint32_t done_delta = done_now - pre_done;
        // global_sent counts frames the host has put on the wire so
        // far (cumulative across passes).  rx_delta counts frames the
        // FPGA parser accepted, so global_sent - rx_delta is what
        // never made it — frame-CRC drops at the eth MAC, NIC queue
        // overflows, switch drops, etc.  Those chunks won't have
        // ACKs either, and pass 2+ will catch them.
        uint32_t eth_drops_total = (global_sent > rx_delta)
                                 ? (global_sent - rx_delta) : 0;
        std::printf("[pass %d] +acked=%d  missing=%d  "
                    "fpga rx_delta=%u done_delta=%u  "
                    "parser->mig=%d/peak%d  eth_ring=peak%d/max%d  "
                    "eth_drops(cumulative)=%u\n",
                    pass, newly_acked, still_missing,
                    rx_delta, done_delta,
                    int(rx_now - done_now),
                    fpga_backlog_peak.load(std::memory_order_relaxed),
                    eth_ring_peak.load(std::memory_order_relaxed),
                    eth_ring_max.load(std::memory_order_relaxed),
                    eth_drops_total);
        std::fflush(stdout);

        if (still_missing == 0) break;
        // Note: FPGA done_count == n_real doesn't *prove* per-chunk
        // correctness — UDP has only a weak 16-bit checksum and the
        // parser doesn't verify it, so a flipped address bit could
        // route a chunk to the wrong DDR3 offset with done_count
        // still incrementing.  And unack'd chunks aren't necessarily
        // "ACK-path lost" — they could be frame-CRC drops that never
        // reached the FPGA at all.  Always run pass 2+ on the unack'd
        // set; idempotent for chunks that did land correctly.
        (void)done_delta;  // informational only
    }

    poll_stop.store(true, std::memory_order_relaxed);
    poller.join();

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    int    a    = acked_total.load(std::memory_order_relaxed);
    double mbps = (a * (double)CHUNK) / secs / 1e6;
    int    missing = 0;
    for (int i = 0; i < n_real; ++i) if (!chunk_acked[i]) ++missing;
    std::printf("[upload] done in %.2fs, %d/%d acked (%.2f MB/s), passes=%d, missing=%d\n",
                secs, a, n_real, mbps, retries_total + 1, missing);
    u.set_recv_timeout(RECV_TIMEOUT_MS);
    return {(size_t)n_bytes, retries_total, a, secs};
}

// ---------------------------------------------------------------------
// Subcommand: verify
// ---------------------------------------------------------------------
// The diagnostic snapshot in the BFP layer captures the FIRST AXI
// read the streamer issues after restart: dbg_first_araddr (0x013) +
// dbg_first_rdata (0x014..0x023, 16 words = 512 bits = 64 bytes).
// We pulse restart, wait for dbg_first_r_seen, then compare the 64
// bytes against the file at the captured offset.
bool verify_first_read(Udp& u, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    u.drain();
    auto status0 = reg_read(u, REG_DBG_STATUS, 1, 30)[0];
    std::printf("[verify] pre-restart  0x012=0x%08x  (calib=%d, ar_seen=%d, r_seen=%d)\n",
                status0, status0 & 1, (status0 >> 1) & 1, (status0 >> 2) & 1);
    if ((status0 & 1) == 0) {
        std::printf("[verify] init_calib_complete=0 — DDR3 not ready, aborting\n");
        return false;
    }

    // Pulse restart so dbg_first_* re-arm with the next streamer cycle.
    reg_write(u, REG_RESTART, 1, 31);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Poll until the streamer has issued and received its first AR/R.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    uint32_t status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        status = reg_read(u, REG_DBG_STATUS, 1, 32)[0];
        if (((status >> 1) & 1) && ((status >> 2) & 1)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (((status >> 1) & 1) == 0 || ((status >> 2) & 1) == 0) {
        std::printf("[verify] streamer never produced a captured AR/R within 10s "
                    "(0x012=0x%08x)\n", status);
        return false;
    }

    uint32_t ar_addr = reg_read(u, REG_DBG_ARADDR, 1, 33)[0];
    auto rdata_words = reg_read(u, REG_DBG_RDATA0, 16, 34);
    if (rdata_words.size() != 16) {
        std::printf("[verify] expected 16 rdata words, got %zu\n", rdata_words.size());
        return false;
    }

    f.seekg(ar_addr);
    uint8_t expected[64];
    if (!f.read(reinterpret_cast<char*>(expected), 64)) {
        std::printf("[verify] file too short to read 64 B at offset 0x%08x\n", ar_addr);
        return false;
    }
    uint8_t got[64];
    for (int i = 0; i < 16; ++i) std::memcpy(got + 4 * i, &rdata_words[i], 4);

    bool ok = std::memcmp(expected, got, 64) == 0;
    std::printf("[verify] first AR addr = 0x%08x  →  %s\n",
                ar_addr, ok ? "MATCH" : "MISMATCH");
    if (!ok) {
        auto hex = [](uint8_t b) {
            char s[3]; std::snprintf(s, sizeof(s), "%02x", b); return std::string(s);
        };
        std::printf("  expected:");
        for (int i = 0; i < 64; ++i) std::printf(" %s", hex(expected[i]).c_str());
        std::printf("\n  got     :");
        for (int i = 0; i < 64; ++i) std::printf(" %s", hex(got[i]).c_str());
        std::printf("\n");
    }
    return ok;
}

// ---------------------------------------------------------------------
// Subcommand: restart + poll done
// ---------------------------------------------------------------------
// REG_DONE = {30'd0, lay_done_latched, lay_done}.  `lay_done` is a
// 1-cycle pulse easy to miss; rely on the latched bit (bit 1).
bool restart_and_wait_done(Udp& u, double timeout_s = 60.0) {
    u.drain();
    reg_write(u, REG_RESTART, 1, 40);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(static_cast<long long>(timeout_s * 1000));
    uint32_t last = ~0u;
    while (std::chrono::steady_clock::now() < deadline) {
        uint32_t v = reg_read(u, REG_DONE, 1, 41)[0];
        if (v != last) {
            std::printf("  REG_DONE = 0x%08x  (latched=%d, live=%d)\n",
                        v, (v >> 1) & 1, v & 1);
            last = v;
        }
        if (v & 0x3) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return false;
}

// ---------------------------------------------------------------------
// Vocab loader + token decoder (no torch, no transformers).
// ---------------------------------------------------------------------
// Binary format produced by host/gen_bfp_vocab_bin.py:
//   char[4]  magic "BFPV"
//   uint32   version (=1)
//   uint32   n_tokens
//   uint32   blob_size
//   { uint32 offset; uint32 length; } [n_tokens]
//   uint8    blob[blob_size]
struct Vocab {
    std::vector<std::pair<uint32_t, uint32_t>> entries;  // (offset, length)
    std::vector<uint8_t>                       blob;

    static Vocab load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open vocab " + path);
        char magic[4];
        if (!f.read(magic, 4) || std::memcmp(magic, "BFPV", 4) != 0)
            throw std::runtime_error("bad vocab magic in " + path);
        uint32_t version = 0, n = 0, blob_size = 0;
        f.read(reinterpret_cast<char*>(&version), 4);
        f.read(reinterpret_cast<char*>(&n), 4);
        f.read(reinterpret_cast<char*>(&blob_size), 4);
        if (version != 1)
            throw std::runtime_error("unsupported vocab version "
                                     + std::to_string(version));
        Vocab v;
        v.entries.resize(n);
        for (uint32_t i = 0; i < n; ++i)
            f.read(reinterpret_cast<char*>(&v.entries[i]), sizeof(v.entries[i]));
        v.blob.resize(blob_size);
        if (!f.read(reinterpret_cast<char*>(v.blob.data()), blob_size))
            throw std::runtime_error("vocab truncated");
        std::fprintf(stderr, "[vocab] loaded %u tokens, %u byte blob (%s)\n",
                     n, blob_size, path.c_str());
        return v;
    }

    // Decode one token id to its raw UTF-8 byte slice.
    std::string_view bytes_for(uint32_t id) const {
        if (id >= entries.size()) return {};
        auto [off, len] = entries[id];
        if (off + len > blob.size()) return {};
        return std::string_view(reinterpret_cast<const char*>(blob.data() + off), len);
    }
};

std::string decode_tokens(const Vocab& v, const std::vector<uint16_t>& tokens) {
    std::string out;
    for (auto t : tokens) {
        auto sv = v.bytes_for(t);
        out.append(sv.data(), sv.size());
    }
    return out;
}

// ---------------------------------------------------------------------
// Subcommand: verify-sampled
// ---------------------------------------------------------------------
// Sample N random AR/R pairs from the BFP autoregress's natural DDR3
// read traffic (via the re-armable snapshot at 0x048 / 0x012 /
// 0x013 / 0x014..0x023) and compare each 64-byte burst beat against
// the local weight image.  The autoregress is in a perpetual loop, so
// each arm catches whatever read it issues next — over many samples
// we cover the bulk of the weight address space.
//
// IMPORTANT: the current snapshot captures the first AR and first R
// AFTER arm independently; the streamer uses bursts up to 256 beats
// with arid=0, so a captured (araddr, rdata) pair is only correlated
// when we arm during an idle window between bursts.  Mid-burst arms
// produce desync'd pairs that LOOK like mismatches.  Run the verify
// against Python-uploaded DDR3 first to measure the noise floor, then
// against C++-uploaded DDR3 — a significant divergence is real
// corruption; a similar mismatch rate is just AR/R desync.
struct VerifySample {
    uint32_t addr;
    std::array<uint8_t, 64> rdata;
    bool matches;
};

int verify_sampled(Udp& u, const std::string& path, int n_samples) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::streamsize n_bytes = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> blob(static_cast<size_t>(n_bytes));
    if (!f.read(reinterpret_cast<char*>(blob.data()), n_bytes))
        throw std::runtime_error("read failed");

    std::printf("[verify] %d samples against %s (%lld bytes)\n",
                n_samples, path.c_str(), (long long)n_bytes);

    int matched = 0, mismatched = 0, oob = 0, timeouts = 0;
    std::vector<VerifySample> mismatch_log;

    uint8_t seq = 100;
    for (int i = 0; i < n_samples; ++i) {
        // Arm: any write to 0x048 toggles the arm flop.
        reg_write(u, 0x048, 1, seq++);
        // Poll status until ar_seen AND r_seen both set.
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::milliseconds(500);
        uint32_t status = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            status = reg_read(u, REG_DBG_STATUS, 1, seq++)[0];
            if (((status >> 1) & 1) && ((status >> 2) & 1)) break;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        if (!(((status >> 1) & 1) && ((status >> 2) & 1))) {
            ++timeouts;
            continue;
        }
        // Fetch addr + 16 words of rdata (0x013, 0x014..0x023).
        uint32_t addr = reg_read(u, REG_DBG_ARADDR, 1, seq++)[0] & 0x3fffffff;
        auto words = reg_read(u, REG_DBG_RDATA0, 16, seq++);
        if (words.size() != 16) { ++timeouts; continue; }
        std::array<uint8_t, 64> got{};
        for (int j = 0; j < 16; ++j)
            std::memcpy(&got[j * 4], &words[j], 4);

        if ((size_t)addr + 64 > blob.size()) { ++oob; continue; }
        // Trustworthy byte ranges: regmap addrs 0x014..0x016 + 0x01D..0x023
        // are pure dbg_first_rdata; 0x017..0x01C are other diagnostic
        // registers (ila / eth_ring / ddr_wr_*) and steal slots within
        // the rdata window.  So bytes 0..11 (words 0..2) and bytes 36..63
        // (words 9..15) are clean; bytes 12..35 (words 3..8) are noise.
        // Total trustworthy: 12 + 28 = 40 of 64 bytes per sample.
        bool match_lo = std::memcmp(&got[0],  blob.data() + addr,      12) == 0;
        bool match_hi = std::memcmp(&got[36], blob.data() + addr + 36, 28) == 0;
        bool match = match_lo && match_hi;
        if (match) ++matched; else {
            ++mismatched;
            if (mismatch_log.size() < 8)
                mismatch_log.push_back({addr, got, false});
        }

        if ((i + 1) % 100 == 0) {
            std::printf("  %5d/%-5d  matched=%d  mismatched=%d  "
                        "timeouts=%d  oob=%d\r",
                        i + 1, n_samples, matched, mismatched, timeouts, oob);
            std::fflush(stdout);
        }
    }
    std::printf("\n[verify] done: %d/%d matched (%.1f%%), "
                "%d mismatched, %d timeouts, %d out-of-range\n",
                matched, n_samples, 100.0 * matched / std::max(1, n_samples),
                mismatched, timeouts, oob);

    if (!mismatch_log.empty()) {
        std::printf("[verify] first %zu mismatches (showing trustworthy bytes only):\n",
                    mismatch_log.size());
        for (auto& s : mismatch_log) {
            std::printf("  addr 0x%08x:\n    bytes 0..11  expected", s.addr);
            for (int j = 0; j < 12; ++j) std::printf(" %02x", blob[s.addr + j]);
            std::printf("\n                 got     ");
            for (int j = 0; j < 12; ++j) std::printf(" %02x", s.rdata[j]);
            std::printf("\n    bytes 36..63 expected");
            for (int j = 36; j < 64; ++j) std::printf(" %02x", blob[s.addr + j]);
            std::printf("\n                 got     ");
            for (int j = 36; j < 64; ++j) std::printf(" %02x", s.rdata[j]);
            std::printf("\n");
        }
    }
    return mismatched == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------
// Subcommand: read-crc
// ---------------------------------------------------------------------
// Reads the FPGA's accumulated CRC32 over every 512-bit AXI rdata beat
// the BFP master has seen since the last restart pulse.  After a
// single-shot autoregress run completes (has_run=1), the value is
// stable and represents a hash of *exactly the data the LLM consumed*.
//
// Compare across runs:
//   - Same CRC after Python vs C++ upload ⇒ both uploads delivered
//     byte-identical data to the autoregress's view; any token
//     divergence is downstream of DDR3.
//   - Different CRC ⇒ DDR3 contents really differ between the two
//     upload paths even where verify-sampled couldn't see it.
void print_crc(Udp& u) {
    uint32_t status = reg_read(u, REG_HAS_RUN, 1, 60)[0];
    uint32_t crc    = reg_read(u, REG_RDATA_CRC, 1, 61)[0];
    std::printf("has_run = %u\n", status & 1);
    std::printf("CRC32   = 0x%08x\n", crc);
}

// ---------------------------------------------------------------------
// Subcommand: tokens
// ---------------------------------------------------------------------
std::vector<uint16_t> fetch_tokens(Udp& u, int n_steps) {
    int nwords = (n_steps * 16 + 31) / 32;
    auto words = reg_read(u, REG_RESULT, static_cast<uint8_t>(nwords), 50);
    if ((int)words.size() != nwords) {
        throw std::runtime_error("expected " + std::to_string(nwords)
                                 + " result words, got "
                                 + std::to_string(words.size()));
    }
    std::vector<uint16_t> tokens;
    tokens.reserve(n_steps);
    uint64_t accum = 0;
    int      bits  = 0;
    for (auto w : words) {
        accum |= (uint64_t)w << bits;
        bits += 32;
        while (bits >= 16 && (int)tokens.size() < n_steps) {
            tokens.push_back(static_cast<uint16_t>(accum & 0xffff));
            accum >>= 16;
            bits  -= 16;
        }
    }
    return tokens;
}

void print_tokens(Udp& u, const Vocab* vocab, int n_steps) {
    auto tokens = fetch_tokens(u, n_steps);
    std::printf("RTL_TOKENS:");
    for (auto t : tokens) std::printf(" %u", t);
    std::printf("\n");
    if (vocab) {
        std::string text = decode_tokens(*vocab, tokens);
        std::printf("DECODED:\n%s\n", text.c_str());
    }
}

// ---------------------------------------------------------------------

void usage() {
    std::fprintf(stderr,
        "usage: bfp_client [-p IP:PORT] [-v VOCAB_BIN] [-m MAC]\n"
        "                  [-L LEAD_LIMIT] [-P POLL_MS] <cmd> [args]\n"
        "  -r  target upload rate in MB/s, steady-paced per-chunk    (default 2.0)\n"
        "      Mirrors Python's structurally-slow loop — burst sends seem\n"
        "       to race the FPGA's eth_ctrl→MIG handshake even at safe averages.\n"
        "  -L  chunks the sender may run ahead of FPGA done_count    (default 1024)\n"
        "  -P  ms between done_count polls                           (default 20)\n"
        "  -B  parser→MIG backlog ceiling = rx_count - done_count    (default 32)\n"
        "      Sender pauses if backlog grows past this — catches MIG read/write\n"
        "       contention with autoregress that average-rate pacing misses.\n"
        "      (-L / -P / -B are backstops; primary throttle is -r.)\n"
        "  -n  total tokens to fetch = NPROMPT+NGEN (default 19, max 64)\n"
        "  discover           look up the FPGA's IP from its MAC (default %s)\n"
        "                     in /proc/net/arp; seed the cache by pinging the\n"
        "                     /24 around -p's IP (default %s) if not found\n"
        "  upload  <bin>      bulk-write the weight image to DDR3 base 0\n"
        "  verify  <bin>      restart, capture first AR, compare 64 B\n"
        "  verify-sampled <bin> [-N count]\n"
        "                     sample N (default 1000) random reads from the\n"
        "                     autoregress's natural DDR3 traffic and compare\n"
        "                     each to <bin>.  Note: bursts cause AR/R desync\n"
        "                     mismatches; measure noise floor first.\n"
        "  restart            pulse restart, poll until done (or 60 s)\n"
        "  tokens             read 10 result words from 0x1D0\n"
        "                     (with -v: also print decoded text)\n"
        "  read-crc           read FPGA CRC32 of every AXI rdata beat the\n"
        "                     BFP master has seen since last restart\n"
        "  all     <bin>      upload → verify → restart → tokens\n",
        FPGA_MAC, DEFAULT_PEER_IP);
}

}  // namespace

int main(int argc, char** argv) {
    std::string peer_ip    = DEFAULT_PEER_IP;
    uint16_t    peer_port  = DEFAULT_PEER_PORT;
    bool        peer_explicit = false;
    std::string vocab_path;
    std::string fpga_mac   = FPGA_MAC;
    int         lead_limit = 1024;     // backstop only — primary pace is -r
    int         poll_ms    = 20;
    double      target_mbps = 2.0;     // matches Python's measured 1.96 MB/s
    int         backlog_limit = 32;    // pause sender when parser→MIG backlog > this
    int         n_steps    = N_STEPS_DEFAULT;
    int         verify_count = 1000;   // -N for verify-sampled

    int argi = 1;
    while (argi < argc && argv[argi][0] == '-') {
        std::string a = argv[argi];
        if (a == "-p" && argi + 1 < argc) {
            std::string s = argv[++argi];
            auto colon = s.find(':');
            if (colon == std::string::npos) { usage(); return 2; }
            peer_ip   = s.substr(0, colon);
            peer_port = static_cast<uint16_t>(std::stoi(s.substr(colon + 1)));
            peer_explicit = true;
        } else if (a == "-v" && argi + 1 < argc) {
            vocab_path = argv[++argi];
        } else if (a == "-m" && argi + 1 < argc) {
            fpga_mac = argv[++argi];
        } else if (a == "-L" && argi + 1 < argc) {
            lead_limit = std::atoi(argv[++argi]);
        } else if (a == "-P" && argi + 1 < argc) {
            poll_ms = std::atoi(argv[++argi]);
        } else if (a == "-r" && argi + 1 < argc) {
            target_mbps = std::atof(argv[++argi]);
            if (target_mbps <= 0) {
                std::fprintf(stderr, "-r rate must be > 0 MB/s\n");
                return 2;
            }
        } else if (a == "-B" && argi + 1 < argc) {
            backlog_limit = std::atoi(argv[++argi]);
        } else if (a == "-N" && argi + 1 < argc) {
            verify_count = std::atoi(argv[++argi]);
        } else if (a == "-n" && argi + 1 < argc) {
            n_steps = std::atoi(argv[++argi]);
            if (n_steps < 1 || n_steps > N_STEPS_MAX) {
                std::fprintf(stderr, "n_steps must be 1..%d\n", N_STEPS_MAX);
                return 2;
            }
        } else if (a == "-h" || a == "--help") {
            usage(); return 0;
        } else {
            usage(); return 2;
        }
        ++argi;
    }
    if (argi >= argc) { usage(); return 2; }

    std::unique_ptr<Vocab> vocab;
    if (!vocab_path.empty())
        vocab = std::make_unique<Vocab>(Vocab::load(vocab_path));

    std::string cmd = argv[argi++];
    try {
        if (cmd == "discover") {
            std::string ip = lookup_ip_for_mac(fpga_mac);
            if (ip.empty()) {
                std::fprintf(stderr,
                    "[discover] %s not in /proc/net/arp; seeding subnet around %s …\n",
                    fpga_mac.c_str(), peer_ip.c_str());
                seed_arp_subnet(peer_ip);
                ip = lookup_ip_for_mac(fpga_mac);
            }
            if (ip.empty()) {
                std::fprintf(stderr, "[discover] no host with MAC %s found\n",
                             fpga_mac.c_str());
                return 1;
            }
            std::printf("%s\n", ip.c_str());
            return 0;
        }

        // For commands that hit the FPGA, auto-discover the peer IP from
        // the hardcoded MAC unless the user pinned -p explicitly.  Keeps
        // `bfp_client all <bin>` self-contained (no shell composition).
        if (!peer_explicit) {
            std::string ip = lookup_ip_for_mac(fpga_mac);
            if (ip.empty()) {
                std::fprintf(stderr,
                    "[discover] %s not in /proc/net/arp; seeding /24 around %s …\n",
                    fpga_mac.c_str(), peer_ip.c_str());
                seed_arp_subnet(peer_ip);
                ip = lookup_ip_for_mac(fpga_mac);
            }
            if (!ip.empty() && ip != peer_ip) {
                std::fprintf(stderr, "[discover] FPGA at %s (MAC %s)\n",
                             ip.c_str(), fpga_mac.c_str());
                peer_ip = ip;
            } else if (ip.empty()) {
                std::fprintf(stderr,
                    "[discover] no MAC %s in ARP cache; trying default %s\n",
                    fpga_mac.c_str(), peer_ip.c_str());
            }
        }

        Udp u(peer_ip, peer_port);

        if (cmd == "upload") {
            if (argi >= argc) { usage(); return 2; }
            upload(u, peer_ip, peer_port, argv[argi], 0, 4,
                   lead_limit, poll_ms, target_mbps, backlog_limit);
            return 0;
        }
        if (cmd == "verify") {
            if (argi >= argc) { usage(); return 2; }
            return verify_first_read(u, argv[argi]) ? 0 : 1;
        }
        if (cmd == "verify-sampled") {
            if (argi >= argc) { usage(); return 2; }
            return verify_sampled(u, argv[argi], verify_count);
        }
        if (cmd == "restart") {
            bool ok = restart_and_wait_done(u);
            std::printf("[restart] %s\n", ok ? "done" : "TIMEOUT");
            return ok ? 0 : 1;
        }
        if (cmd == "tokens") {
            print_tokens(u, vocab.get(), n_steps);
            return 0;
        }
        if (cmd == "read-crc") {
            print_crc(u);
            return 0;
        }
        if (cmd == "all") {
            if (argi >= argc) { usage(); return 2; }
            std::string bin = argv[argi];
            auto bv = reg_read(u, REG_BUILD_VERSION, 1, 1)[0];
            std::printf("[all] FPGA at %s:%u  BUILD_VERSION = 0x%08x\n",
                        peer_ip.c_str(), peer_port, bv);
            upload(u, peer_ip, peer_port, bin, 0, 4,
                   lead_limit, poll_ms, target_mbps, backlog_limit);
            // Settle: the BFP autoregress auto-restarts in a perpetual
            // loop (bfp_start_r self-fires whenever lay_done is low), so
            // it's almost certainly mid-run when upload finishes — its
            // KV cache and intermediate state reflect partially-uploaded
            // data.  Give it a moment of clean runs on the final DDR3
            // contents before pulsing restart, so the run we capture
            // hasn't seen any of the in-progress upload.
            std::printf("[all] settling 3 s before restart pulse …\n");
            std::this_thread::sleep_for(std::chrono::seconds(3));
            if (!restart_and_wait_done(u)) {
                std::printf("[all] restart-wait timed out\n");
                return 1;
            }
            print_tokens(u, vocab.get(), n_steps);
            print_crc(u);
            return 0;
        }

        usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
