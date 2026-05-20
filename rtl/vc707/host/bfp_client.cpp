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
#include <sstream>
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
constexpr uint16_t REG_BRAM_TARGET   = 0x060;   // write: {inc[31], 8'd0, kind[22:18], addr[17:0]}
constexpr uint16_t REG_BRAM_DATA     = 0x061;   // write: {16'd0, data[15:0]} — pulses BRAM write
constexpr uint16_t REG_BRAM_READ     = 0x062;   // read: {16'd0, BRAM[kind, addr]} — port-A readback
constexpr uint16_t REG_N_PROMPT      = 0x063;   // r/w: active prompt length (1..NPROMPT_MAX)
constexpr uint16_t REG_BUILD_VERSION = 0x10F;
constexpr uint16_t REG_RESULT        = 0x1D0;
constexpr uint16_t REG_DONE          = 0x1F0;   // {30'd0, lay_done_latched, lay_done}
constexpr uint16_t REG_RESTART       = 0x1F1;   // bit 0: write 1 to pulse restart

// With NPROMPT_MAX=48 + NGEN=15 the result-tokens buffer is 63 slots.
// The host reads all of them and trims based on the active prompt
// length reported by the FPGA at 0x063.  Earlier bitstreams used
// NPROMPT=4 / NGEN=15 = 19 slots; the new default reads more but
// short bitstreams just have zero-padded tail words.
constexpr int N_STEPS_DEFAULT = 63;
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
    //
    // target_mbps==0 selects probe-then-commit: a calibration ramp
    // before the bulk upload picks the rate that minimises projected
    // total time = pass1_time(r) * (1 + 2*loss_rate(r)) + retry_pass_overhead.
    // Final chunk_period is set after the probe.
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

    uint32_t global_sent = 0;  // total chunks sent across all passes (incl. probe)

    // -----------------------------------------------------------------
    // Probe-then-commit (target_mbps==0).
    //
    // Sends short bursts of PROBE_CHUNKS at each candidate rate over
    // a disjoint segment of the file, measures ACK-loss, and picks the
    // rate that minimises projected wall-clock for the bulk upload:
    //
    //   total(r) ≈ file_chunks * CHUNK / (r * 1e6) * (1 + 2*loss(r))
    //            + RETRY_PASS_OVERHEAD * (loss(r) > 0.01 ? 2 : 1)
    //
    // Probe-acked chunks count toward chunk_acked[] so pass 1 doesn't
    // resend them — the probe doubles as real work.  If a rate's loss
    // exceeds HARD_LOSS_CAP we stop ramping (higher rates almost
    // certainly fare worse on the same link).
    // -----------------------------------------------------------------
    if (target_mbps <= 0.0) {
        const std::vector<double> CANDIDATES   = {2.0, 4.0, 6.0, 8.0, 10.0};
        constexpr int             PROBE_CHUNKS = 32768;   // 2 MB per burst
        constexpr double          HARD_LOSS_CAP = 0.10;   // stop ramping above this
        constexpr double          RETRY_PASS_OVERHEAD_S = 3.0;

        std::printf("[probe] auto-rate probe: candidates");
        for (auto r : CANDIDATES) std::printf(" %.0f", r);
        std::printf(" MB/s × %d chunks (%.2f MB each)\n",
                    PROBE_CHUNKS, PROBE_CHUNKS * (double)CHUNK / 1e6);
        std::fflush(stdout);

        double best_rate = CANDIDATES.front();
        double best_cost = 1e30;
        int    probe_base = 0;   // chunk-index cursor for disjoint probe regions

        // One shared drainer covers all probe iterations.
        std::deque<int>   probe_seq_q[256];
        std::mutex        probe_seq_mu;
        std::atomic<bool> probe_drain_stop{false};
        std::thread probe_drainer([&]() {
            u.set_recv_timeout(20);
            uint8_t rx[2048]; size_t n = 0;
            while (!probe_drain_stop.load(std::memory_order_relaxed)) {
                if (!u.recv(rx, sizeof(rx), &n)) continue;
                if (n < 1 || rx[0] != FT_ACK) continue;
                uint8_t s = (n >= 2) ? rx[1] : 0xff;
                if (s == 0xff) continue;
                std::lock_guard<std::mutex> g(probe_seq_mu);
                if (probe_seq_q[s].empty()) continue;
                int chunk_idx = probe_seq_q[s].front();
                probe_seq_q[s].pop_front();
                if (!chunk_acked[chunk_idx]) {
                    chunk_acked[chunk_idx] = 1;
                    acked_total.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });

        for (size_t ci = 0; ci < CANDIDATES.size(); ++ci) {
            double r = CANDIDATES[ci];
            int  burst_chunks = std::min(PROBE_CHUNKS, n_real - probe_base);
            if (burst_chunks <= 0) break;   // file too small for further probes

            auto probe_period = std::chrono::nanoseconds(
                static_cast<long long>(1e9 * CHUNK / (r * 1e6)));

            int acked_before = acked_total.load(std::memory_order_relaxed);
            auto burst_t0 = std::chrono::steady_clock::now();
            auto next_send_at = burst_t0;
            uint8_t seq = 0;
            for (int i = 0; i < burst_chunks; ++i) {
                while (std::chrono::steady_clock::now() < next_send_at)
                    std::this_thread::yield();
                next_send_at += probe_period;
                // Same backstops as main pass.  These will stretch
                // send_secs above the nominal r when the wire / MIG
                // path can't keep up — and that stretch is exactly
                // what the cost model needs to see.
                while (true) {
                    uint32_t fdone   = fpga_done_delta.load(std::memory_order_relaxed);
                    int      backlog = fpga_backlog.load(std::memory_order_relaxed);
                    bool ahead   = (global_sent > fdone + (uint32_t)lead_limit);
                    bool stuffed = (backlog > backlog_limit);
                    if (!ahead && !stuffed) break;
                    std::this_thread::yield();
                }
                int chunk_idx = probe_base + i;
                uint32_t addr = base + static_cast<uint32_t>(chunk_idx) * CHUNK;
                uint8_t tx[6 + CHUNK];
                encode_ddr_write(tx, seq, addr, blob.data() + (size_t)chunk_idx * CHUNK);
                u.send(tx, sizeof(tx));
                {
                    std::lock_guard<std::mutex> g(probe_seq_mu);
                    probe_seq_q[seq].push_back(chunk_idx);
                }
                ++seq;
                ++global_sent;
            }
            // Capture send-only wall-clock BEFORE the drain — this is
            // the actual achievable throughput at target r (backstops
            // included).  Earlier versions divided by send+drain time,
            // which made all measurements look identical because the
            // 500ms drain dominated the short burst.
            auto burst_send_end = std::chrono::steady_clock::now();
            double send_secs = std::chrono::duration<double>(
                burst_send_end - burst_t0).count();

            // Adaptive drain: keep watching acked_total until it stops
            // growing (200ms quiet window) so loss reflects real drops
            // rather than slow-arriving ACKs we cut off too early.
            int last_acked = acked_total.load(std::memory_order_relaxed);
            int stable_iters = 0;
            while (stable_iters < 4) {     // 4 × 50ms = 200ms quiet
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                int now = acked_total.load(std::memory_order_relaxed);
                if (now == last_acked) ++stable_iters;
                else { stable_iters = 0; last_acked = now; }
            }

            int    acked_now    = acked_total.load(std::memory_order_relaxed);
            int    acked_burst  = acked_now - acked_before;
            double loss_rate    = 1.0 - (double)acked_burst / (double)burst_chunks;
            if (loss_rate < 0) loss_rate = 0;   // clip rounding
            double send_mbps = burst_chunks * (double)CHUNK / 1e6 / send_secs;

            // Project total wall-clock using the ACTUAL achievable send
            // rate (send_mbps), not the nominal target r — they diverge
            // once the wire saturates.
            double bulk_secs    = (double)n_real * CHUNK / 1e6 / send_mbps;
            double retry_factor = 1.0 + 2.0 * loss_rate;
            double overhead_s   = (loss_rate > 0.01) ? 2*RETRY_PASS_OVERHEAD_S
                                                    : RETRY_PASS_OVERHEAD_S;
            double cost = bulk_secs * retry_factor + overhead_s;

            std::printf("[probe] r=%4.1f MB/s  sent=%d  acked=%d  loss=%.3f%%  "
                        "send=%.2f MB/s  cost=%.1fs\n",
                        r, burst_chunks, acked_burst, loss_rate * 100.0,
                        send_mbps, cost);
            std::fflush(stdout);

            if (cost < best_cost) {
                best_cost = cost;
                best_rate = r;
            }
            probe_base += burst_chunks;
            if (loss_rate > HARD_LOSS_CAP) {
                std::printf("[probe] loss exceeds %.0f%% — stopping ramp\n",
                            HARD_LOSS_CAP * 100.0);
                break;
            }
        }

        probe_drain_stop.store(true, std::memory_order_relaxed);
        probe_drainer.join();
        target_mbps = best_rate;
        std::printf("[probe] committed target_mbps = %.1f "
                    "(projected upload cost %.1f s)\n",
                    target_mbps, best_cost);
        std::fflush(stdout);
    }

    auto chunk_period = std::chrono::nanoseconds(
        static_cast<long long>(1e9 * CHUNK / (target_mbps * 1e6)));
    std::printf("[upload] steady pace: target %.2f MB/s "
                "(%.1f µs/chunk).  Burst-safety net: -L %d -P %d.\n",
                target_mbps,
                std::chrono::duration<double, std::micro>(chunk_period).count(),
                lead_limit, poll_ms);

    auto t0 = std::chrono::steady_clock::now();
    int retries_total = 0;

    for (int pass = 1; pass <= max_passes; ++pass) {
        // Build the list of chunks still needing a send for this pass.
        // Pass 1 skips chunks that the probe already got ACKed — probe
        // doubles as real work, no need to resend.
        std::vector<int> work;
        if (pass == 1) {
            for (int i = 0; i < n_real; ++i)
                if (!chunk_acked[i]) work.push_back(i);
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

// Decode one UTF-8 codepoint starting at s[i]; advance i.  On invalid
// input, returns the raw byte as codepoint and advances by 1.
static uint32_t utf8_next_cp(const std::string& s, size_t& i) {
    unsigned char b = static_cast<unsigned char>(s[i]);
    if (b < 0x80) { ++i; return b; }
    int n; uint32_t cp;
    if      ((b & 0xe0) == 0xc0) { n = 2; cp = b & 0x1f; }
    else if ((b & 0xf0) == 0xe0) { n = 3; cp = b & 0x0f; }
    else if ((b & 0xf8) == 0xf0) { n = 4; cp = b & 0x07; }
    else                         { ++i; return b; }
    if (i + n > s.size())        { ++i; return b; }
    for (int k = 1; k < n; ++k) {
        unsigned char c = static_cast<unsigned char>(s[i + k]);
        if ((c & 0xc0) != 0x80)  { ++i; return b; }
        cp = (cp << 6) | (c & 0x3f);
    }
    i += n;
    return cp;
}

static void cp_to_utf8(uint32_t cp, std::string& out) {
    if      (cp < 0x80)    { out.push_back(static_cast<char>(cp)); }
    else if (cp < 0x800)   { out.push_back(0xc0 | (cp >> 6));
                             out.push_back(0x80 | (cp & 0x3f)); }
    else if (cp < 0x10000) { out.push_back(0xe0 | (cp >> 12));
                             out.push_back(0x80 | ((cp >> 6) & 0x3f));
                             out.push_back(0x80 | (cp & 0x3f)); }
    else                   { out.push_back(0xf0 | (cp >> 18));
                             out.push_back(0x80 | ((cp >> 12) & 0x3f));
                             out.push_back(0x80 | ((cp >> 6) & 0x3f));
                             out.push_back(0x80 | (cp & 0x3f)); }
}

// If `cp` is a codepoint reachable in Windows-1252 single-byte encoding,
// return the byte (0x00-0xFF).  Else -1.  ASCII passes through; latin-1
// supplement passes through; the 27 Win1252 specials in 0x80-0x9F (€, …,
// smart-quotes, em-dash, etc.) get mapped back to their single byte.
static int cp_to_win1252_byte(uint32_t cp) {
    if (cp < 0x80 || (cp >= 0xA0 && cp <= 0xFF)) return static_cast<int>(cp);
    switch (cp) {
        case 0x20AC: return 0x80;  case 0x201A: return 0x82;
        case 0x0192: return 0x83;  case 0x201E: return 0x84;
        case 0x2026: return 0x85;  case 0x2020: return 0x86;
        case 0x2021: return 0x87;  case 0x02C6: return 0x88;
        case 0x2030: return 0x89;  case 0x0160: return 0x8A;
        case 0x2039: return 0x8B;  case 0x0152: return 0x8C;
        case 0x017D: return 0x8E;  case 0x2018: return 0x91;
        case 0x2019: return 0x92;  case 0x201C: return 0x93;
        case 0x201D: return 0x94;  case 0x2022: return 0x95;
        case 0x2013: return 0x96;  case 0x2014: return 0x97;
        case 0x02DC: return 0x98;  case 0x2122: return 0x99;
        case 0x0161: return 0x9A;  case 0x203A: return 0x9B;
        case 0x0153: return 0x9C;  case 0x017E: return 0x9E;
        case 0x0178: return 0x9F;
    }
    return -1;
}

// Greedy ftfy: scan for runs of codepoints that map back to single
// Windows-1252 bytes >= 0x80.  Reinterpret as raw bytes and try UTF-8
// decode.  If the result produces codepoints outside the byte range
// (i.e., decoded a multi-byte UTF-8 sequence the model emitted as
// individual byte-level BPE tokens), substitute it.  Recovers em-
// dashes / smart-quotes / etc. from web-mojibake patterns the base
// model learned.
static std::string fix_mojibake(const std::string& s) {
    std::vector<uint32_t> cps;
    for (size_t i = 0; i < s.size(); ) cps.push_back(utf8_next_cp(s, i));
    std::string out;
    for (size_t i = 0; i < cps.size(); ) {
        int b0 = cp_to_win1252_byte(cps[i]);
        if (b0 < 0x80) {
            cp_to_utf8(cps[i], out); ++i; continue;
        }
        size_t j = i;
        std::string bytes;
        while (j < cps.size()) {
            int b = cp_to_win1252_byte(cps[j]);
            if (b < 0x80) break;
            bytes.push_back(static_cast<char>(b));
            ++j;
        }
        // Try UTF-8 decode of the byte run.
        std::string decoded;
        size_t bi = 0;
        bool ok = !bytes.empty();
        bool gained = false;
        while (bi < bytes.size()) {
            size_t prev = bi;
            uint32_t cp2 = utf8_next_cp(bytes, bi);
            if (bi == prev + 1 && static_cast<unsigned char>(bytes[prev]) >= 0x80) {
                ok = false; break;  // lead byte couldn't form a sequence
            }
            if (cp2 > 0xFF) gained = true;
            cp_to_utf8(cp2, decoded);
        }
        if (ok && gained) {
            out += decoded;
        } else {
            for (size_t k = i; k < j; ++k) cp_to_utf8(cps[k], out);
        }
        i = j;
    }
    return out;
}

std::string decode_tokens(const Vocab& v, const std::vector<uint16_t>& tokens) {
    std::string raw;
    for (auto t : tokens) {
        auto sv = v.bytes_for(t);
        raw.append(sv.data(), sv.size());
    }
    return fix_mojibake(raw);
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
// Subcommand: load-roms
// ---------------------------------------------------------------------
// Streams per-model BRAM init data from a model directory into the
// FPGA's on-chip ROMs.  Phase 1 BRAMs:
//   kind 0  rom_G1_m   (NL × D 16-b mantissas)         G1_m.hex
//   kind 1  rom_G1_e   (NL × NT_D 8-b exponents)       G1_e.hex
//   kind 2  rom_G2_m                                   G2_m.hex
//   kind 3  rom_G2_e                                   G2_e.hex
//   kind 4  rom_NW_m   (D 16-b mantissas, decode head) NORM_W_m.hex
//   kind 5  rom_NW_e   (NT_D 8-b exponents)            NORM_W_e.hex
//   kind 6  prompt_rom (N_PROMPT 16-b tokens)          PROMPT.hex
//
// .hex files are HuggingFace-bake outputs in $readmemh format —
// whitespace/comment-tolerant lines of hex words, one entry per line.
//
// Wire protocol:
//   1. write 0x060 = (inc=1<<31) | (kind<<18) | base_addr   to set target
//   2. for each entry value: write 0x061 = value            (addr auto-increments)
// Packed into FT_REG_WRITE frames with up to 16 writes per UDP packet
// to keep upload time bounded — full SmolLM2-135M Phase 1 set is
// ~37000 entries = ~2300 packets ≈ 1 s.

static std::vector<uint16_t> parse_hex_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint16_t> out;
    std::string line;
    while (std::getline(f, line)) {
        // strip // comments
        auto cslash = line.find("//");
        if (cslash != std::string::npos) line.resize(cslash);
        auto chash  = line.find('#');
        if (chash  != std::string::npos) line.resize(chash);
        std::istringstream is(line);
        std::string tok;
        while (is >> tok) {
            try {
                unsigned long v = std::stoul(tok, nullptr, 16);
                out.push_back(static_cast<uint16_t>(v & 0xFFFF));
            } catch (...) { /* skip garbage tokens */ }
        }
    }
    return out;
}

// Pack a batch of writes (up to 16, FT_REG_WRITE max) into one frame.
// Each entry is {addr_LE_u16, data_LE_u32, pad_LE_u16}, 8 bytes per entry,
// after an 8-byte header — matches eth_ctrl's parser.
struct RegW { uint16_t addr; uint32_t data; };

static void send_reg_write_batch(Udp& u, uint8_t seq,
                                 const std::vector<RegW>& batch) {
    if (batch.empty()) return;
    if (batch.size() > 16) throw std::runtime_error("batch too big");
    uint8_t tx[8 + 16 * 8];
    tx[0] = FT_REG_WRITE;
    tx[1] = seq;
    tx[2] = static_cast<uint8_t>(batch.size());
    for (int i = 3; i < 8; ++i) tx[i] = 0;
    for (size_t i = 0; i < batch.size(); ++i) {
        uint8_t* p = tx + 8 + i * 8;
        p[0] = static_cast<uint8_t>(batch[i].addr & 0xff);
        p[1] = static_cast<uint8_t>((batch[i].addr >> 8) & 0xff);
        p[2] = static_cast<uint8_t>(batch[i].data        & 0xff);
        p[3] = static_cast<uint8_t>((batch[i].data >>  8) & 0xff);
        p[4] = static_cast<uint8_t>((batch[i].data >> 16) & 0xff);
        p[5] = static_cast<uint8_t>((batch[i].data >> 24) & 0xff);
        p[6] = 0;
        p[7] = 0;
    }
    u.send(tx, 8 + batch.size() * 8);
}

// One mismatched BRAM entry: address, what the file says it should be,
// and what readback actually returned.
struct Mismatch { size_t addr; uint16_t want; uint16_t got; };

// Iterate every entry in `entries`, drive 0x060 with that address, read
// 0x062, and return the list of addresses that don't match.  Two
// round-trips per entry, ~7 s for the full Phase-1 set.
static std::vector<Mismatch> find_mismatches(
    Udp& u, int kind, const std::vector<uint16_t>& entries, uint8_t& seq) {
    std::vector<Mismatch> bad;
    bool is_exp = (kind == 1 || kind == 3 || kind == 5);
    for (size_t i = 0; i < entries.size(); ++i) {
        uint32_t target = (0u << 31)
                        | (static_cast<uint32_t>(kind & 0x1f) << 18)
                        | (static_cast<uint32_t>(i) & 0x3ffff);
        std::vector<RegW> one = {{REG_BRAM_TARGET, target}};
        send_reg_write_batch(u, seq++, one);
        uint16_t got  = reg_read(u, REG_BRAM_READ, 1, seq++)[0] & 0xffff;
        uint16_t want = entries[i];
        uint16_t got_cmp  = is_exp ? (got  & 0xff) : got;
        uint16_t want_cmp = is_exp ? (want & 0xff) : want;
        if (got_cmp != want_cmp) {
            bad.push_back({i, want, got});
        }
    }
    return bad;
}

// Patch a list of mismatched entries by writing each one individually
// in a 2-entry FT_REG_WRITE (0x060 target + 0x061 data) so the eth_ctrl
// dispatcher emits them back-to-back without re-ordering risk.
static void patch_rom(Udp& u, int kind,
                      const std::vector<Mismatch>& bad, uint8_t& seq) {
    for (const auto& m : bad) {
        uint32_t target = (0u << 31)
                        | (static_cast<uint32_t>(kind & 0x1f) << 18)
                        | (static_cast<uint32_t>(m.addr) & 0x3ffff);
        std::vector<RegW> two = {
            {REG_BRAM_TARGET, target},
            {REG_BRAM_DATA,   static_cast<uint32_t>(m.want)},
        };
        send_reg_write_batch(u, seq++, two);
    }
}

static void load_rom(Udp& u, int kind, const std::vector<uint16_t>& entries,
                     uint8_t& seq) {
    std::printf("[load-roms] kind=%d entries=%zu", kind, entries.size());
    std::fflush(stdout);
    if (entries.empty()) { std::printf("\n"); return; }
    // 1. Set target: inc=1, kind, base=0
    uint32_t target = (1u << 31)
                    | (static_cast<uint32_t>(kind & 0x1f) << 18)
                    | 0u /* base_addr=0 */;
    std::vector<RegW> first = {{REG_BRAM_TARGET, target}};
    send_reg_write_batch(u, seq++, first);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    // 2. Stream data writes — pack 16 entries per FT_REG_WRITE packet.
    // The destination BRAM is true-dual-port with port A clocked by
    // eth_clk (host side), so every regmap write to 0x061 lands as
    // a write on the very next eth_clk edge.  No CDC, no risk of
    // dropped pulses regardless of packet density.
    constexpr size_t PER_PKT = 16;
    for (size_t i = 0; i < entries.size(); i += PER_PKT) {
        size_t n = std::min(PER_PKT, entries.size() - i);
        std::vector<RegW> batch;
        batch.reserve(n);
        for (size_t j = 0; j < n; ++j) {
            batch.push_back({REG_BRAM_DATA, static_cast<uint32_t>(entries[i + j])});
        }
        send_reg_write_batch(u, seq++, batch);
    }

    // 3. Self-heal: verify-then-patch up to 3 rounds.  UDP frames can
    // drop in transit (no retransmit in the streaming write loop), so
    // a single dropped frame leaves up to 16 entries stale.  Reading
    // back through 0x062 and rewriting just the bad ones recovers
    // without re-streaming the whole rom.  ~10 s amortised verify time
    // per kind on the happy path; near-zero patch traffic when clean.
    constexpr int MAX_HEAL_ROUNDS = 3;
    for (int round = 0; round < MAX_HEAL_ROUNDS; ++round) {
        auto bad = find_mismatches(u, kind, entries, seq);
        if (bad.empty()) {
            std::printf("  OK%s\n",
                        round > 0 ? (" (healed in " + std::to_string(round) + " round(s))").c_str() : "");
            return;
        }
        std::printf("\n  round %d: patching %zu/%zu", round + 1, bad.size(), entries.size());
        std::fflush(stdout);
        patch_rom(u, kind, bad, seq);
    }
    // Final check — if still bad, complain loudly with the first 8 bad addrs.
    auto bad = find_mismatches(u, kind, entries, seq);
    if (bad.empty()) {
        std::printf("  OK (healed in %d round(s))\n", MAX_HEAL_ROUNDS);
    } else {
        std::printf("\n  FAIL: %zu entries still mismatched after %d patch rounds:\n",
                    bad.size(), MAX_HEAL_ROUNDS);
        for (size_t k = 0; k < bad.size() && k < 8; ++k) {
            std::printf("    [%zu] want=0x%04x got=0x%04x\n",
                        bad[k].addr, bad[k].want, bad[k].got);
        }
    }
}

// ---------------------------------------------------------------------
// Subcommand: verify-roms
// ---------------------------------------------------------------------
// Stand-alone diagnostic.  load-roms already self-heals via the same
// readback path, so verify-roms is mainly useful for post-inference
// checks ("did the BRAMs survive a run?") and for cases where the user
// wants to confirm contents without a re-load.
static int verify_rom(Udp& u, int kind, const std::vector<uint16_t>& entries,
                      uint8_t& seq, const char* name) {
    if (entries.empty()) return 0;
    std::printf("[verify-roms] kind=%d (%s) entries=%zu ... ", kind, name, entries.size());
    std::fflush(stdout);
    auto bad = find_mismatches(u, kind, entries, seq);
    for (size_t k = 0; k < bad.size() && k < 8; ++k) {
        std::printf("\n  [%zu] want=0x%04x got=0x%04x",
                    bad[k].addr, bad[k].want, bad[k].got);
    }
    std::printf("\n  %s: %zu/%zu mismatches\n",
                bad.empty() ? "OK" : "FAIL", bad.size(), entries.size());
    return bad.empty() ? 0 : 1;
}

// ---------------------------------------------------------------------
// Subcommand: peek-layer
// ---------------------------------------------------------------------
// Reads the persistent hout_m / hout_e arrays inside smollm_layer_bfp
// via debug wr_kind=10/11.  Since the multilayer wrapper iterates a
// SINGLE layer instance NL times, after the run finishes these hold
// the LAST layer's hidden_out — i.e. the decode_head's input.  Lets
// us see whether the layer chain is collapsing to zero / NaN before
// the head turns it into garbage token IDs.
//
// Output format: two columns per tile — mantissas (D entries, signed
// 16-bit) and shared exponents (NT_D entries, signed 8-bit).
static int peek_layer(Udp& u, int D) {
    if (D <= 0 || D % 16 != 0) {
        std::fprintf(stderr, "peek-layer: D must be a positive multiple of 16 (got %d)\n", D);
        return 2;
    }
    const int NT_D = D / 16;
    uint8_t seq = 200;
    std::vector<int16_t> mant(D);
    std::vector<int8_t>  expn(NT_D);
    for (int i = 0; i < D; ++i) {
        uint32_t target = (0u << 31)
                        | (uint32_t(10 & 0x1f) << 18)
                        | (uint32_t(i) & 0x3ffff);
        std::vector<RegW> one = {{REG_BRAM_TARGET, target}};
        send_reg_write_batch(u, seq++, one);
        uint16_t got = reg_read(u, REG_BRAM_READ, 1, seq++)[0] & 0xffff;
        mant[i] = int16_t(got);
    }
    for (int i = 0; i < NT_D; ++i) {
        uint32_t target = (0u << 31)
                        | (uint32_t(11 & 0x1f) << 18)
                        | (uint32_t(i) & 0x3ffff);
        std::vector<RegW> one = {{REG_BRAM_TARGET, target}};
        send_reg_write_batch(u, seq++, one);
        uint16_t got = reg_read(u, REG_BRAM_READ, 1, seq++)[0] & 0xffff;
        expn[i] = int8_t(got & 0xff);
    }
    // Print one tile per line so it's easy to spot zero/NaN regions.
    std::printf("[peek-layer] D=%d NT_D=%d (last layer's hout_m / hout_e)\n", D, NT_D);
    int zero_tiles = 0;
    int nonzero_mants = 0;
    for (int t = 0; t < NT_D; ++t) {
        int tile_nz = 0;
        std::printf("tile %02d  e=%4d  m:", t, int(expn[t]));
        for (int k = 0; k < 16; ++k) {
            int16_t m = mant[t*16 + k];
            std::printf(" %6d", int(m));
            if (m != 0) { tile_nz++; nonzero_mants++; }
        }
        std::printf("  (%d nz)\n", tile_nz);
        if (tile_nz == 0) zero_tiles++;
    }
    std::printf("[peek-layer] summary: %d/%d tiles all-zero, %d/%d mantissas non-zero\n",
                zero_tiles, NT_D, nonzero_mants, D);
    return 0;
}

int verify_roms(Udp& u, const std::string& dir) {
    struct Entry { int kind; const char* name; bool optional; };
    const Entry roms[] = {
        {0, "G1_m.hex",     false},
        {1, "G1_e.hex",     false},
        {2, "G2_m.hex",     false},
        {3, "G2_e.hex",     false},
        {4, "NORM_W_m.hex", false},
        {5, "NORM_W_e.hex", false},
        {6, "PROMPT.hex",   false},
    };
    uint8_t seq = 150;
    auto t0 = std::chrono::steady_clock::now();
    int total_fail = 0;
    const char* env_prefix = std::getenv("MGRT_PREFIX");
    std::string prefix = env_prefix ? env_prefix : "lbfp_full_";
    for (const auto& r : roms) {
        std::string p1 = dir + "/" + prefix + r.name;
        std::string p2 = dir + "/" + r.name;
        std::ifstream probe1(p1);
        std::string path = probe1 ? p1 : p2;
        try {
            auto entries = parse_hex_file(path);
            total_fail += verify_rom(u, r.kind, entries, seq, r.name);
        } catch (const std::exception& e) {
            if (r.optional) {
                std::fprintf(stderr, "[verify-roms] skip %s: %s\n", r.name, e.what());
            } else {
                std::fprintf(stderr, "[verify-roms] %s: %s\n", r.name, e.what());
                return 1;
            }
        }
    }
    auto secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("[verify-roms] %s in %.2f s\n",
                total_fail == 0 ? "ALL OK" : "FAILED", secs);
    return total_fail;
}

int load_roms(Udp& u, const std::string& dir) {
    struct Entry { int kind; const char* name; bool optional; };
    const Entry roms[] = {
        {0, "G1_m.hex",     false},
        {1, "G1_e.hex",     false},
        {2, "G2_m.hex",     false},
        {3, "G2_e.hex",     false},
        {4, "NORM_W_m.hex", false},
        {5, "NORM_W_e.hex", false},
        {6, "PROMPT.hex",   false},
    };
    uint8_t seq = 200;
    auto t0 = std::chrono::steady_clock::now();
    const char* env_prefix = std::getenv("MGRT_PREFIX");
    std::string prefix = env_prefix ? env_prefix : "lbfp_full_";
    size_t prompt_len = 0;   // captured from kind=6's .hex file; written to 0x063 at end
    for (const auto& r : roms) {
        // Try the model-dir-prefixed name first (eg lbfp_full_G1_m.hex,
        // or whatever MGRT_PREFIX overrides to), then the bare name.
        std::string p1 = dir + "/" + prefix + r.name;
        std::string p2 = dir + "/" + r.name;
        std::ifstream probe1(p1);
        std::string path = probe1 ? p1 : p2;
        try {
            auto entries = parse_hex_file(path);
            if (r.kind == 6) prompt_len = entries.size();
            load_rom(u, r.kind, entries, seq);
        } catch (const std::exception& e) {
            if (r.optional) {
                std::fprintf(stderr, "[load-roms] skip %s: %s\n", r.name, e.what());
            } else {
                std::fprintf(stderr, "[load-roms] %s: %s\n", r.name, e.what());
                return 1;
            }
        }
    }
    // Tell the FPGA how many of the prompt_rom entries to actually
    // consume — defaults to the value baked at synth time, but any
    // length up to NPROMPT_MAX (64) works on the same bitstream.
    if (prompt_len > 0) {
        std::vector<RegW> set = {{REG_N_PROMPT, static_cast<uint32_t>(prompt_len)}};
        send_reg_write_batch(u, seq++, set);
        auto got = reg_read(u, REG_N_PROMPT, 1, seq++)[0];
        if (got != prompt_len) {
            std::fprintf(stderr,
                "[load-roms] WARN: n_prompt_active set to %zu but readback %u\n",
                prompt_len, got);
        } else {
            std::printf("[load-roms] n_prompt_active = %zu\n", prompt_len);
        }
    }
    auto secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::printf("[load-roms] done in %.2f s\n", secs);
    return 0;
}

// ---------------------------------------------------------------------
// Subcommand: read-crc
// ---------------------------------------------------------------------
// Reads the FPGA's rolling XOR-with-rotate hash over every 512-bit
// AXI rdata beat the BFP master has seen since the last restart pulse.
// (Was strict CRC32 originally, but the 64-byte unrolled XOR chain
// blew the 5 ns ui_clk timing by 750 ps — swapped for a much shorter
// rotate+XOR-tree that fits 5 ns easily.  Weaker collision properties
// than CRC32 but adequate for the comparative diagnostic.)  After a
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
    std::printf("rdata_hash = 0x%08x\n", crc);
}

// ---------------------------------------------------------------------
// Subcommand: tokens
// ---------------------------------------------------------------------
std::vector<uint16_t> fetch_tokens(Udp& u, int n_steps) {
    int nwords = (n_steps * 16 + 31) / 32;
    // FastTrans MAX_REG_READS caps each FT_REG_READ at 19 words.  When
    // n_steps > 38 (NPROMPT_MAX + N_GEN = 63 → 32 words) we split into
    // multiple reads and concatenate.  Keeps the host compatible with
    // either the legacy (MAX_REG_READS=19) or a future bigger cap.
    constexpr int MAX_PER_READ = 19;
    std::vector<uint32_t> words;
    words.reserve(nwords);
    uint8_t seq = 50;
    for (int off = 0; off < nwords; off += MAX_PER_READ) {
        int chunk = std::min(MAX_PER_READ, nwords - off);
        auto part = reg_read(u, REG_RESULT + off, static_cast<uint8_t>(chunk), seq++);
        if ((int)part.size() != chunk) {
            throw std::runtime_error("expected " + std::to_string(chunk)
                                     + " result words at offset "
                                     + std::to_string(off) + ", got "
                                     + std::to_string(part.size()));
        }
        words.insert(words.end(), part.begin(), part.end());
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
    // Trim to active_prompt + N_GEN (= 15) — the buffer is sized for
    // NPROMPT_MAX prompts, but a short prompt leaves trailing zero
    // slots that aren't meaningful output.  Older bitstreams that
    // don't expose 0x063 just return 0; treat that as "no trim".
    try {
        uint32_t active = reg_read(u, REG_N_PROMPT, 1, 60)[0];
        if (active > 0 && active < tokens.size()) {
            size_t keep = std::min<size_t>(active + 15, tokens.size());
            tokens.resize(keep);
        }
    } catch (...) { /* old bitstream — leave tokens untrimmed */ }
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
        "  -r  target upload rate in MB/s, steady-paced per-chunk    (default 2.0,\n"
        "      0 = auto-probe: ramp 2/4/6/8/10 MB/s, pick min projected total)\n"
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
        "  tokens             pulse restart, wait for autoregress done,\n"
        "                     read 10 result words from 0x1D0, decode\n"
        "                     through vocab (auto-found), print hash\n"
        "  peek-tokens        read current result_tokens without restart\n"
        "  peek-layer <D>     read last layer's hout_m/hout_e via debug\n"
        "                     wr_kind=10/11 (D=hidden dim, e.g. 960 for\n"
        "                     smollm360, 576 for smollm135)\n"
        "  read-crc           read FPGA rolling hash of weight-bus reads\n"
        "                     since last restart (matches across runs of\n"
        "                     the same DDR3 contents)\n"
        "  load-roms <dir>    stream per-model BRAM init (G1/G2 gammas,\n"
        "                     final NORM_W, prompt tokens) from .hex files\n"
        "                     in <dir> via the host-write protocol\n"
        "  verify-roms <dir>  read every BRAM entry back and compare to\n"
        "                     the .hex files (~7 s; per-entry round-trip)\n"
        "  all     <bin>      load-roms → verify-roms → upload → restart\n"
        "                     → tokens (rom dir = bin's parent)\n",
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
            if (target_mbps < 0) {
                std::fprintf(stderr, "-r rate must be ≥ 0 (0 = auto-probe)\n");
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

    // Auto-discover the vocab: if -v wasn't given, try sensible default
    // paths next to the binary so `bfp_client tokens` decodes by default
    // instead of speaking in raw token IDs.  Silent if no vocab is found.
    if (vocab_path.empty()) {
        const char* candidates[] = {
            "host/bfp_vocab.bin",
            "bfp_vocab.bin",
            "release/bfp_vocab.bin",
            "../host/bfp_vocab.bin",
        };
        for (const char* p : candidates) {
            std::ifstream probe(p, std::ios::binary);
            if (probe) { vocab_path = p; break; }
        }
    }
    std::unique_ptr<Vocab> vocab;
    if (!vocab_path.empty()) {
        try { vocab = std::make_unique<Vocab>(Vocab::load(vocab_path)); }
        catch (const std::exception& e) {
            std::fprintf(stderr, "[vocab] load failed (%s): %s — tokens "
                         "will print raw IDs only\n", vocab_path.c_str(), e.what());
        }
    }

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
            // Single-shot autoregress means result_tokens latches at the
            // last run's output and stays until next restart.  So
            // `tokens` always pulses restart and waits for the new run
            // to finish — otherwise running `tokens` twice gives identical
            // (stale) output, which is a confusing default.  Use
            // `peek-tokens` for the read-only "show me what's already
            // there" case.
            if (!restart_and_wait_done(u)) {
                std::printf("[tokens] restart-wait timed out\n");
                return 1;
            }
            print_tokens(u, vocab.get(), n_steps);
            print_crc(u);
            return 0;
        }
        if (cmd == "peek-tokens") {
            print_tokens(u, vocab.get(), n_steps);
            print_crc(u);
            return 0;
        }
        if (cmd == "peek-layer") {
            if (argi >= argc) { usage(); return 2; }
            int D = std::atoi(argv[argi]);
            return peek_layer(u, D);
        }
        if (cmd == "read-crc") {
            print_crc(u);
            return 0;
        }
        if (cmd == "load-roms") {
            if (argi >= argc) { usage(); return 2; }
            return load_roms(u, argv[argi]);
        }
        if (cmd == "verify-roms") {
            if (argi >= argc) { usage(); return 2; }
            return verify_roms(u, argv[argi]);
        }
        if (cmd == "all") {
            if (argi >= argc) { usage(); return 2; }
            std::string bin = argv[argi];
            auto bv = reg_read(u, REG_BUILD_VERSION, 1, 1)[0];
            std::printf("[all] FPGA at %s:%u  BUILD_VERSION = 0x%08x\n",
                        peer_ip.c_str(), peer_port, bv);
            // ROM directory — derive from bin's parent dir.  A fresh
            // bitstream has all-zero G1/G2/NORM_W/PROMPT BRAMs; without
            // load-roms the forward pass collapses to <|endoftext|> for
            // every emitted token regardless of the DDR3 image.  Keep
            // verify-roms in the chain too so a corrupted upload (or a
            // mid-build silent BRAM clobber) is caught before tokens.
            std::string rom_dir;
            {
                auto slash = bin.find_last_of('/');
                rom_dir = (slash == std::string::npos) ? "." : bin.substr(0, slash);
            }
            std::printf("[all] load-roms %s\n", rom_dir.c_str());
            if (int rc = load_roms(u, rom_dir); rc != 0) return rc;
            std::printf("[all] verify-roms %s\n", rom_dir.c_str());
            if (int rc = verify_roms(u, rom_dir); rc != 0) return rc;
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
