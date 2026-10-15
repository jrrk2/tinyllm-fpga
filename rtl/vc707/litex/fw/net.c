/* net.c -- bfp_client's UDP protocol on LiteEth (libliteeth udp).
 *
 * Same wire format as the PicoSoC firmware (fw_engine.c) so host/bfp_client
 * and host/fpga_per_layer_dump.py work unchanged, pointed at this SoC's IP:
 *   FT_REG_READ  [0x02 seq addr_lo addr_hi n 0 0 0]        -> FT_REG_RSP
 *   FT_REG_WRITE [0x01 seq n 0 0 0 {pad16 addr16 data32}*] -> FT_ACK
 *   FT_DDR_WRITE  [0x0A seq addr32 data64]                 -> FT_ACK
 *   FT_DDR_WRITE_N[0x0D seq addr32 data(N*64)]             -> FT_ACK
 * The old regmap addresses are translated to the tinyllm CSRs here, and the
 * upload flow-control counters the client polls are kept in software.
 *
 * Replies go to whoever asked.  libliteeth's udp_send() only knows one
 * peer (the last ARP-resolved address), so a frame from a new host is
 * dropped once while its MAC is resolved from the main loop; the client
 * retries register reads, so nothing is lost. */
#include <stdio.h>
#include <string.h>
#include <libliteeth/udp.h>
#include <libliteeth/inet.h>
#include "tinyllm.h"

#define FPGA_UDP_PORT 19783u
#define FT_REG_WRITE   0x01u
#define FT_REG_READ    0x02u
#define FT_REG_RSP     0x03u
#define FT_ACK         0x06u
#define FT_DDR_WRITE   0x0Au
#define FT_DDR_WRITE_N 0x0Du
#define FT_DDR_BLAST   0x0Eu   /* FT_DDR_WRITE_N without an ACK (fire-and-forget) */
#define FT_CKSUM_REQ   0x10u   /* {u32 base, u32 block_bytes, u16 first, u16 n} -> FT_CKSUM_RSP */
#define FT_CKSUM_RSP   0x11u   /* {u16 first, u16 n, u16 pad, u16 pad, u32 sum[n]} */
#define CKSUM_MAX_BLOCKS 64u
#define DDR_WRITE_N_MAX 22u

/* Block checksum shared with the host (bfp_client blast): a Fletcher-style
 * pair over little-endian 32-bit words, so a missing, duplicated or
 * misplaced 64-byte chunk changes it.  ~3 cycles/word: 1 MiB in ~10 ms. */
static uint32_t cksum_block(const uint32_t *w, uint32_t nwords) {
    uint32_t s1 = 0, s2 = 0;
    for (uint32_t i = 0; i < nwords; i++) { s1 += w[i]; s2 += s1; }
    return s1 ^ (s2 * 0x9E3779B1u);
}

static uint32_t chunks_rx, chunks_done, acks_tx;
static uint32_t peer_ip;          /* host we can reply to (ARP resolved) */
static uint32_t pending_ip;       /* host that needs resolving */
static uint16_t last_dbg_addr;

static uint32_t reg_read(uint16_t a) {
    uint32_t st = tinyllm_status_read();
    /* The old 0x014..0x023 first-R-beat window was aliased by the registers at
     * 0x017..0x01C (the client's `verify` only ever saw words 0-2 and 12-15);
     * keep that precedence: named registers first, window second. */
    switch (a) {
        case 0x000: return 0x5452474Du;                              /* 'MGRT' */
        case 0x00A: return tinyllm_snap_layer_sel_read();
        case 0x012: return 1u | ((st & ST_FIRST_AR) ? 2u : 0u) | ((st & ST_FIRST_R) ? 4u : 0u)
                            | ((st & ST_FIRST_R) ? 8u : 0u);          /* calib, ar, r, snap_done */
        case 0x013: return tinyllm_dbg_first_araddr_read();
        case 0x018: return 31u << 10;                                /* ring: empty, 31 slots */
        case 0x019: return chunks_rx;
        case 0x01A: case 0x01B: return chunks_done;
        case 0x01C: return acks_tx;
        case 0x049: return (st & ST_HAS_RUN) ? 1u : 0u;
        case 0x04A: return tinyllm_weight_hash_read();
        case 0x062: return tinyllm_bram_rdata_read();
        case 0x063: return tinyllm_n_prompt_read();
        case 0x064: return tinyllm_snap_step_sel_read();
        case 0x065: return tinyllm_freeze_en_read();
        case 0x066: return tinyllm_trig_cyc_read();
        case 0x067: return tinyllm_trig_cyc_en_read();
        case 0x068: return tinyllm_dbg_cyc_read();
        case 0x06A: return tinyllm_dbg_cur_layer_read() | ((st & ST_FROZEN) ? 0x100u : 0u);
        case 0x10F: return tinyllm_build_version_read();
        case 0x1F0: return ((st & ST_DONE_LATCHED) ? 2u : 0u) | ((st & ST_DONE) ? 1u : 0u);
        default:    break;
    }
    if (a >= 0x014 && a <= 0x023) return eng_first_rdata_word_le(a - 0x014);
    if (a >= 0x1D0 && a < 0x1D0 + CSR_TINYLLM_RESULT_SIZE) return eng_result_word_le(a - 0x1D0);
    return 0;
}

static void reg_write(uint16_t a, uint32_t v) {
    switch (a) {
        case 0x00A: tinyllm_snap_layer_sel_write(v); break;
        case 0x048: tinyllm_arm_write(1); break;
        case 0x060: tinyllm_bram_target_write(v); break;
        case 0x061: tinyllm_bram_data_write(v & 0xFFFFu); break;
        case 0x063: tinyllm_n_prompt_write(v); break;
        case 0x064: tinyllm_snap_step_sel_write(v); break;
        case 0x065: tinyllm_freeze_en_write(v & 1u); break;
        case 0x066: tinyllm_trig_cyc_write(v); break;
        case 0x067: tinyllm_trig_cyc_en_write(v & 1u); break;
        case 0x069: tinyllm_dbg_stage_sel_write(v); break;
        case 0x1F1: if (v & 1u) eng_restart(); break;
        default: break;
    }
}

static void reply(uint16_t dst_port, uint32_t len) {
    if (udp_send(FPGA_UDP_PORT, dst_port, len)) acks_tx++;
}

static void rx_callback(uint32_t src_ip, uint16_t src_port, uint16_t dst_port,
                        void *data, uint32_t length) {
    if (dst_port != FPGA_UDP_PORT || length < 2) return;
    const uint8_t *p = data;
    uint8_t ft = p[0], seq = p[1];
    int can_reply = (src_ip == peer_ip);
    if (!can_reply) pending_ip = src_ip;
    uint8_t *tx = udp_get_tx_buffer();

    if (ft == FT_DDR_WRITE || ft == FT_DDR_WRITE_N || ft == FT_DDR_BLAST) {
        uint32_t addr = p[2] | (p[3] << 8) | ((uint32_t)p[4] << 16) | ((uint32_t)p[5] << 24);
        uint32_t n = (ft == FT_DDR_WRITE) ? 1u : ((length > 6u) ? ((length - 6u) >> 6) : 0u);
        if (n > DDR_WRITE_N_MAX) n = DDR_WRITE_N_MAX;
        addr &= 0x3FFFFFFFu;
        if (addr + n * 64u <= MAIN_RAM_SIZE) {
            memcpy((void *)(MAIN_RAM_BASE + addr), p + 6, n * 64u);
            chunks_rx += n; chunks_done += n;
        }
        if (ft == FT_DDR_BLAST || !can_reply) return;
        tx[0] = FT_ACK; tx[1] = seq;
        reply(src_port, 2);
    } else if (ft == FT_CKSUM_REQ && length >= 14) {
        uint32_t base  = p[2] | (p[3] << 8) | ((uint32_t)p[4] << 16) | ((uint32_t)p[5] << 24);
        uint32_t bsz   = p[6] | (p[7] << 8) | ((uint32_t)p[8] << 16) | ((uint32_t)p[9] << 24);
        uint32_t first = p[10] | (p[11] << 8), n = p[12] | (p[13] << 8);
        if (n > CKSUM_MAX_BLOCKS) n = CKSUM_MAX_BLOCKS;
        if (!can_reply) return;
        flush_cpu_dcache();                      /* read what is really in DRAM */
#ifdef CONFIG_L2_SIZE
        flush_l2_cache();
#endif
        tx[0] = FT_CKSUM_RSP; tx[1] = seq; tx[2] = first; tx[3] = first >> 8;
        tx[4] = n; tx[5] = n >> 8; tx[6] = tx[7] = 0;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t off = base + (first + i) * bsz, v = 0;
            if ((bsz & 3u) == 0 && off + bsz <= MAIN_RAM_SIZE)
                v = cksum_block((const uint32_t *)(MAIN_RAM_BASE + off), bsz >> 2);
            tx[8 + 4*i] = v; tx[9 + 4*i] = v >> 8; tx[10 + 4*i] = v >> 16; tx[11 + 4*i] = v >> 24;
        }
        reply(src_port, 8 + 4 * n);
    } else if (ft == FT_REG_WRITE) {
        uint32_t n = p[2];
        for (uint32_t i = 0; i < n && 14u + 8u * i <= length; i++) {
            const uint8_t *e = p + 6 + 8 * i;             /* {pad16, addr16, data32} */
            uint16_t a = e[2] | (e[3] << 8);
            uint32_t v = e[4] | (e[5] << 8) | ((uint32_t)e[6] << 16) | ((uint32_t)e[7] << 24);
            reg_write(a, v);
        }
        if (!can_reply) return;
        tx[0] = FT_ACK; tx[1] = seq;
        reply(src_port, 2);
    } else if (ft == FT_REG_READ) {
        uint16_t a = p[2] | (p[3] << 8);
        uint32_t n = p[4]; if (n > 19) n = 19;
        last_dbg_addr = a;
        if (!can_reply) return;
        tx[0] = FT_REG_RSP; tx[1] = seq; tx[2] = a; tx[3] = a >> 8; tx[4] = n;
        tx[5] = tx[6] = tx[7] = 0;
        for (uint32_t i = 0; i < n; i++) {
            uint32_t v = reg_read(a + i);
            tx[8 + 4*i] = v; tx[9 + 4*i] = v >> 8; tx[10 + 4*i] = v >> 16; tx[11 + 4*i] = v >> 24;
        }
        reply(src_port, 8 + 4 * n);
    } else {                                                /* FT_PROG_* etc.: ack only */
        if (!can_reply) return;
        tx[0] = FT_ACK; tx[1] = seq;
        reply(src_port, 2);
    }
}

void net_init(void) {
    uint8_t mac[6] = {MACADDR1, MACADDR2, MACADDR3, MACADDR4, MACADDR5, MACADDR6};
    eth_init();
    udp_start(mac, IPTOINT(LOCALIP1, LOCALIP2, LOCALIP3, LOCALIP4));
    udp_set_callback(rx_callback);
    printf("eth: %d.%d.%d.%d:%u  mac %02x:%02x:%02x:%02x:%02x:%02x\n",
           LOCALIP1, LOCALIP2, LOCALIP3, LOCALIP4, FPGA_UDP_PORT,
           mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

void net_service(void) {
    udp_service();
    if (pending_ip && pending_ip != peer_ip) {
        uint32_t ip = pending_ip;
        if (udp_arp_resolve(ip)) peer_ip = ip;
        pending_ip = 0;
    }
}

void net_stats(void) {
    printf("eth chunks rx=%lu done=%lu acks=%lu peer=%lu.%lu.%lu.%lu last_reg=0x%03x\n",
           (unsigned long)chunks_rx, (unsigned long)chunks_done, (unsigned long)acks_tx,
           (unsigned long)(peer_ip >> 24), (unsigned long)((peer_ip >> 16) & 255),
           (unsigned long)((peer_ip >> 8) & 255), (unsigned long)(peer_ip & 255), last_dbg_addr);
}
