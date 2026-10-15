/* tok.c -- greedy longest-match tokeniser over the vocab table in DDR3
 * (host/gen_serial_vocab.py; VOCAB.BIN on the SD card or
 * `bfp_client -a 0x3C000000 upload serial_vocab.bin`):
 *   u32[0] 'SVOC'  [1] n_tokens  [2] order_off  [3] entries_off  [4] blob_off
 *   byte 32: first-byte index, 256 x {u32 start, u32 count} into `order`
 *   order:   u16 token ids grouped by first byte, longest token first
 *   entries: u32 per token id = (len << 24) | blob_off
 *   blob:    raw token bytes */
#include <stdio.h>
#include "tinyllm.h"

#define SVOC_MAGIC 0x434F5653u
static const uint8_t  *sv_base;
static const uint16_t *sv_order;
static const uint32_t *sv_entries;
static const uint8_t  *sv_blob;
static int sv_ok;

static inline uint32_t rd32(const uint8_t *p) { return *(const volatile uint32_t *)p; }

int vocab_init(void) {
    flush_cpu_dcache();
#ifdef CONFIG_L2_SIZE
    flush_l2_cache();
#endif
    sv_base = (const uint8_t *)(MAIN_RAM_BASE + VOCAB_OFFSET);
    if (rd32(sv_base) != SVOC_MAGIC) { sv_ok = 0; return 0; }
    sv_order   = (const uint16_t *)(sv_base + rd32(sv_base + 8));
    sv_entries = (const uint32_t *)(sv_base + rd32(sv_base + 12));
    sv_blob    = sv_base + rd32(sv_base + 16);
    sv_ok = 1;
    return 1;
}

void print_token(uint32_t t) {
    if (!sv_ok) { printf("[%lu]", (unsigned long)t); return; }
    uint32_t e = sv_entries[t], len = e >> 24, off = e & 0xFFFFFFu;
    for (uint32_t i = 0; i < len; i++) putchar(sv_blob[off + i]);
}

int tokenize(const char *s, int n, uint16_t *out, int max) {
    if (!sv_ok && !vocab_init()) return -1;
    int pos = 0, cnt = 0;
    while (pos < n && cnt < max) {
        const uint8_t *ix = sv_base + 32 + (uint32_t)(uint8_t)s[pos] * 8;
        uint32_t start = rd32(ix), count = rd32(ix + 4);
        uint32_t best = 0; int best_len = 0;
        for (uint32_t i = start; i < start + count; i++) {
            uint32_t tok = sv_order[i];
            uint32_t e = sv_entries[tok];
            int len = (int)(e >> 24); uint32_t off = e & 0xFFFFFFu;
            if (len > n - pos) continue;
            int k = 0;
            while (k < len && (uint8_t)s[pos + k] == sv_blob[off + k]) k++;
            if (k == len) { best = tok; best_len = len; break; }
        }
        if (best_len == 0) { pos++; continue; }
        out[cnt++] = (uint16_t)best; pos += best_len;
    }
    return cnt;
}
