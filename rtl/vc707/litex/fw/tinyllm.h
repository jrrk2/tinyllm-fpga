/* tinyllm firmware: shared declarations.  CSR accessors come from the LiteX
 * build (generated/csr.h); SoC constants (TINYLLM_*) from generated/soc.h. */
#ifndef TINYLLM_FW_H
#define TINYLLM_FW_H
#include <stdint.h>
#include <generated/csr.h>
#include <generated/soc.h>
#include <generated/mem.h>
#include <system.h>

/* DDR3 layout (byte offsets into main_ram): the weight image at 0 exactly as
 * lbfp_full_DDR3.bin lays it out (ws_base_* CSRs default to it), the serial
 * vocab table at 960 MiB, this firmware at 1008 MiB (see linker.ld). */
#define WEIGHTS_OFFSET   TINYLLM_WEIGHTS_OFFSET
#define VOCAB_OFFSET     0x3C000000u
#define FW_OFFSET        0x3F000000u
#define DDR(off)         ((volatile uint8_t *)(MAIN_RAM_BASE + (off)))

#define N_STEPS      (TINYLLM_NPROMPT_MAX + TINYLLM_NGEN)
#define NPROMPT_MAX  TINYLLM_NPROMPT_MAX
#define N_GEN        TINYLLM_NGEN

/* status CSR bits (litex/tinyllm.py, field order) */
#define ST_DONE          (1u << 0)
#define ST_DONE_LATCHED  (1u << 1)
#define ST_HAS_RUN       (1u << 2)
#define ST_FIRST_AR      (1u << 3)
#define ST_FIRST_R       (1u << 4)
#define ST_FROZEN        (1u << 5)

/* BRAM kinds (host/bfp_client.cpp load-roms) */
#define BRAM_KIND_PROMPT 6u

static inline void eng_bram_write(uint32_t kind, uint32_t addr, uint16_t data) {
    tinyllm_bram_target_write((kind << 18) | (addr & 0x3FFFFu));
    tinyllm_bram_data_write(data);
}
static inline uint16_t eng_bram_read(uint32_t kind, uint32_t addr) {
    tinyllm_bram_target_write((kind << 18) | (addr & 0x3FFFFu));
    return (uint16_t)tinyllm_bram_rdata_read();
}
/* Make everything the CPU / SD DMA wrote through the L2 visible to the
 * engine's own LiteDRAM port, then pulse restart. */
static inline void eng_restart(void) {
    flush_cpu_dcache();
#ifdef CONFIG_L2_SIZE
    flush_l2_cache();
#endif
    tinyllm_restart_write(1);
}
/* Wide CSRStatus: LiteX stores the most significant 32-bit word at the lowest
 * address.  Result token k (bits [16k+15:16k] of the 16*N_STEPS-bit vector)
 * therefore sits in word (nwords-1 - k/2), low half for even k. */
static inline uint32_t eng_result_word_le(unsigned i) {   /* old 0x1D0+i view */
    return csr_read_simple(CSR_TINYLLM_RESULT_ADDR + 4u * (CSR_TINYLLM_RESULT_SIZE - 1u - i));
}
static inline uint16_t eng_result_token(unsigned k) {
    uint32_t w = eng_result_word_le(k >> 1);
    return (uint16_t)((k & 1u) ? (w >> 16) : (w & 0xFFFFu));
}
static inline uint32_t eng_first_rdata_word_le(unsigned i) {
    return csr_read_simple(CSR_TINYLLM_DBG_FIRST_RDATA_ADDR + 4u * (CSR_TINYLLM_DBG_FIRST_RDATA_SIZE - 1u - i));
}

/* net.c */
void net_init(void);
void net_service(void);
void net_stats(void);
/* tok.c */
int  vocab_init(void);
int  tokenize(const char *s, int n, uint16_t *out, int max);
void print_token(uint32_t t);
/* sdload.c */
int  sd_load_all(void);
int  sd_ls(void);
/* main.c */
void run_prompt_tokens(const uint16_t *toks, int nt);
#endif
