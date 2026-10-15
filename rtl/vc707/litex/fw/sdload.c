/* sdload.c -- model setup from the SD card (LiteSDCard + FatFs, read-only).
 *
 * Files (FAT, 8.3 names), made by litex/sd/make_sd_bundle.py:
 *   WEIGHTS.BIN  the DDR3 weight image (lbfp_full_DDR3.bin) -> main_ram + 0
 *   ROMS.BIN     the BRAM ROM set (gammas, final norm, prompt):
 *                records {u32 kind, u32 n, u16 data[n], pad to 4}
 *   VOCAB.BIN    the serial-prompt vocab table           -> main_ram + VOCAB_OFFSET
 *   (TINYLLM.CFG optional: "ws_base_<region>=0x..." overrides, one per line)
 * The SD DMA and the CPU both write through the L2 cache; eng_restart()
 * flushes it before the engine's own DRAM port reads anything. */
#include <stdio.h>
#include <string.h>
#include <liblitesdcard/sdcard.h>
#include <libfatfs/ff.h>
#include <libbase/progress.h>
#include "tinyllm.h"

static int sd_ready;

static int sd_open(FATFS *fs) {
    if (!sd_ready) {
        if (!sdcard_init()) { printf("sd: no card / init failed\n"); return 0; }
        fatfs_set_ops_sdcard();
        sd_ready = 1;
    }
    FRESULT fr = f_mount(fs, "", 1);
    if (fr != FR_OK) { printf("sd: mount failed (FatFs %d)\n", fr); sd_ready = 0; return 0; }
    return 1;
}

/* Stream a file into DDR3 at `dst`; returns bytes read or -1. */
static long sd_file_to_ram(const char *name, uintptr_t dst, size_t max) {
    FIL f;
    if (f_open(&f, name, FA_READ) != FR_OK) { printf("sd: %s not found\n", name); return -1; }
    unsigned long len = f_size(&f);
    if (len > max) { printf("sd: %s too large (%lu > %lu)\n", name, len, (unsigned long)max); f_close(&f); return -1; }
    printf("sd: %s -> 0x%08lx (%lu bytes)\n", name, (unsigned long)dst, len);
    init_progression_bar(len);
    unsigned long off = 0; UINT br;
    while (off < len) {
        if (f_read(&f, (void *)(dst + off), 0x8000, &br) != FR_OK) { printf("\nsd: read error\n"); f_close(&f); return -1; }
        if (br == 0) break;
        off += br;
        if ((off & 0xFFFFF) == 0 || off == len) show_progress(off);
    }
    printf("\n");
    f_close(&f);
    return (long)off;
}

static int sd_load_roms(void) {
    FIL f;
    if (f_open(&f, "ROMS.BIN", FA_READ) != FR_OK) { printf("sd: ROMS.BIN not found\n"); return 0; }
    static uint16_t buf[8192];
    int nrec = 0;
    for (;;) {
        uint32_t hdr[2]; UINT br;
        if (f_read(&f, hdr, 8, &br) != FR_OK || br < 8) break;
        uint32_t kind = hdr[0], n = hdr[1], done = 0;
        tinyllm_bram_target_write((1u << 31) | (kind << 18));   /* auto-increment from 0 */
        while (done < n) {
            uint32_t want = n - done; if (want > 8192) want = 8192;
            if (f_read(&f, buf, want * 2, &br) != FR_OK || br != want * 2) { printf("sd: ROMS.BIN truncated\n"); f_close(&f); return 0; }
            for (uint32_t i = 0; i < want; i++) tinyllm_bram_data_write(buf[i]);
            done += want;
        }
        if (n & 1) { uint16_t pad; f_read(&f, &pad, 2, &br); }
        printf("sd: ROM kind %lu: %lu entries\n", (unsigned long)kind, (unsigned long)n);
        if (kind == BRAM_KIND_PROMPT) tinyllm_n_prompt_write(n);
        nrec++;
    }
    f_close(&f);
    return nrec;
}

int sd_load_all(void) {
    FATFS fs;
    if (!sd_open(&fs)) return 0;
    int ok = 1;
    if (sd_file_to_ram("WEIGHTS.BIN", MAIN_RAM_BASE + WEIGHTS_OFFSET, TINYLLM_BASE_END) < 0) ok = 0;
    if (sd_load_roms() == 0) ok = 0;
    if (sd_file_to_ram("VOCAB.BIN", MAIN_RAM_BASE + VOCAB_OFFSET, FW_OFFSET - VOCAB_OFFSET) < 0) ok = 0;
    f_mount(0, "", 0);
    flush_cpu_dcache();
#ifdef CONFIG_L2_SIZE
    flush_l2_cache();
#endif
    vocab_init();
    printf("sd: setup %s\n", ok ? "complete" : "INCOMPLETE");
    return ok;
}

/* True when a card answers and holds WEIGHTS.BIN (cheap enough at boot). */
int sd_card_present(void) {
    FATFS fs; FIL f;
    if (!sd_open(&fs)) return 0;
    int ok = (f_open(&f, "WEIGHTS.BIN", FA_READ) == FR_OK);
    if (ok) f_close(&f);
    f_mount(0, "", 0);
    return ok;
}

int sd_ls(void) {
    FATFS fs; DIR d; FILINFO fi;
    if (!sd_open(&fs)) return 0;
    if (f_opendir(&d, "/") != FR_OK) { printf("sd: opendir failed\n"); f_mount(0, "", 0); return 0; }
    while (f_readdir(&d, &fi) == FR_OK && fi.fname[0]) printf("  %10lu  %s\n", (unsigned long)fi.fsize, fi.fname);
    f_closedir(&d);
    f_mount(0, "", 0);
    return 1;
}
