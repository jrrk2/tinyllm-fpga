/* fw_shell.c — PicoSoC board bring-up firmware (Build B: ethernet over LSU).
 * Proves the SoC is alive (UART + LEDs) AND drives ethernet through the LSU
 * frame-bus bridge at 0x2000_0000:
 *   - init the MAC, enable RX
 *   - poll RSR; on a received frame, print buf/len + first words over UART, free
 *   - periodically TX a broadcast heartbeat (host sees it via tcpdump)
 * No globals so startup needs no data init. */
#include <stdint.h>

#define reg_uart_clkdiv (*(volatile uint32_t *)0x02000004)
#define reg_uart_data   (*(volatile uint32_t *)0x02000008)
#define reg_leds        (*(volatile uint32_t *)0x03000000)

#define UART_DIV 1085   /* 125 MHz / 115200 */

/* ---- ethernet LSU bridge (0x2000_0000 + lsu_byte_addr) ---- */
#define ETH(off) (*(volatile uint32_t *)(0x20000000u + (off)))
#define ETH_MACLO 0x0800
#define ETH_MACHI 0x0808
#define ETH_TPLR  0x0810   /* write {tbuf<<11 | byte_len} -> send */
#define ETH_RXEN  0x0828
#define ETH_RSR   0x0830   /* rd: [15]=ready,[4:0]=buf ; wr: (buf+1) frees */
#define ETH_RXLEN(b)    (0x0C00u + ((b) << 3))
#define ETH_RX(b, w)    (0x10000u + ((uint32_t)(b) << 11) + ((uint32_t)(w) << 3))
#define ETH_TX(t, w)    (0x1000u  + ((uint32_t)(t) << 11) + ((uint32_t)(w) << 3))

/* SoC MAC 02:00:00:4D:47:32 (one off the HW eth_ctrl's ...31). */
#define MAC_LO 0x004D4732u   /* MAC[31:0]  */
#define MAC_HI 0x00000200u   /* MAC[47:32] */

/* ---- DDR data window (0x3000_0000 + 16 MB; bank select at 0x3100_0000) ---- */
#define DDR(off)  (*(volatile uint32_t *)(0x30000000u + (off)))
#define DDR_BANK  (*(volatile uint32_t *)0x31000000u)

static void putc_(char c) { if (c == '\n') reg_uart_data = '\r'; reg_uart_data = c; }
static void print_(const char *s) { while (*s) putc_(*s++); }
static void hex_(uint32_t v, int nyb) {
    for (int i = nyb - 1; i >= 0; i--) putc_("0123456789abcdef"[(v >> (4 * i)) & 15]);
}

static void eth_init(void) {
    ETH(ETH_MACLO) = MAC_LO;
    ETH(ETH_MACHI) = MAC_HI | (1u << 22);   /* + enable */
    ETH(ETH_RXEN)  = 31;
}

static int eth_rx_poll(void) {            /* -> buffer index, or -1 */
    uint32_t rsr = ETH(ETH_RSR);
    return (rsr & (1u << 15)) ? (int)(rsr & 0x1f) : -1;
}

static void eth_tx_heartbeat(uint8_t tbuf) {
    /* dst=broadcast, src=our MAC, ethertype 0x88B5, "PICOSOC" payload, 60 B. */
    static const uint8_t src[6] = {0x02, 0x00, 0x00, 0x4D, 0x47, 0x32};
    uint8_t f[64];
    for (int i = 0; i < 64; i++) f[i] = 0;
    for (int i = 0; i < 6; i++) f[i] = 0xFF;          /* dst broadcast */
    for (int i = 0; i < 6; i++) f[6 + i] = src[i];     /* src */
    f[12] = 0x88; f[13] = 0xB5;                         /* ethertype */
    const char *msg = "PICOSOC-SHELL";
    for (int i = 0; msg[i]; i++) f[14 + i] = (uint8_t)msg[i];
    for (int w = 0; w < 8; w++) {                       /* 64 B = 8 x 64-bit */
        uint32_t lo = f[w*8] | (f[w*8+1] << 8) | (f[w*8+2] << 16) | (f[w*8+3] << 24);
        uint32_t hi = f[w*8+4] | (f[w*8+5] << 8) | (f[w*8+6] << 16) | (f[w*8+7] << 24);
        ETH(ETH_TX(tbuf, w))     = lo;
        ETH(ETH_TX(tbuf, w) + 4) = hi;
    }
    ETH(ETH_TPLR) = ((uint32_t)tbuf << 11) | 60u;       /* trigger send */
}

void main(void) {
    reg_uart_clkdiv = UART_DIV;
    reg_leds = 0x01;
    print_("\n=== PicoSoC shell (Build B: ethernet) on VC707 ===\n");
    eth_init();
    print_("MAC 02:00:00:4D:47:32, RX enabled. Heartbeat TX every ~1s.\n");

    /* DDR data-window self-test: write a pattern, read it back. */
    DDR_BANK = 0;
    for (uint32_t i = 0; i < 16; i++) DDR(i * 4) = 0xA5A50000u + i;
    int ddr_ok = 1;
    for (uint32_t i = 0; i < 16; i++)
        if (DDR(i * 4) != 0xA5A50000u + i) ddr_ok = 0;
    print_("DDR self-test: ");
    print_(ddr_ok ? "PASS\n" : "FAIL\n");
    print_("  DDR[0]="); hex_(DDR(0), 8); print_(" DDR[4]="); hex_(DDR(16), 8); putc_('\n');

    uint32_t led = 1, spin = 0, hbcnt = 0;
    uint8_t  tbuf = 0;
    for (;;) {
        /* RX */
        int b = eth_rx_poll();
        if (b >= 0) {
            uint32_t len = ETH(ETH_RXLEN(b)) & 0x7FF;
            print_("RX buf="); hex_(b, 1); print_(" len="); hex_(len, 3); print_(":");
            for (uint32_t w = 0; w < 4; w++) {
                putc_(' '); hex_(ETH(ETH_RX(b, w)), 8);
                putc_(' '); hex_(ETH(ETH_RX(b, w) + 4), 8);
            }
            putc_('\n');
            ETH(ETH_RSR) = (uint32_t)((b + 1) & 0x1f);   /* free buffer */
        }

        /* UART echo */
        int32_t c = reg_uart_data;
        if (c != -1) putc_((char)c);

        /* periodic heartbeat TX + LED walk */
        if (++spin >= 1000000u) {
            spin = 0;
            led = ((led << 1) | (led >> 6)) & 0x7F; if (!led) led = 1;
            reg_leds = led;
            if (++hbcnt >= 60u) { hbcnt = 0; eth_tx_heartbeat(tbuf); tbuf ^= 1; }
        }
    }
}
