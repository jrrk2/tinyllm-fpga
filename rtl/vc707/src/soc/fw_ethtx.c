/* fw_ethtx.c — minimal ethernet TX probe.  Brings up the MAC and broadcasts one
 * fixed 60-byte frame (~1/s) so the host can confirm the SoC->bridge->MAC->wire
 * path with tcpdump, independent of RX / the reply path / bfp_client.
 *
 * Watch on the host:  sudo tcpdump -nei <iface> ether proto 0x88b5
 *   (or:  sudo tcpdump -nei <iface> ether src 02:00:00:4d:47:31)
 * Expect a broadcast frame, src 02:00:00:4d:47:31, ethertype 0x88B5, payload
 * starting "PICOSOC-ETHTX", once per ~second.  A '.' is printed on the UART per TX. */
#include <stdint.h>

#define reg_uart_clkdiv (*(volatile uint32_t *)0x02000004)
#define reg_uart_data   (*(volatile uint32_t *)0x02000008)
#define UART_DIV 1085

#define ETH(off)  (*(volatile uint32_t *)(0x20000000u + (off)))
#define ETH_MACLO 0x0800u
#define ETH_MACHI 0x0808u   /* [22]=enable, [15:0]=mac[47:32] */
#define ETH_RXEN  0x0828u
#define ETH_TPLR  0x0810u   /* write {tbuf<<11 | byte_len} -> send */
#define ETH_TX(t, w) (0x1000u + ((uint32_t)(t) << 11) + ((uint32_t)(w) << 3))

static void putc_(char c) { if (c == '\n') reg_uart_data = '\r'; reg_uart_data = c; }
static void print_(const char *s) { while (*s) putc_(*s++); }

static void send_frame(uint8_t tbuf) {
    uint8_t f[64];
    for (int i = 0; i < 64; i++) f[i] = 0;
    for (int i = 0; i < 6; i++) f[i] = 0xFF;            /* dst = broadcast */
    f[6]=0x02; f[7]=0x00; f[8]=0x00; f[9]=0x4D; f[10]=0x47; f[11]=0x31;  /* src MAC */
    f[12] = 0x88; f[13] = 0xB5;                          /* ethertype 0x88B5 (experimental) */
    const char *msg = "PICOSOC-ETHTX";
    for (int i = 0; msg[i]; i++) f[14 + i] = (uint8_t)msg[i];
    for (int w = 0; w < 8; w++) {                        /* 64 B = 8 x 64-bit LSU words */
        uint32_t lo = f[w*8] | (f[w*8+1]<<8) | (f[w*8+2]<<16) | (f[w*8+3]<<24);
        uint32_t hi = f[w*8+4] | (f[w*8+5]<<8) | (f[w*8+6]<<16) | (f[w*8+7]<<24);
        ETH(ETH_TX(tbuf, w))     = lo;
        ETH(ETH_TX(tbuf, w) + 4) = hi;
    }
    ETH(ETH_TPLR) = ((uint32_t)tbuf << 11) | 60u;        /* trigger send, 60-byte frame */
}

void main(void) {
    reg_uart_clkdiv = UART_DIV;
    print_("\n=== ethtx: broadcast 0x88B5 'PICOSOC-ETHTX' ~1/s ===\n");
    ETH(ETH_MACLO) = 0x004D4731u;
    ETH(ETH_MACHI) = 0x00000200u | (1u << 22);           /* mac + enable */
    ETH(ETH_RXEN)  = 31;
    uint8_t tbuf = 0;
    for (;;) {
        send_frame(tbuf);
        tbuf ^= 1;
        putc_('.');
        for (volatile uint32_t d = 0; d < 5000000u; d++) { }   /* ~1 s busy-wait */
    }
}
