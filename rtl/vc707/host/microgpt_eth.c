/* microgpt_eth.c — minimal raw-Ethernet launcher for vc707_microgpt_eth.
 *
 * Linux only (AF_PACKET). Build:
 *   gcc -O2 -Wall -Wextra microgpt_eth.c -o microgpt_eth
 * Run (needs CAP_NET_RAW):
 *   sudo ./microgpt_eth eth0 probe
 *   sudo ./microgpt_eth eth0 gen 15 0.5 2
 *
 * Mirrors microgpt_eth.py for callers that want a C path instead.
 */

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <linux/if_packet.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define ETH_TYPE_MG  0x4D47
#define FT_REG_WRITE 0x01
#define FT_REG_READ  0x02
#define FT_REG_RSP   0x03
#define FT_NAK       0x05
#define FT_ACK       0x06

static const uint8_t FPGA_MAC[6] = {0x02,0x00,0x00,0x4D,0x47,0x31};

#define REG_MAGIC       0x000
#define REG_VERSION     0x001
#define REG_CTRL        0x002
#define REG_STATUS      0x003
#define REG_GEN_CFG     0x004
#define REG_SEED        0x005
#define REG_OUT_BASE    0x018

static int open_iface(const char *iface, uint8_t host_mac[6], int *ifindex)
{
    int s = socket(AF_PACKET, SOCK_RAW, htons(ETH_TYPE_MG));
    if (s < 0) { perror("socket"); return -1; }

    struct ifreq ifr = {0};
    strncpy(ifr.ifr_name, iface, IFNAMSIZ - 1);
    if (ioctl(s, SIOCGIFINDEX, &ifr) < 0) { perror("SIOCGIFINDEX"); close(s); return -1; }
    *ifindex = ifr.ifr_ifindex;

    if (ioctl(s, SIOCGIFHWADDR, &ifr) < 0) { perror("SIOCGIFHWADDR"); close(s); return -1; }
    memcpy(host_mac, ifr.ifr_hwaddr.sa_data, 6);

    struct sockaddr_ll sll = {0};
    sll.sll_family   = AF_PACKET;
    sll.sll_protocol = htons(ETH_TYPE_MG);
    sll.sll_ifindex  = *ifindex;
    if (bind(s, (struct sockaddr *)&sll, sizeof(sll)) < 0) {
        perror("bind"); close(s); return -1;
    }

    struct timeval tv = { .tv_sec = 0, .tv_usec = 500000 };
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    return s;
}

static int send_frame(int s, int ifindex, const uint8_t host_mac[6],
                      const uint8_t *body, size_t blen)
{
    uint8_t buf[64 + 256];
    size_t off = 0;
    memcpy(buf + off, FPGA_MAC, 6); off += 6;
    memcpy(buf + off, host_mac, 6); off += 6;
    buf[off++] = (ETH_TYPE_MG >> 8) & 0xFF;
    buf[off++] = ETH_TYPE_MG & 0xFF;
    memcpy(buf + off, body, blen); off += blen;
    while (off < 60) buf[off++] = 0;

    struct sockaddr_ll sll = {0};
    sll.sll_family   = AF_PACKET;
    sll.sll_protocol = htons(ETH_TYPE_MG);
    sll.sll_ifindex  = ifindex;
    sll.sll_halen    = 6;
    memcpy(sll.sll_addr, FPGA_MAC, 6);

    return sendto(s, buf, off, 0, (struct sockaddr *)&sll, sizeof(sll));
}

static int reg_write(int s, int ifindex, const uint8_t host_mac[6], uint8_t seq,
                     const uint16_t *addrs, const uint32_t *data, uint8_t n)
{
    uint8_t body[2 + 8 + 16 * 8];
    size_t off = 0;
    body[off++] = FT_REG_WRITE; body[off++] = seq;
    body[off++] = n; body[off++] = seq;
    memset(body + off, 0, 6); off += 6;
    for (uint8_t i = 0; i < n; i++) {
        body[off++] =  addrs[i]        & 0xFF;
        body[off++] = (addrs[i] >> 8)  & 0xFF;
        body[off++] =  data[i]         & 0xFF;
        body[off++] = (data[i]  >> 8)  & 0xFF;
        body[off++] = (data[i]  >> 16) & 0xFF;
        body[off++] = (data[i]  >> 24) & 0xFF;
        body[off++] = 0; body[off++] = 0;
    }
    return send_frame(s, ifindex, host_mac, body, off);
}

static int reg_read(int s, int ifindex, const uint8_t host_mac[6], uint8_t seq,
                    uint16_t start_addr, uint8_t nwords, uint32_t *out)
{
    uint8_t body[2 + 8] = {FT_REG_READ, seq,
                            start_addr & 0xFF, (start_addr >> 8) & 0xFF,
                            nwords, seq, 0, 0, 0, 0};
    if (send_frame(s, ifindex, host_mac, body, sizeof(body)) < 0)
        return -1;

    uint8_t buf[2048];
    for (int tries = 0; tries < 10; tries++) {
        ssize_t n = recv(s, buf, sizeof(buf), 0);
        if (n < 24) continue;
        if (buf[12] != ((ETH_TYPE_MG >> 8) & 0xFF) || buf[13] != (ETH_TYPE_MG & 0xFF))
            continue;
        if (buf[14] == FT_NAK) {
            fprintf(stderr, "NAK seq=%u err=0x%02x\n", buf[16], buf[17]);
            return -1;
        }
        if (buf[14] != FT_REG_RSP || buf[15] != seq) continue;
        uint8_t r_n = buf[18];
        for (uint8_t i = 0; i < r_n && i < nwords; i++) {
            out[i] = (uint32_t)buf[24 + 4*i]
                   | ((uint32_t)buf[25 + 4*i] << 8)
                   | ((uint32_t)buf[26 + 4*i] << 16)
                   | ((uint32_t)buf[27 + 4*i] << 24);
        }
        return r_n;
    }
    return -1;
}

static int cmd_probe(int s, int ifindex, const uint8_t host_mac[6])
{
    uint32_t v[4];
    if (reg_read(s, ifindex, host_mac, 1, REG_MAGIC, 4, v) != 4) {
        fprintf(stderr, "probe failed\n"); return 1;
    }
    printf("magic   = 0x%08x  ('%c%c%c%c')\n", v[0],
        v[0] & 0xFF, (v[0]>>8)&0xFF, (v[0]>>16)&0xFF, (v[0]>>24)&0xFF);
    printf("version = 0x%08x\n", v[1]);
    printf("ctrl    = 0x%08x\n", v[2]);
    printf("status  = 0x%08x\n", v[3]);
    return 0;
}

static int cmd_gen(int s, int ifindex, const uint8_t host_mac[6],
                   int steps, double temperature, uint32_t seed)
{
    uint16_t temp_q88 = (uint16_t)(temperature * 256.0);
    uint32_t cfg = ((uint32_t)temp_q88 << 16) | ((uint32_t)(steps & 0xFF) << 8);
    uint16_t a[2] = {REG_GEN_CFG, REG_SEED};
    uint32_t d[2] = {cfg, seed};
    if (reg_write(s, ifindex, host_mac, 10, a, d, 2) < 0) {
        perror("reg_write cfg"); return 1;
    }
    uint16_t a2[1] = {REG_CTRL};
    uint32_t d2[1] = {0x1};
    if (reg_write(s, ifindex, host_mac, 11, a2, d2, 1) < 0) {
        perror("reg_write start"); return 1;
    }

    uint32_t status = 0;
    for (int i = 0; i < 1000; i++) {
        if (reg_read(s, ifindex, host_mac, 12, REG_STATUS, 1, &status) != 1)
            continue;
        if (status & 0x4) break;
        usleep(5000);
    }
    if (!(status & 0x4)) { fprintf(stderr, "timeout, status=0x%08x\n", status); return 1; }

    uint8_t out_len = (status >> 24) & 0xFF;
    if (out_len == 0) out_len = 1;
    uint32_t toks[16];
    int n = reg_read(s, ifindex, host_mac, 13, REG_OUT_BASE, out_len, toks);
    if (n <= 0) { fprintf(stderr, "token read failed\n"); return 1; }

    printf("status=0x%08x out_len=%u tokens=[", status, out_len);
    for (int i = 0; i < n; i++) printf(i ? ",%u" : "%u", toks[i] & 0xFF);
    printf("] name=\"");
    for (int i = 0; i < n; i++) {
        uint8_t t = toks[i] & 0xFF;
        if (t < 26) putchar('a' + t);
    }
    printf("\"\n");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <iface> probe\n"
                        "       %s <iface> gen <steps> <temperature> <seed>\n",
                argv[0], argv[0]);
        return 2;
    }
    uint8_t host_mac[6]; int ifindex;
    int s = open_iface(argv[1], host_mac, &ifindex);
    if (s < 0) return 1;

    int rc = 0;
    if (!strcmp(argv[2], "probe")) {
        rc = cmd_probe(s, ifindex, host_mac);
    } else if (!strcmp(argv[2], "gen") && argc >= 6) {
        rc = cmd_gen(s, ifindex, host_mac,
                     atoi(argv[3]), atof(argv[4]), strtoul(argv[5], NULL, 0));
    } else {
        fprintf(stderr, "unknown command\n"); rc = 2;
    }
    close(s);
    return rc;
}
