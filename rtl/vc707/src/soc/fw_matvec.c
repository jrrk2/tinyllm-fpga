/* Stage: PicoSoC weight-feed debug mode (matvec level).
 * Feed a real matvec_bfp_engine beat-by-beat over iomem with an IDENTITY
 * weight matrix (diagonal = 1.0) and x[lane] = (lane+1)*1000.  With identity
 * weights the matvec output equals x, so the TB checks out[lane] ~= x[lane]
 * with no requant golden needed.  Proves the CPU can deliver weights one beat
 * at a time and the engine computes off that gapped, MIG-free feed. */
#include <stdint.h>

#define FEED ((volatile uint32_t *)0x10000000u)
/* reg map: [0]=CTRL [1]=XM [2]=XE [4..11]=WM(256b) [12..15]=WE(128b) */

#define IN_DIM 16
#define UNIT   0x4000u   /* 2^14 ; with w_exp=1 -> 0x4000*2^(1-15)=1.0 */
#define WEXP_4 0x01010101u  /* four lanes, each exponent = 1 */

void main(void)
{
    FEED[0] = 1;                       /* pulse start_matvec (1 cycle before 1st beat) */

    for (uint32_t col = 0; col < IN_DIM; col++) {
        FEED[1] = (col + 1u) * 1000u;  /* x_mant[col] */
        FEED[2] = 15u;                 /* x_exp = 15 -> scale 2^0 */

        /* weight beat: lane==col -> 1.0, else 0.  16 lanes x 16b -> 8 words. */
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t lo = (2u * w == col)     ? UNIT : 0u;
            uint32_t hi = (2u * w + 1u == col) ? UNIT : 0u;
            FEED[4 + w] = (hi << 16) | lo;
        }
        /* per-lane weight exponents, all = 1 */
        for (uint32_t w = 0; w < 4; w++)
            FEED[12 + w] = WEXP_4;

        FEED[0] = (col == IN_DIM - 1) ? 6u : 2u;   /* push (last on final col) */
    }

    for (;;) { }
}
