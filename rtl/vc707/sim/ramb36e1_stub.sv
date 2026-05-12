// ramb36e1_stub.sv — Verilator-only behavioural model of the Xilinx
// 7-series RAMB36E1 primitive.  Just enough to simulate the
// auto-generated brom_*.sv wrappers (host/gen_brom_sv.py) which use
// 2K x 18 TDP read-only mode with INIT_xx initialization.
//
// Goal: avoid a separate `ifdef VERILATOR fallback inside each brom
// module so test coverage runs through the EXACT same RTL path that
// Vivado synthesizes.  Vivado uses the real unisim primitive; this stub
// supplies an equivalent for sim.
//
// Limitations (acceptable for our use):
//   * Only port A read is modelled (port B is unused by brom).
//   * Only 2K x 18 mode (so READ_WIDTH_A == 18; the 2 parity bits are
//     not modelled — INITP_xx are accepted but ignored).
//   * Address mapping: ADDRARDADDR[14:4] selects 2K-deep entry.
//     ADDRARDADDR[15] (cascade) and [3:0] (sub-bit-position in narrow
//     widths) are ignored.
//   * Single-cycle read latency (DOA_REG = 0).  REGCEAREGCE / RSTREG*
//     are ignored — we do not model the optional output register.
//   * Writes are no-ops (WRITE_WIDTH_A == 0 in our brom).
//
// If Vivado later introduces brom variants with different widths or
// dual-port writes, extend this stub rather than reverting to the
// behavioural fallback.

`ifdef VERILATOR

`default_nettype none

module RAMB36E1 #(
  parameter             RAM_MODE        = "TDP",
  parameter integer     READ_WIDTH_A    = 0,
  parameter integer     WRITE_WIDTH_A   = 0,
  parameter integer     READ_WIDTH_B    = 0,
  parameter integer     WRITE_WIDTH_B   = 0,
  parameter integer     DOA_REG         = 0,
  parameter integer     DOB_REG         = 0,
  parameter             WRITE_MODE_A    = "WRITE_FIRST",
  parameter             WRITE_MODE_B    = "WRITE_FIRST",
  parameter             EN_ECC_READ     = "FALSE",
  parameter             EN_ECC_WRITE    = "FALSE",
  parameter             RAM_EXTENSION_A = "NONE",
  parameter             RAM_EXTENSION_B = "NONE",
  parameter             SIM_DEVICE      = "VIRTEX7",
  parameter [35:0]      INIT_A          = 36'h0,
  parameter [35:0]      INIT_B          = 36'h0,
  parameter [35:0]      SRVAL_A         = 36'h0,
  parameter [35:0]      SRVAL_B         = 36'h0,
  // 128 × 256-bit data init strings (full RAMB36 storage = 32 Kbit)
  parameter [255:0]     INIT_00 = 256'h0, INIT_01 = 256'h0, INIT_02 = 256'h0, INIT_03 = 256'h0,
  parameter [255:0]     INIT_04 = 256'h0, INIT_05 = 256'h0, INIT_06 = 256'h0, INIT_07 = 256'h0,
  parameter [255:0]     INIT_08 = 256'h0, INIT_09 = 256'h0, INIT_0A = 256'h0, INIT_0B = 256'h0,
  parameter [255:0]     INIT_0C = 256'h0, INIT_0D = 256'h0, INIT_0E = 256'h0, INIT_0F = 256'h0,
  parameter [255:0]     INIT_10 = 256'h0, INIT_11 = 256'h0, INIT_12 = 256'h0, INIT_13 = 256'h0,
  parameter [255:0]     INIT_14 = 256'h0, INIT_15 = 256'h0, INIT_16 = 256'h0, INIT_17 = 256'h0,
  parameter [255:0]     INIT_18 = 256'h0, INIT_19 = 256'h0, INIT_1A = 256'h0, INIT_1B = 256'h0,
  parameter [255:0]     INIT_1C = 256'h0, INIT_1D = 256'h0, INIT_1E = 256'h0, INIT_1F = 256'h0,
  parameter [255:0]     INIT_20 = 256'h0, INIT_21 = 256'h0, INIT_22 = 256'h0, INIT_23 = 256'h0,
  parameter [255:0]     INIT_24 = 256'h0, INIT_25 = 256'h0, INIT_26 = 256'h0, INIT_27 = 256'h0,
  parameter [255:0]     INIT_28 = 256'h0, INIT_29 = 256'h0, INIT_2A = 256'h0, INIT_2B = 256'h0,
  parameter [255:0]     INIT_2C = 256'h0, INIT_2D = 256'h0, INIT_2E = 256'h0, INIT_2F = 256'h0,
  parameter [255:0]     INIT_30 = 256'h0, INIT_31 = 256'h0, INIT_32 = 256'h0, INIT_33 = 256'h0,
  parameter [255:0]     INIT_34 = 256'h0, INIT_35 = 256'h0, INIT_36 = 256'h0, INIT_37 = 256'h0,
  parameter [255:0]     INIT_38 = 256'h0, INIT_39 = 256'h0, INIT_3A = 256'h0, INIT_3B = 256'h0,
  parameter [255:0]     INIT_3C = 256'h0, INIT_3D = 256'h0, INIT_3E = 256'h0, INIT_3F = 256'h0,
  parameter [255:0]     INIT_40 = 256'h0, INIT_41 = 256'h0, INIT_42 = 256'h0, INIT_43 = 256'h0,
  parameter [255:0]     INIT_44 = 256'h0, INIT_45 = 256'h0, INIT_46 = 256'h0, INIT_47 = 256'h0,
  parameter [255:0]     INIT_48 = 256'h0, INIT_49 = 256'h0, INIT_4A = 256'h0, INIT_4B = 256'h0,
  parameter [255:0]     INIT_4C = 256'h0, INIT_4D = 256'h0, INIT_4E = 256'h0, INIT_4F = 256'h0,
  parameter [255:0]     INIT_50 = 256'h0, INIT_51 = 256'h0, INIT_52 = 256'h0, INIT_53 = 256'h0,
  parameter [255:0]     INIT_54 = 256'h0, INIT_55 = 256'h0, INIT_56 = 256'h0, INIT_57 = 256'h0,
  parameter [255:0]     INIT_58 = 256'h0, INIT_59 = 256'h0, INIT_5A = 256'h0, INIT_5B = 256'h0,
  parameter [255:0]     INIT_5C = 256'h0, INIT_5D = 256'h0, INIT_5E = 256'h0, INIT_5F = 256'h0,
  parameter [255:0]     INIT_60 = 256'h0, INIT_61 = 256'h0, INIT_62 = 256'h0, INIT_63 = 256'h0,
  parameter [255:0]     INIT_64 = 256'h0, INIT_65 = 256'h0, INIT_66 = 256'h0, INIT_67 = 256'h0,
  parameter [255:0]     INIT_68 = 256'h0, INIT_69 = 256'h0, INIT_6A = 256'h0, INIT_6B = 256'h0,
  parameter [255:0]     INIT_6C = 256'h0, INIT_6D = 256'h0, INIT_6E = 256'h0, INIT_6F = 256'h0,
  parameter [255:0]     INIT_70 = 256'h0, INIT_71 = 256'h0, INIT_72 = 256'h0, INIT_73 = 256'h0,
  parameter [255:0]     INIT_74 = 256'h0, INIT_75 = 256'h0, INIT_76 = 256'h0, INIT_77 = 256'h0,
  parameter [255:0]     INIT_78 = 256'h0, INIT_79 = 256'h0, INIT_7A = 256'h0, INIT_7B = 256'h0,
  parameter [255:0]     INIT_7C = 256'h0, INIT_7D = 256'h0, INIT_7E = 256'h0, INIT_7F = 256'h0,
  // 16 × 256-bit parity init strings (4 Kbit) — accepted, not modelled.
  parameter [255:0]     INITP_00 = 256'h0, INITP_01 = 256'h0, INITP_02 = 256'h0, INITP_03 = 256'h0,
  parameter [255:0]     INITP_04 = 256'h0, INITP_05 = 256'h0, INITP_06 = 256'h0, INITP_07 = 256'h0,
  parameter [255:0]     INITP_08 = 256'h0, INITP_09 = 256'h0, INITP_0A = 256'h0, INITP_0B = 256'h0,
  parameter [255:0]     INITP_0C = 256'h0, INITP_0D = 256'h0, INITP_0E = 256'h0, INITP_0F = 256'h0
)(
  input  wire         CLKARDCLK,
  input  wire         CLKBWRCLK,
  input  wire         ENARDEN,
  input  wire         ENBWREN,
  input  wire         REGCEAREGCE,
  input  wire         REGCEB,
  input  wire         RSTRAMARSTRAM,
  input  wire         RSTRAMB,
  input  wire         RSTREGARSTREG,
  input  wire         RSTREGB,
  input  wire         CASCADEINA,
  input  wire         CASCADEINB,
  input  wire         INJECTSBITERR,
  input  wire         INJECTDBITERR,
  input  wire [15:0]  ADDRARDADDR,
  input  wire [15:0]  ADDRBWRADDR,
  input  wire [31:0]  DIADI,
  input  wire [3:0]   DIPADIP,
  input  wire [31:0]  DIBDI,
  input  wire [3:0]   DIPBDIP,
  input  wire [3:0]   WEA,
  input  wire [7:0]   WEBWE,
  output reg  [31:0]  DOADO,
  output reg  [3:0]   DOPADOP,
  output reg  [31:0]  DOBDO,
  output reg  [3:0]   DOPBDOP,
  output wire         CASCADEOUTA,
  output wire         CASCADEOUTB,
  output wire         SBITERR,
  output wire         DBITERR,
  output wire [8:0]   ECCPARITY,
  output wire [8:0]   RDADDRECC
);

  // ------------------------------------------------------------------
  // Pack the 128 INIT_xx parameters into one wide bit vector.  The
  // underlying RAMB36 storage is 32 Kbit organised low→high as
  // INIT_00, INIT_01, ..., INIT_7F.  In 2Kx18 mode entry N occupies
  // bits [N*18 : N*18+17] of the storage; bits 0..15 are data, 16..17
  // are parity (INITP_xx, ignored here).  We approximate by reading
  // entry N's data as bits [N*16 : N*16+15] of a 16-bit-strided pack
  // — which matches what host/gen_brom_sv.py emits (the generator
  // packs 16 entries per INIT_xx, lane j at bit positions j*16..j*16+15).
  // ------------------------------------------------------------------
  localparam [128*256-1:0] PACKED = {
    INIT_7F, INIT_7E, INIT_7D, INIT_7C, INIT_7B, INIT_7A, INIT_79, INIT_78,
    INIT_77, INIT_76, INIT_75, INIT_74, INIT_73, INIT_72, INIT_71, INIT_70,
    INIT_6F, INIT_6E, INIT_6D, INIT_6C, INIT_6B, INIT_6A, INIT_69, INIT_68,
    INIT_67, INIT_66, INIT_65, INIT_64, INIT_63, INIT_62, INIT_61, INIT_60,
    INIT_5F, INIT_5E, INIT_5D, INIT_5C, INIT_5B, INIT_5A, INIT_59, INIT_58,
    INIT_57, INIT_56, INIT_55, INIT_54, INIT_53, INIT_52, INIT_51, INIT_50,
    INIT_4F, INIT_4E, INIT_4D, INIT_4C, INIT_4B, INIT_4A, INIT_49, INIT_48,
    INIT_47, INIT_46, INIT_45, INIT_44, INIT_43, INIT_42, INIT_41, INIT_40,
    INIT_3F, INIT_3E, INIT_3D, INIT_3C, INIT_3B, INIT_3A, INIT_39, INIT_38,
    INIT_37, INIT_36, INIT_35, INIT_34, INIT_33, INIT_32, INIT_31, INIT_30,
    INIT_2F, INIT_2E, INIT_2D, INIT_2C, INIT_2B, INIT_2A, INIT_29, INIT_28,
    INIT_27, INIT_26, INIT_25, INIT_24, INIT_23, INIT_22, INIT_21, INIT_20,
    INIT_1F, INIT_1E, INIT_1D, INIT_1C, INIT_1B, INIT_1A, INIT_19, INIT_18,
    INIT_17, INIT_16, INIT_15, INIT_14, INIT_13, INIT_12, INIT_11, INIT_10,
    INIT_0F, INIT_0E, INIT_0D, INIT_0C, INIT_0B, INIT_0A, INIT_09, INIT_08,
    INIT_07, INIT_06, INIT_05, INIT_04, INIT_03, INIT_02, INIT_01, INIT_00
  };

  // 2K-deep, 16-bit-wide storage (parity ignored).
  reg [15:0] mem [0:2047];
  integer    idx;
  initial begin
    for (idx = 0; idx < 2048; idx = idx + 1)
      mem[idx] = PACKED[idx*16 +: 16];
    DOADO   = 32'h0;
    DOPADOP = 4'h0;
    DOBDO   = 32'h0;
    DOPBDOP = 4'h0;
  end

  // Port A: 1-cycle synchronous read.  In 2Kx18 mode the 11-bit row
  // index lives in ADDRARDADDR[14:4] (gen_brom_sv.py drives it as
  // {1'b1, addr_11bit, 4'b0}).
  always @(posedge CLKARDCLK) begin
    if (ENARDEN) begin
      DOADO[15:0]   <= mem[ADDRARDADDR[14:4]];
      DOADO[31:16]  <= 16'h0;
      DOPADOP       <= 4'h0;
    end
  end

  // Unused outputs
  assign CASCADEOUTA = 1'b0;
  assign CASCADEOUTB = 1'b0;
  assign SBITERR     = 1'b0;
  assign DBITERR     = 1'b0;
  assign ECCPARITY   = 9'h0;
  assign RDADDRECC   = 9'h0;

endmodule

`default_nettype wire

`endif  // VERILATOR
