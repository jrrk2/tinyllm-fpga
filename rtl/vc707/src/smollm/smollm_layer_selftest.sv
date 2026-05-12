// smollm_layer_selftest.sv — FPGA wrapper around smollm_layer.
//
// Drives one layer call at boot using the test data baked into
// layer_test_data.svh, latches the first 16 lanes of hidden_out for the
// host to read via the register map, and a `done` flag.
//
// Two modes:
//   - default (BROM): weights via $readmemh — small dims fit FPGA BRAM
//   - `MICROGPT_DDR3_WEIGHTS`: weights streamed from DDR3 via AXI;
//     forwards the AXI master ports + clk_axi/rst_axi to the layer.
//
// Same pattern as matvec_selftest / rmsnorm_selftest.

`ifdef MICROGPT_DDR3_WEIGHTS
  `include "layer_ddr3_bases.svh"
`endif

`default_nettype none

module smollm_layer_selftest #(
  // SmolLM2-135M real dims.  Buffer refactor (MV_CAPTURE) makes the
  // *_buf arrays BRAM-inferable, so D=576 fits VC707 again.  hidden1_buf
  // and down_buf stay in registers because the hidden_out generate-loop
  // reads all D entries in parallel — that's ~18K FFs, acceptable.
  parameter int D       = 576,
  parameter int H_Q     = 9,
  parameter int H_KV    = 3,
  parameter int HD      = 64,
  parameter int FFN     = 1536,
  parameter int MAX_CTX = 4,
  // Result snapshot: first 64 lanes of hidden_out via the existing
  // 32-word regmap window (0x1D0..0x1EF, 2 lanes per 32-bit word).
  // Selftest_verify reads + checks 64 lanes against the reference.
  parameter int RESULT_LANES = 64
)(
  input  wire                          clk,        // core_clk (50 MHz)
  input  wire                          rst,
  input  wire                          restart,
  output logic [RESULT_LANES*16-1:0]   result,
  output logic                         done

`ifdef MICROGPT_DDR3_WEIGHTS
  ,
  // ILA probe taps forwarded from the inner smollm_layer.
  output wire [2:0]    ila_ws_state_axi,
  output wire          ila_start_load_axi,
  output wire [1:0]    ila_tile_idx,
  output wire [6:0]    ila_beat_idx,
  output wire [4:0]    ila_state,
  output wire [2:0]    ila_mv_phase,
  output wire [10:0]   ila_cnt,
  output wire [6:0]    ila_chunk,
  output wire [2:0]    ila_ws_matvec_id,
  output wire          ila_ws_load_req,
  output wire          ila_ws_ready,
  output wire [10:0]   ila_ws_rd_addr,
  output wire [127:0]  ila_ws_weight_data,
  output wire [127:0]  ila_eng_w,
  output wire [15:0]   ila_eng_in_value,
  output wire          ila_eng_in_valid,
  output wire          ila_eng_in_last,
  output wire          ila_eng_acc_clear,
  output wire          ila_eng_out_valid,
  // Diagnostics forwarded out for the host regmap.
  output wire [29:0]                   dbg_first_araddr,
  output wire [511:0]                  dbg_first_rdata,
  output wire                          dbg_first_ar_seen,
  output wire                          dbg_first_r_seen,
  output wire [511:0]                  dbg_eng_w_packed,
  output wire [511:0]                  dbg_wd_packed,
  output wire [63:0]                   dbg_in_value_packed,
  output wire                          dbg_snap_done_o,
  input  wire                          clk_axi,    // ui_clk (200 MHz)
  input  wire                          rst_axi,
  // AXI4 read master to MIG
  output wire                          m_axi_arvalid,
  input  wire                          m_axi_arready,
  output wire [4:0]                    m_axi_arid,
  output wire [29:0]                   m_axi_araddr,
  output wire [7:0]                    m_axi_arlen,
  output wire [2:0]                    m_axi_arsize,
  output wire [1:0]                    m_axi_arburst,
  output wire                          m_axi_arlock,
  output wire [3:0]                    m_axi_arcache,
  output wire [2:0]                    m_axi_arprot,
  output wire [3:0]                    m_axi_arqos,
  input  wire                          m_axi_rvalid,
  output wire                          m_axi_rready,
  input  wire  [4:0]                   m_axi_rid,
  input  wire  [511:0]                 m_axi_rdata,
  input  wire  [1:0]                   m_axi_rresp,
  input  wire                          m_axi_rlast
`endif
);

  logic                lay_start;
  logic [10:0]         lay_pos;
  logic [4:0]          lay_kv_pos;
  wire  [D*16-1:0]     lay_hidden_in;
  wire  [D*16-1:0]     lay_hidden_out;
  logic                lay_done;

  // Hidden_in baked in as a case-statement ROM.  Previously $readmemh
  // into an unpacked array with a generate-loop reader was silently
  // dropped by Vivado synth ("invalid memory name") — sf_hid_rom ended
  // up all zero, so RMSNorm input was 0 and matvec MAC'd zero×W.
`include "layer_hidden_in_packed.svh"
  genvar gh;
  generate
    for (gh = 0; gh < D; gh++)
      assign lay_hidden_in[gh*16 +: 16] = layer_hidden_in_lut(gh);
  endgenerate

  smollm_layer #(
    .D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD), .FFN(FFN), .MAX_CTX(MAX_CTX)
`ifdef MICROGPT_DDR3_WEIGHTS
    ,
    .BASE_Q   (`MICROGPT_LAYER_BASE_Q),
    .BASE_K   (`MICROGPT_LAYER_BASE_K),
    .BASE_V   (`MICROGPT_LAYER_BASE_V),
    .BASE_O   (`MICROGPT_LAYER_BASE_O),
    .BASE_GATE(`MICROGPT_LAYER_BASE_GATE),
    .BASE_UP  (`MICROGPT_LAYER_BASE_UP),
    .BASE_DOWN(`MICROGPT_LAYER_BASE_DOWN)
`endif
  ) i_layer (
    .clk(clk), .rst(rst | restart),
    .start(lay_start),
    .pos(lay_pos), .kv_pos(lay_kv_pos),
    .hidden_in(lay_hidden_in),
    .hidden_out(lay_hidden_out),
    .done(lay_done)
`ifdef MICROGPT_DDR3_WEIGHTS
    ,
    .ila_ws_state_axi  (ila_ws_state_axi),
    .ila_start_load_axi(ila_start_load_axi),
    .ila_tile_idx      (ila_tile_idx),
    .ila_beat_idx      (ila_beat_idx),
    .ila_state         (ila_state),
    .ila_mv_phase      (ila_mv_phase),
    .ila_cnt           (ila_cnt),
    .ila_chunk         (ila_chunk),
    .ila_ws_matvec_id  (ila_ws_matvec_id),
    .ila_ws_load_req   (ila_ws_load_req),
    .ila_ws_ready      (ila_ws_ready),
    .ila_ws_rd_addr    (ila_ws_rd_addr),
    .ila_ws_weight_data(ila_ws_weight_data),
    .ila_eng_w         (ila_eng_w),
    .ila_eng_in_value  (ila_eng_in_value),
    .ila_eng_in_valid  (ila_eng_in_valid),
    .ila_eng_in_last   (ila_eng_in_last),
    .ila_eng_acc_clear (ila_eng_acc_clear),
    .ila_eng_out_valid (ila_eng_out_valid),
    .dbg_first_araddr (dbg_first_araddr),
    .dbg_first_rdata  (dbg_first_rdata),
    .dbg_first_ar_seen(dbg_first_ar_seen),
    .dbg_first_r_seen (dbg_first_r_seen),
    .dbg_eng_w_packed (dbg_eng_w_packed),
    .dbg_wd_packed    (dbg_wd_packed),
    .dbg_in_value_packed(dbg_in_value_packed),
    .dbg_snap_done_o  (dbg_snap_done_o),
    .clk_axi(clk_axi), .rst_axi(rst_axi),
    .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
    .m_axi_arid(m_axi_arid),       .m_axi_araddr(m_axi_araddr),
    .m_axi_arlen(m_axi_arlen),     .m_axi_arsize(m_axi_arsize),
    .m_axi_arburst(m_axi_arburst), .m_axi_arlock(m_axi_arlock),
    .m_axi_arcache(m_axi_arcache), .m_axi_arprot(m_axi_arprot),
    .m_axi_arqos(m_axi_arqos),
    .m_axi_rvalid(m_axi_rvalid),   .m_axi_rready(m_axi_rready),
    .m_axi_rid(m_axi_rid),         .m_axi_rdata(m_axi_rdata),
    .m_axi_rresp(m_axi_rresp),     .m_axi_rlast(m_axi_rlast)
`endif
  );

  always_ff @(posedge clk) begin
    if (rst | restart) begin
      lay_start    <= 1'b0;
      lay_pos      <= 11'd3;
      lay_kv_pos   <= 5'd3;
      result       <= '0;
      done         <= 1'b0;
    end else begin
      if (!done) lay_start <= 1'b1;
      if (lay_done && !done) begin
        // Capture only the first RESULT_LANES lanes for the host regmap.
        result <= lay_hidden_out[RESULT_LANES*16-1:0];
        done   <= 1'b1;
        lay_start <= 1'b0;
      end
    end
  end

endmodule

`default_nettype wire
