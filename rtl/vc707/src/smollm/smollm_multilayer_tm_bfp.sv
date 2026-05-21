// smollm_multilayer_tm_bfp.sv — block-FP time-mux wrapper.
//
// Wraps a single smollm_layer_bfp instance (NL layers in its ROM banks
// or DDR3 image), calls it NL times per token-step, bumping layer_idx
// 0..NL-1 and feeding each layer's hidden_out back as the next layer's
// hidden_in.  The KV cache inside the layer persists per-layer because
// the layer owns the per-layer cache slice indexed by {layer_idx,…}.
//
// The caller passes the 14 layer-0 matrix base addresses (W?_m / W?_e);
// this wrapper computes per-layer offsets (base + lay_idx × per-layer
// bytes) and feeds them to the inner layer each S_PULSE so the streamer
// fetches that layer's slice from DDR3.

`include "bfp_format.svh"

`default_nettype none

module smollm_multilayer_tm_bfp #(
  parameter int    D       = 576,
  parameter int    H_Q     = 9,
  parameter int    H_KV    = 3,
  parameter int    HD      = 64,
  parameter int    FFN     = 1536,
  parameter int    MAX_CTX = 64,
  parameter int    NL      = 30,
  parameter        PREFIX  = "lbfp_full_",
  parameter int    AXI_ADDR_WIDTH = 30,
  parameter int    AXI_ID_WIDTH   = 5
)(
  input  wire                                       clk,
  input  wire                                       rst,
  input  wire                                       start,
  input  wire [10:0]                                pos,
  input  wire [6:0]                                 kv_pos,
  input  wire signed [D*BFP_MANT_W-1:0]             hidden_in_m,
  input  wire signed [(D/BFP_TILE)*BFP_EXP_W-1:0]   hidden_in_e,
  output logic signed [D*BFP_MANT_W-1:0]            hidden_out_m,
  output logic signed [(D/BFP_TILE)*BFP_EXP_W-1:0]  hidden_out_e,
  output logic                                      done,
  output wire  [31:0]                               weight_hash,
  // Pass-through host BRAM write port (kinds 0..3 land in this layer's
  // gammas; other kinds pass through harmlessly).
  input  wire [4:0]                                 wr_kind,
  input  wire [17:0]                                wr_addr,
  input  wire [15:0]                                wr_data,
  input  wire                                       wr_en,
  input  wire                                       clk_wr,
  output wire [15:0]                                wr_rdata,
  // Per-layer hidden-state snapshot.  During a run, the hidden_out of the
  // layer whose index == snap_layer_sel, at token position pos == snap_step_sel,
  // is latched into snap_m/snap_e.  Host reads it back via wr_kind 12 (mantissa)
  // / 13 (per-tile exponent) — see host/fpga_per_layer_dump.py.  One layer per
  // run (host re-runs per layer); FF-backed, no extra BRAM.
  input  wire [4:0]                                 snap_layer_sel,
  input  wire [10:0]                                snap_step_sel,
  // Freeze: when set, the engine HALTS at layer snap_layer_sel of token-step
  // snap_step_sel instead of running the rest of the stack — so the inner
  // layer's hout (read via wr_kind 10/11) is left holding that layer's output.
  // `done` is then held high so the autoregress drains its remaining token
  // steps without re-running the layer (generated tokens are garbage — that's
  // expected; this mode is for inspecting one frozen layer, e.g. via ILA).
  // Cleared by rst (= core reset or restart pulse).
  input  wire                                       freeze_en,
  // Absolute clock-count trigger.  cyc counts core_clk cycles from rst (=
  // restart).  When trig_cyc_en, the engine freezes at the first layer
  // boundary on/after cyc == trig_cyc — a coarse "stop near cycle N" that's
  // timing-safe (freeze only acts at the S_NEXT boundary, low fanout).
  input  wire                                       trig_cyc_en,
  input  wire [31:0]                                trig_cyc,
  output wire [31:0]                                dbg_cyc,        // cycles since rst
  output wire [4:0]                                 dbg_cur_layer,  // lay_idx when frozen
  output wire                                       dbg_frozen,     // freeze latched
  // DDR3 streamer ports.  Layer-0 base addresses for each weight
  // matrix — this module derives per-layer bases as `base + lay_idx
  // × W?_*_PL` internally.
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WQ_m,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WQ_e,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WK_m,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WK_e,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WV_m,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WV_e,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WO_m,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WO_e,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WG_m,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WG_e,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WU_m,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WU_e,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WDN_m,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_WDN_e,
  input  wire                                       clk_axi,
  input  wire                                       rst_axi,
  output wire                                       m_axi_arvalid,
  input  wire                                       m_axi_arready,
  output wire [AXI_ID_WIDTH-1:0]                    m_axi_arid,
  output wire [AXI_ADDR_WIDTH-1:0]                  m_axi_araddr,
  output wire [7:0]                                 m_axi_arlen,
  output wire [2:0]                                 m_axi_arsize,
  output wire [1:0]                                 m_axi_arburst,
  output wire                                       m_axi_arlock,
  output wire [3:0]                                 m_axi_arcache,
  output wire [2:0]                                 m_axi_arprot,
  output wire [3:0]                                 m_axi_arqos,
  input  wire                                       m_axi_rvalid,
  output wire                                       m_axi_rready,
  input  wire [AXI_ID_WIDTH-1:0]                    m_axi_rid,
  input  wire [511:0]                               m_axi_rdata,
  input  wire [1:0]                                 m_axi_rresp,
  input  wire                                       m_axi_rlast
);

  localparam int NT_D = D / BFP_TILE;

  // Per-layer slice sizes (bytes) within each matrix region of DDR3.
  // Mantissa entry = LANES × 16 b = 32 B; exp entry = LANES × 8 b = 16 B.
  // Per-layer = CHUNKS_OUT × D_in entries.
  localparam int W_Q_M_PL  = (D       / 16) * D    * 32;
  localparam int W_Q_E_PL  = (D       / 16) * (D    / 16) * 16;
  localparam int W_K_M_PL  = ((H_KV*HD)/ 16) * D    * 32;
  localparam int W_K_E_PL  = ((H_KV*HD)/ 16) * (D    / 16) * 16;
  localparam int W_V_M_PL  = ((H_KV*HD)/ 16) * D    * 32;
  localparam int W_V_E_PL  = ((H_KV*HD)/ 16) * (D    / 16) * 16;
  localparam int W_O_M_PL  = (D       / 16) * D    * 32;
  localparam int W_O_E_PL  = (D       / 16) * (D    / 16) * 16;
  localparam int W_G_M_PL  = (FFN     / 16) * D    * 32;
  localparam int W_G_E_PL  = (FFN     / 16) * (D    / 16) * 16;
  localparam int W_U_M_PL  = (FFN     / 16) * D    * 32;
  localparam int W_U_E_PL  = (FFN     / 16) * (D    / 16) * 16;
  localparam int W_DN_M_PL = (D       / 16) * FFN  * 32;
  localparam int W_DN_E_PL = (D       / 16) * (FFN  / 16) * 16;

  // Hidden state shuttle between layers.
  logic signed [D*BFP_MANT_W-1:0]      h_state_m;
  logic signed [NT_D*BFP_EXP_W-1:0]    h_state_e;

  // Inner layer wires.
  logic                                lay_start;
  // Layer index — was hardcoded 5 bits, fits NL=30 (135M) and NL=32
  // (360M) exactly but no headroom beyond.  Now derived from NL so any
  // future SmolLM2 variant scales automatically.
  logic [$clog2(NL+1)-1:0]             lay_idx;
  wire  signed [D*BFP_MANT_W-1:0]      lay_hidden_out_m;
  wire  signed [NT_D*BFP_EXP_W-1:0]    lay_hidden_out_e;
  wire                                 lay_done;
  wire  [6:0]                          ignore_state;
  wire  [11:0]                         ignore_cnt;
  wire  [6:0]                          ignore_chunk;

  // Per-layer base offsets (registered for timing).
  logic [AXI_ADDR_WIDTH-1:0] lay_base_WQ_m, lay_base_WQ_e;
  logic [AXI_ADDR_WIDTH-1:0] lay_base_WK_m, lay_base_WK_e;
  logic [AXI_ADDR_WIDTH-1:0] lay_base_WV_m, lay_base_WV_e;
  logic [AXI_ADDR_WIDTH-1:0] lay_base_WO_m, lay_base_WO_e;
  logic [AXI_ADDR_WIDTH-1:0] lay_base_WG_m, lay_base_WG_e;
  logic [AXI_ADDR_WIDTH-1:0] lay_base_WU_m, lay_base_WU_e;
  logic [AXI_ADDR_WIDTH-1:0] lay_base_WDN_m, lay_base_WDN_e;

  always_comb begin
    lay_base_WQ_m  = ws_base_WQ_m  + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_Q_M_PL);
    lay_base_WQ_e  = ws_base_WQ_e  + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_Q_E_PL);
    lay_base_WK_m  = ws_base_WK_m  + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_K_M_PL);
    lay_base_WK_e  = ws_base_WK_e  + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_K_E_PL);
    lay_base_WV_m  = ws_base_WV_m  + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_V_M_PL);
    lay_base_WV_e  = ws_base_WV_e  + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_V_E_PL);
    lay_base_WO_m  = ws_base_WO_m  + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_O_M_PL);
    lay_base_WO_e  = ws_base_WO_e  + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_O_E_PL);
    lay_base_WG_m  = ws_base_WG_m  + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_G_M_PL);
    lay_base_WG_e  = ws_base_WG_e  + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_G_E_PL);
    lay_base_WU_m  = ws_base_WU_m  + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_U_M_PL);
    lay_base_WU_e  = ws_base_WU_e  + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_U_E_PL);
    lay_base_WDN_m = ws_base_WDN_m + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_DN_M_PL);
    lay_base_WDN_e = ws_base_WDN_e + AXI_ADDR_WIDTH'(lay_idx) * AXI_ADDR_WIDTH'(W_DN_E_PL);
  end

  smollm_layer_bfp #(
    .D(D), .H_Q(H_Q), .H_KV(H_KV), .HD(HD),
    .FFN(FFN), .MAX_CTX(MAX_CTX), .NL(NL), .PREFIX(PREFIX),
    .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
    .AXI_ID_WIDTH(AXI_ID_WIDTH)
  ) i_lay (
    .clk(clk), .rst(rst),
    .start(lay_start),
    .pos(pos), .kv_pos(kv_pos),
    .layer_idx(lay_idx),
    .hidden_in_m(h_state_m), .hidden_in_e(h_state_e),
    .hidden_out_m(lay_hidden_out_m), .hidden_out_e(lay_hidden_out_e),
    .done(lay_done),
    .wr_kind(wr_kind), .wr_addr(wr_addr), .wr_data(wr_data), .wr_en(wr_en),
    .clk_wr(clk_wr), .wr_rdata(wr_rdata),
    .snap_layer_sel(snap_layer_sel), .snap_step_sel(snap_step_sel),
    .ws_base_WQ_m(lay_base_WQ_m),   .ws_base_WQ_e(lay_base_WQ_e),
    .ws_base_WK_m(lay_base_WK_m),   .ws_base_WK_e(lay_base_WK_e),
    .ws_base_WV_m(lay_base_WV_m),   .ws_base_WV_e(lay_base_WV_e),
    .ws_base_WO_m(lay_base_WO_m),   .ws_base_WO_e(lay_base_WO_e),
    .ws_base_WG_m(lay_base_WG_m),   .ws_base_WG_e(lay_base_WG_e),
    .ws_base_WU_m(lay_base_WU_m),   .ws_base_WU_e(lay_base_WU_e),
    .ws_base_WDN_m(lay_base_WDN_m), .ws_base_WDN_e(lay_base_WDN_e),
    .clk_axi(clk_axi), .rst_axi(rst_axi),
    .m_axi_arvalid(m_axi_arvalid),
    .m_axi_arready(m_axi_arready),
    .m_axi_arid   (m_axi_arid),
    .m_axi_araddr (m_axi_araddr),
    .m_axi_arlen  (m_axi_arlen),
    .m_axi_arsize (m_axi_arsize),
    .m_axi_arburst(m_axi_arburst),
    .m_axi_arlock (m_axi_arlock),
    .m_axi_arcache(m_axi_arcache),
    .m_axi_arprot (m_axi_arprot),
    .m_axi_arqos  (m_axi_arqos),
    .m_axi_rvalid (m_axi_rvalid),
    .m_axi_rready (m_axi_rready),
    .m_axi_rid    (m_axi_rid),
    .m_axi_rdata  (m_axi_rdata),
    .m_axi_rresp  (m_axi_rresp),
    .m_axi_rlast  (m_axi_rlast),
    .dbg_state(ignore_state), .dbg_cnt(ignore_cnt), .dbg_chunk(ignore_chunk),
    .weight_hash(weight_hash)
  );

  // ---------------------------------------------------------------------------
  // FSM — unchanged from the BRAM-only version.
  // ---------------------------------------------------------------------------
  typedef enum logic [3:0] {
    S_IDLE, S_LATCH, S_PULSE, S_WAIT, S_CAPTURE, S_NEXT, S_DONE, S_FROZEN
  } st_t;
  st_t state;
  logic frozen;        // latched at the trigger (counter match or cycle count); held until rst
  logic [31:0] cyc;    // free-running core_clk cycle counter, reset on rst (= restart)
  assign dbg_cyc       = cyc;
  assign dbg_cur_layer = lay_idx[4:0];
  assign dbg_frozen    = frozen;

  always_ff @(posedge clk) begin
    if (rst) begin
      state        <= S_IDLE;
      lay_idx      <= '0;
      lay_start    <= 1'b0;
      h_state_m    <= '0;
      h_state_e    <= '0;
      hidden_out_m <= '0;
      hidden_out_e <= '0;
      frozen       <= 1'b0;
      cyc          <= 32'd0;
      done         <= 1'b0;
    end else begin
      lay_start <= 1'b0;
      done      <= 1'b0;
      if (!frozen) cyc <= cyc + 32'd1;   // stop counting once halted
      case (state)
        S_IDLE: if (start) begin
          h_state_m <= hidden_in_m;
          h_state_e <= hidden_in_e;
          lay_idx   <= '0;
          state     <= S_LATCH;
        end
        S_LATCH: state <= S_PULSE;
        S_PULSE: begin
          lay_start <= 1'b1;
          state     <= S_WAIT;
        end
        S_WAIT: if (lay_done) state <= S_CAPTURE;
        S_CAPTURE: begin
          h_state_m <= lay_hidden_out_m;
          h_state_e <= lay_hidden_out_e;
          state     <= S_NEXT;
        end
        S_NEXT: begin
          if ((freeze_en && lay_idx == snap_layer_sel && pos == snap_step_sel) ||
              (trig_cyc_en && cyc >= trig_cyc)) begin
            // Freeze at this layer boundary — either the programmed (layer,step)
            // counter match, or the first boundary on/after the absolute cycle
            // trigger.  The inner layer's stage RAMs (and hout, wr_kind 10/11)
            // hold this layer's values.  Hold done in S_FROZEN so the
            // autoregress drains the rest without re-running the layer.
            frozen       <= 1'b1;
            hidden_out_m <= h_state_m;
            hidden_out_e <= h_state_e;
            state        <= S_DONE;
          end else if (lay_idx == NL - 1) begin
            hidden_out_m <= h_state_m;
            hidden_out_e <= h_state_e;
            state        <= S_DONE;
          end else begin
            lay_idx <= lay_idx + 1'b1;
            state   <= S_PULSE;
          end
        end
        S_DONE: begin
          done  <= 1'b1;
          state <= frozen ? S_FROZEN : S_IDLE;
        end
        S_FROZEN: begin
          done  <= 1'b1;   // held high; `start` ignored until rst clears frozen
          state <= S_FROZEN;
        end
        default: state <= S_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
