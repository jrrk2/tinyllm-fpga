// embed_lookup_bfp.sv — BFP embedding-table row lookup.
//
//   in:  token_id (16-bit), start pulse
//   out: hidden_m (D mantissas packed), hidden_e (NT_D exps packed), done
//
//   STREAM_LOOKUP=0 (selftest): the full VOCAB × D table is held in
//   on-chip BRAM, loaded via $readmemh from the
//   lbfp_full_EMBED_LOOKUP_{m,e}.hex files baked by
//   gen_smollm_blockfp_full.py.  One row read = D + NT_D BRAM cycles.
//
//   STREAM_LOOKUP=1 (VC707 deploy): the table lives in DDR3.  On `start`
//   the module issues an AXI burst to fetch the requested token's row
//   (BEATS_M=ceil(D*2/64) mantissa beats + 1 exponent beat) into a small
//   on-chip staging buffer, then drives hidden_m / hidden_e by slicing
//   the buffered beats.  Mantissa rows are contiguous (1152 B for D=576)
//   and natively 64 B-aligned per token; exponent rows are 36 B per row
//   and the host pads each one to AXI_DATA_WIDTH/8 bytes (= 64 B) so a
//   single AR can land the whole row in one beat.

`include "bfp_format.svh"

`default_nettype none

module embed_lookup_bfp #(
  parameter int D             = 576,
  parameter int VOCAB         = 49152,
  parameter     PREFIX        = "lbfp_",
  parameter bit STREAM_LOOKUP = 1'b0,
  parameter int AXI_ADDR_WIDTH = 30,
  parameter int AXI_ID_WIDTH   = 5
)(
  input  wire                                       clk,
  input  wire                                       rst,
  input  wire                                       start,
  input  wire [15:0]                                token_id,
  output logic signed [D*BFP_MANT_W-1:0]            hidden_m,
  output logic signed [(D/BFP_TILE)*BFP_EXP_W-1:0]  hidden_e,
  output logic                                      done,
  // DDR3 lookup interface (used iff STREAM_LOOKUP=1).
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_EMBED_LU_m,
  input  wire [AXI_ADDR_WIDTH-1:0]                  ws_base_EMBED_LU_e,
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

  localparam int NT_D            = D / BFP_TILE;
  localparam int AXI_DATA_WIDTH  = 512;
  localparam int ROW_M_BYTES     = D * 2;                              // 1152 for D=576
  localparam int BEATS_M         = (ROW_M_BYTES + (AXI_DATA_WIDTH/8) - 1) / (AXI_DATA_WIDTH/8);
  localparam int MANTS_PER_BEAT  = (AXI_DATA_WIDTH/8) / 2;             // 32 for 512-bit data

  // Output buffers (registered) — interface unchanged from the BRAM path.
  logic signed [BFP_MANT_W-1:0] m_buf [0:D-1];
  logic signed [BFP_EXP_W -1:0] e_buf [0:NT_D-1];

  always_comb begin
    for (int i = 0; i < D; i++)
      hidden_m[i*BFP_MANT_W +: BFP_MANT_W] = m_buf[i];
    for (int t = 0; t < NT_D; t++)
      hidden_e[t*BFP_EXP_W  +: BFP_EXP_W ] = e_buf[t];
  end

  // ==========================================================================
  // STREAM_LOOKUP=0 — original BRAM + $readmemh path.
  // ==========================================================================
  generate if (!STREAM_LOOKUP) begin : g_bram

    (* ram_style = "block" *) logic signed [BFP_MANT_W-1:0] rom_m [0:VOCAB*D-1];
    (* ram_style = "block" *) logic signed [BFP_EXP_W -1:0] rom_e [0:VOCAB*NT_D-1];

`ifdef MICROGPT_WEIGHT_DIR
    initial begin
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "EMBED_LOOKUP_m.hex"}, rom_m);
      $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "EMBED_LOOKUP_e.hex"}, rom_e);
    end
`else
    initial begin
      $readmemh({PREFIX, "EMBED_LOOKUP_m.hex"}, rom_m);
      $readmemh({PREFIX, "EMBED_LOOKUP_e.hex"}, rom_e);
    end
`endif

    typedef enum logic [2:0] { B_IDLE, B_READ_M, B_READ_E, B_DONE } b_st_t;
    b_st_t bstate;
    logic [$clog2(D)-1:0]    bcnt_m;
    logic [$clog2(NT_D)-1:0] bcnt_e;
    logic [15:0]             btok;

    always_ff @(posedge clk) begin
      if (rst) begin
        bstate <= B_IDLE;
        bcnt_m <= '0; bcnt_e <= '0;
        btok   <= '0;
        done   <= 1'b0;
      end else begin
        done <= 1'b0;
        case (bstate)
          B_IDLE: if (start) begin
            btok   <= token_id;
            bcnt_m <= '0; bcnt_e <= '0;
            bstate <= B_READ_M;
          end
          B_READ_M: begin
            m_buf[bcnt_m] <= rom_m[btok * D + bcnt_m];
            if (bcnt_m == D-1) begin bcnt_m <= '0; bstate <= B_READ_E; end
            else                     bcnt_m <= bcnt_m + 1'b1;
          end
          B_READ_E: begin
            e_buf[bcnt_e] <= rom_e[btok * NT_D + bcnt_e];
            if (bcnt_e == NT_D-1) begin bcnt_e <= '0; bstate <= B_DONE; end
            else                       bcnt_e <= bcnt_e + 1'b1;
          end
          B_DONE: begin done <= 1'b1; bstate <= B_IDLE; end
          default: bstate <= B_IDLE;
        endcase
      end
    end

    // AXI ports tied off.
    assign m_axi_arvalid = 1'b0;
    assign m_axi_arid    = '0;
    assign m_axi_araddr  = '0;
    assign m_axi_arlen   = '0;
    assign m_axi_arsize  = 3'd6;
    assign m_axi_arburst = 2'b01;
    assign m_axi_arlock  = 1'b0;
    assign m_axi_arcache = 4'b0011;
    assign m_axi_arprot  = 3'b000;
    assign m_axi_arqos   = 4'b0000;
    assign m_axi_rready  = 1'b1;
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused_axi = &{1'b0, m_axi_arready, m_axi_rvalid, m_axi_rid,
                         m_axi_rdata, m_axi_rresp, m_axi_rlast, clk_axi,
                         rst_axi, ws_base_EMBED_LU_m, ws_base_EMBED_LU_e, 1'b0};
    /* verilator lint_on UNUSEDSIGNAL */

  // ==========================================================================
  // STREAM_LOOKUP=1 — DDR3 AXI fetch path.
  // ==========================================================================
  end else begin : g_stream

    // ----------------------------------------------------------------
    // Constant AR signals.
    // ----------------------------------------------------------------
    assign m_axi_arid    = '0;
    assign m_axi_arsize  = 3'd6;
    assign m_axi_arburst = 2'b01;
    assign m_axi_arlock  = 1'b0;
    assign m_axi_arcache = 4'b0011;
    assign m_axi_arprot  = 3'b000;
    assign m_axi_arqos   = 4'b0000;
    assign m_axi_rready  = 1'b1;

    // ----------------------------------------------------------------
    // Core → AXI toggle handshake (clk_core domain).
    // ----------------------------------------------------------------
    logic        core_tog;
    logic [15:0] core_tok;
    always_ff @(posedge clk) begin
      if (rst) begin
        core_tog <= 1'b0;
        core_tok <= 16'd0;
      end else if (start) begin
        core_tog <= ~core_tog;
        core_tok <= token_id;
      end
    end

    // ----------------------------------------------------------------
    // AXI-side: 2FF-sync core_tog and capture token.
    // ----------------------------------------------------------------
    logic [1:0]  tog_sync_axi;
    logic        tog_seen_axi;
    logic        start_axi;
    logic [15:0] axi_tok;
    always_ff @(posedge clk_axi) begin
      if (rst_axi) begin
        tog_sync_axi <= '0; tog_seen_axi <= 1'b0;
        start_axi    <= 1'b0;
        axi_tok      <= 16'd0;
      end else begin
        tog_sync_axi <= {tog_sync_axi[0], core_tog};
        start_axi    <= 1'b0;
        if (tog_sync_axi[1] != tog_seen_axi) begin
          tog_seen_axi <= tog_sync_axi[1];
          start_axi    <= 1'b1;
          axi_tok      <= core_tok;
        end
      end
    end

    // ----------------------------------------------------------------
    // Loader FSM (AXI side).  Stores BEATS_M mantissa beats + 1 exp
    // beat into FF banks.  Each beat is 512 b; total ≈ 9.7 Kbit — fits
    // ~10 RAMB18s if synthesized as BRAM, but at one-row-per-token
    // throughput a registered bank is fine and clearer.
    // ----------------------------------------------------------------
    typedef enum logic [3:0] {
      L_IDLE, L_M_AR, L_M_R, L_E_AR, L_E_R, L_DONE
    } l_st_t;
    l_st_t                       lstate;
    logic [AXI_ADDR_WIDTH-1:0]   addr_m, addr_e;
    logic [$clog2(BEATS_M+1)-1:0] mbeat_idx;
    logic                        done_axi;
    logic [AXI_DATA_WIDTH-1:0]   beat_buf [0:BEATS_M-1];
    logic [AXI_DATA_WIDTH-1:0]   exp_beat;
    logic                        m_arvalid_r;
    logic [AXI_ADDR_WIDTH-1:0]   m_araddr_r;
    logic [7:0]                  m_arlen_r;

    assign m_axi_arvalid = m_arvalid_r;
    assign m_axi_araddr  = m_araddr_r;
    assign m_axi_arlen   = m_arlen_r;

    always_ff @(posedge clk_axi) begin
      if (rst_axi) begin
        lstate      <= L_IDLE;
        m_arvalid_r <= 1'b0;
        m_araddr_r  <= '0;
        m_arlen_r   <= '0;
        addr_m      <= '0; addr_e <= '0;
        mbeat_idx   <= '0;
        done_axi    <= 1'b0;
        exp_beat    <= '0;
      end else begin
        // done_axi is LATCHED in L_DONE and held until the next start_axi
        // arrives.  Clearing it in L_IDLE (as before) made it a 5 ns
        // pulse at 200 MHz ui_clk, which the 50 MHz core-clock 2FF
        // sync missed — autoregress would stall in S_EMB_WAIT forever.
        case (lstate)
          L_IDLE: begin
            if (start_axi) begin
              done_axi  <= 1'b0;
              addr_m    <= ws_base_EMBED_LU_m + AXI_ADDR_WIDTH'(axi_tok) * AXI_ADDR_WIDTH'(ROW_M_BYTES);
              addr_e    <= ws_base_EMBED_LU_e + AXI_ADDR_WIDTH'(axi_tok) * AXI_ADDR_WIDTH'(AXI_DATA_WIDTH/8);
              mbeat_idx <= '0;
              lstate    <= L_M_AR;
            end
          end
          L_M_AR: begin
            m_arvalid_r <= 1'b1;
            m_araddr_r  <= addr_m;
            m_arlen_r   <= 8'(BEATS_M - 1);
            if (m_arvalid_r && m_axi_arready) begin
              m_arvalid_r <= 1'b0;
              lstate      <= L_M_R;
            end
          end
          L_M_R: begin
            if (m_axi_rvalid) begin
              beat_buf[mbeat_idx[$clog2(BEATS_M)-1:0]] <= m_axi_rdata;
              mbeat_idx <= mbeat_idx + 1'b1;
              if (m_axi_rlast) lstate <= L_E_AR;
            end
          end
          L_E_AR: begin
            m_arvalid_r <= 1'b1;
            m_araddr_r  <= addr_e;
            m_arlen_r   <= 8'd0;
            if (m_arvalid_r && m_axi_arready) begin
              m_arvalid_r <= 1'b0;
              lstate      <= L_E_R;
            end
          end
          L_E_R: begin
            if (m_axi_rvalid) begin
              exp_beat <= m_axi_rdata;
              lstate   <= L_DONE;
            end
          end
          L_DONE: begin
            done_axi <= 1'b1;
            lstate   <= L_IDLE;
          end
          default: lstate <= L_IDLE;
        endcase
      end
    end

    // ----------------------------------------------------------------
    // CDC done back to core domain — pulse on rising edge of sync'd
    // done_axi.  After done arrives on the core, beat_buf / exp_beat
    // are stable (writes have settled and won't change until next
    // token_id load) so the combinational slicing below is safe.
    // FPGA: declare a set_max_delay constraint between the two clocks
    // on this path; Verilator: works as-is.
    // ----------------------------------------------------------------
    logic [2:0] done_sync_core;
    always_ff @(posedge clk) begin
      if (rst) begin
        done_sync_core <= '0;
        done           <= 1'b0;
      end else begin
        done_sync_core <= {done_sync_core[1:0], done_axi};
        done           <= done_sync_core[1] & ~done_sync_core[2];   // rising edge
      end
    end

    // ----------------------------------------------------------------
    // Combinational slice of beat_buf into m_buf, exp_beat into e_buf.
    // Driven by the core-side `m_buf` / `e_buf` ports which feed
    // hidden_m / hidden_e via the comb assigns at the top.  Synthesis
    // sees this as a wide mux across BEATS_M beats — for D=576 that's
    // a 18-to-1 mux per output bit, fine on Virtex-7.
    // ----------------------------------------------------------------
    always_comb begin
      for (int i = 0; i < D; i++) begin
        automatic int beat = i / MANTS_PER_BEAT;
        automatic int sub  = i % MANTS_PER_BEAT;
        m_buf[i] = $signed(beat_buf[beat][sub*16 +: 16]);
      end
      for (int t = 0; t < NT_D; t++)
        e_buf[t] = $signed(exp_beat[t*8 +: 8]);
    end

  end
  endgenerate

endmodule

`default_nettype wire
