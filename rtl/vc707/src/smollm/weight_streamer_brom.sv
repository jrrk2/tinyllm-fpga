// weight_streamer_brom.sv — sim-only $readmemh-backed weight source.
//
// Drop-in mock for the FPGA-side weight_streamer.sv: same client-facing
// interface (matvec_id / chunk_idx / load_req / ready / rd_addr / weight_data),
// but loads weights from per-layer hex files into 7 internal BRAMs at
// elaboration time.  Lets us validate the smollm_layer integration in
// simulation before swapping in the AXI-DDR3 path.
//
// Each smollm_layer instance contains one of these.  PREFIX selects the
// per-layer hex file naming (e.g. "full_L0_") consistently with
// smollm_layer's own GAMMA / KV-cache hex prefix.

`default_nettype none

module weight_streamer_brom #(
  parameter int D       = 128,
  parameter int H_KV    = 1,
  parameter int HD      = 64,
  parameter int FFN     = 128,
  parameter        PREFIX = "layer_"
)(
  input  wire                      clk,
  input  wire                      rst,

  // Client request — registered into a "current matvec" context.
  input  wire [2:0]                matvec_id,    // 0=Q 1=K 2=V 3=O 4=GATE 5=UP 6=DOWN
  input  wire [6:0]                chunk_idx,
  input  wire                      load_req,
  output logic                     ready,

  // Combinational consumer port (matches old direct-ROM access semantics).
  input  wire [10:0]               rd_addr,
  output logic [127:0]              weight_data
);

  // Per-matrix dimension constants (in_dim = D for Q/K/V/O/GATE/UP, FFN for DOWN)
  localparam int CHUNKS_Q    = D       / 16;
  localparam int CHUNKS_KV   = (H_KV*HD) / 16;
  localparam int CHUNKS_FFN  = FFN     / 16;

  // ------------------------------------------------------------------
  //  7 weight ROMs (mirrors the layout that used to live in smollm_layer)
  // ------------------------------------------------------------------
  logic [127:0] rom_W_Q     [0:CHUNKS_Q   * D   - 1];
  logic [127:0] rom_W_K     [0:CHUNKS_KV  * D   - 1];
  logic [127:0] rom_W_V     [0:CHUNKS_KV  * D   - 1];
  logic [127:0] rom_W_O     [0:CHUNKS_Q   * D   - 1];
  logic [127:0] rom_W_GATE  [0:CHUNKS_FFN * D   - 1];
  logic [127:0] rom_W_UP    [0:CHUNKS_FFN * D   - 1];
  logic [127:0] rom_W_DOWN  [0:CHUNKS_Q   * FFN - 1];

`ifdef MICROGPT_WEIGHT_DIR
  initial begin
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "W_Q.hex"},    rom_W_Q);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "W_K.hex"},    rom_W_K);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "W_V.hex"},    rom_W_V);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "W_O.hex"},    rom_W_O);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "W_GATE.hex"}, rom_W_GATE);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "W_UP.hex"},   rom_W_UP);
    $readmemh({`MICROGPT_WEIGHT_DIR, "/", PREFIX, "W_DOWN.hex"}, rom_W_DOWN);
  end
`else
  initial begin
    $readmemh({PREFIX, "W_Q.hex"},    rom_W_Q);
    $readmemh({PREFIX, "W_K.hex"},    rom_W_K);
    $readmemh({PREFIX, "W_V.hex"},    rom_W_V);
    $readmemh({PREFIX, "W_O.hex"},    rom_W_O);
    $readmemh({PREFIX, "W_GATE.hex"}, rom_W_GATE);
    $readmemh({PREFIX, "W_UP.hex"},   rom_W_UP);
    $readmemh({PREFIX, "W_DOWN.hex"}, rom_W_DOWN);
  end
`endif

  // ------------------------------------------------------------------
  //  Loader: latches matvec_id + chunk_idx on load_req; drops ready while
  //  pending, then re-asserts ready 1 cycle later.  This 1-cycle handshake
  //  mirrors the AXI-streamer behaviour so the consumer FSM is identical.
  // ------------------------------------------------------------------
  logic [2:0] cur_matvec;
  logic [6:0] cur_chunk;
  logic       load_pending;

  always_ff @(posedge clk) begin
    if (rst) begin
      ready        <= 1'b0;
      cur_matvec   <= '0;
      cur_chunk    <= '0;
      load_pending <= 1'b0;
    end else begin
      if (load_req) begin
        cur_matvec   <= matvec_id;
        cur_chunk    <= chunk_idx;
        ready        <= 1'b0;
        load_pending <= 1'b1;
      end else if (load_pending) begin
        load_pending <= 1'b0;
        ready        <= 1'b1;
      end
    end
  end

  // ------------------------------------------------------------------
  //  Combinational read mux — index the appropriate ROM at
  //    cur_chunk × in_dim_for_that_matrix + rd_addr
  // ------------------------------------------------------------------
  always_comb begin
    case (cur_matvec)
      3'd0: weight_data = rom_W_Q   [cur_chunk * D   + rd_addr];
      3'd1: weight_data = rom_W_K   [cur_chunk * D   + rd_addr];
      3'd2: weight_data = rom_W_V   [cur_chunk * D   + rd_addr];
      3'd3: weight_data = rom_W_O   [cur_chunk * D   + rd_addr];
      3'd4: weight_data = rom_W_GATE[cur_chunk * D   + rd_addr];
      3'd5: weight_data = rom_W_UP  [cur_chunk * D   + rd_addr];
      3'd6: weight_data = rom_W_DOWN[cur_chunk * FFN + rd_addr];
      default: weight_data = '0;
    endcase
  end

endmodule

`default_nettype wire
