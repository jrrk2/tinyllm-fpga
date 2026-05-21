// soc_ddr_bridge.sv — single-beat 32-bit <-> 512-bit AXI master with a
// clk_soc(eth_clk) <-> clk_axi(ui_clk) CDC, so the PicoSoC can use DDR3 as a
// large data memory.  One 32-bit word per request; the CPU stalls (the shell
// holds iomem_ready low) through the MIG latency — fine as a DATA window
// (not execute-in-place; PicoRV32 has no cache).
//
// CDC: the SoC holds the request stable and waits for `done`, so address/data
// are stable across the crossing and only a toggle needs synchronizing.

`default_nettype none

module soc_ddr_bridge (
  // ---- SoC side (clk_soc = eth_clk) ----
  input  wire        clk_soc,
  input  wire        rst_soc,
  input  wire        req,             // 1-cycle pulse: start an access
  input  wire [29:0] addr,            // byte address (32-bit aligned)
  input  wire [31:0] wdata,
  input  wire [3:0]  wstrb,           // 0 => read
  output reg         done,            // 1-cycle pulse on completion
  output reg  [31:0] rdata,

  // ---- AXI side (clk_axi = ui_clk) ----
  input  wire        clk_axi,
  input  wire        rst_axi,
  output reg  [4:0]  m_axi_arid,   output reg  [29:0] m_axi_araddr,
  output reg  [7:0]  m_axi_arlen,  output reg  [2:0]  m_axi_arsize,
  output reg  [1:0]  m_axi_arburst,output reg  [0:0]  m_axi_arlock,
  output reg  [3:0]  m_axi_arcache,output reg  [2:0]  m_axi_arprot,
  output reg  [3:0]  m_axi_arqos,  output reg         m_axi_arvalid,
  input  wire        m_axi_arready,
  output reg         m_axi_rready, input  wire [4:0]  m_axi_rid,
  input  wire [511:0] m_axi_rdata, input  wire [1:0]  m_axi_rresp,
  input  wire        m_axi_rlast,  input  wire        m_axi_rvalid,
  output reg  [4:0]  m_axi_awid,   output reg  [29:0] m_axi_awaddr,
  output reg  [7:0]  m_axi_awlen,  output reg  [2:0]  m_axi_awsize,
  output reg  [1:0]  m_axi_awburst,output reg  [0:0]  m_axi_awlock,
  output reg  [3:0]  m_axi_awcache,output reg  [2:0]  m_axi_awprot,
  output reg  [3:0]  m_axi_awqos,  output reg         m_axi_awvalid,
  input  wire        m_axi_awready,
  output reg  [511:0] m_axi_wdata, output reg  [63:0] m_axi_wstrb,
  output reg         m_axi_wlast,  output reg         m_axi_wvalid,
  input  wire        m_axi_wready,
  output reg         m_axi_bready, input  wire [4:0]  m_axi_bid,
  input  wire [1:0]  m_axi_bresp,  input  wire        m_axi_bvalid
);

  // ---- SoC-domain request capture + CDC toggle ----
  reg        req_tog;                 // toggles on each request
  reg [29:0] q_addr;  reg [31:0] q_wdata;  reg [3:0] q_wstrb;
  always_ff @(posedge clk_soc) begin
    if (rst_soc) begin req_tog <= 1'b0; q_addr <= '0; q_wdata <= '0; q_wstrb <= '0; end
    else if (req) begin
      req_tog <= ~req_tog; q_addr <= addr; q_wdata <= wdata; q_wstrb <= wstrb;
    end
  end

  // ---- AXI-domain: 2FF-sync req toggle, run one beat ----
  reg [2:0]  req_sync;   // {seen, ff1, ff0}
  wire       req_edge = (req_sync[2] ^ req_sync[1]);
  reg [511:0] rd_cap;
  reg        ack_tog;    // toggles when a transaction completes (AXI domain)

  typedef enum logic [2:0] { A_IDLE, A_AR, A_R, A_AW, A_W, A_B } ast_t;
  ast_t ast;
  wire [3:0] lane = q_addr[5:2];   // which 32-bit word in the 512-bit beat

  always_ff @(posedge clk_axi) begin
    if (rst_axi) begin
      req_sync <= 3'b0; ast <= A_IDLE; ack_tog <= 1'b0; rd_cap <= '0;
      m_axi_arid<=0; m_axi_araddr<=0; m_axi_arlen<=0; m_axi_arsize<=3'd6;
      m_axi_arburst<=2'b01; m_axi_arlock<=0; m_axi_arcache<=4'h3; m_axi_arprot<=0;
      m_axi_arqos<=0; m_axi_arvalid<=0; m_axi_rready<=0;
      m_axi_awid<=0; m_axi_awaddr<=0; m_axi_awlen<=0; m_axi_awsize<=3'd6;
      m_axi_awburst<=2'b01; m_axi_awlock<=0; m_axi_awcache<=4'h3; m_axi_awprot<=0;
      m_axi_awqos<=0; m_axi_awvalid<=0;
      m_axi_wdata<=0; m_axi_wstrb<=0; m_axi_wlast<=0; m_axi_wvalid<=0; m_axi_bready<=0;
    end else begin
      req_sync <= {req_sync[1:0], req_tog};
      case (ast)
        A_IDLE: if (req_edge) begin
          if (|q_wstrb) begin
            m_axi_awaddr  <= {q_addr[29:6], 6'b0};
            m_axi_awvalid <= 1'b1;
            m_axi_wdata   <= {16{q_wdata}};                 // replicate to all lanes
            m_axi_wstrb   <= ({60'b0, q_wstrb} << (lane*4)); // enable only this word
            m_axi_wlast   <= 1'b1;
            m_axi_wvalid   <= 1'b1;
            ast <= A_AW;
          end else begin
            m_axi_araddr  <= {q_addr[29:6], 6'b0};
            m_axi_arvalid <= 1'b1;
            ast <= A_AR;
          end
        end
        A_AR: if (m_axi_arready) begin m_axi_arvalid<=0; m_axi_rready<=1; ast<=A_R; end
        A_R:  if (m_axi_rvalid)  begin rd_cap<=m_axi_rdata; m_axi_rready<=0; ack_tog<=~ack_tog; ast<=A_IDLE; end
        A_AW: begin
          if (m_axi_awready) m_axi_awvalid<=0;
          if (m_axi_wready)  m_axi_wvalid <=0;
          if ((m_axi_awready||!m_axi_awvalid) && (m_axi_wready||!m_axi_wvalid)) begin
            m_axi_bready<=1; ast<=A_B;
          end
        end
        A_B:  if (m_axi_bvalid) begin m_axi_bready<=0; ack_tog<=~ack_tog; ast<=A_IDLE; end
        default: ast <= A_IDLE;
      endcase
    end
  end

  // ---- SoC-domain: 2FF-sync ack toggle -> done pulse + capture word ----
  reg [2:0] ack_sync;
  always_ff @(posedge clk_soc) begin
    if (rst_soc) begin ack_sync <= 3'b0; done <= 1'b0; rdata <= '0; end
    else begin
      ack_sync <= {ack_sync[1:0], ack_tog};
      done <= (ack_sync[2] ^ ack_sync[1]);
      if (ack_sync[2] ^ ack_sync[1])
        rdata <= rd_cap[ q_addr[5:2]*32 +: 32 ];   // rd_cap stable by now
    end
  end

endmodule

`default_nettype wire
