`timescale 1ns / 10ps
`include "util.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company: LEED
// Engineer: Philip Canoza
//
// Create Date: 1/09/2019 1:22:35 PM
// Design Name: vanilla
// Module Name: vecmul
// Project Name: RBM_FPGA
// Target Devices: Xilinx Virtex Ultrascale
// Tool Versions:
// Description: An implementation of "vector multiplication" by masking values
//		          then performing addition on the survivors
// Dependencies:
//
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
//
//////////////////////////////////////////////////////////////////////////////////
module vecmul # (

  //Number of precision bits for fixed point calculations
  parameter PRECISION_BITS = 32,
  // Number of nodes
  parameter NUM_NODES = 4,
  // Machine clock frequency in Hz.
  parameter RBM_CLOCK_FREQ = 50_000_000,
  // Extra overflow bits
  parameter OVERFLOW_BITS = 8

) (
  input clk,      //Machine clock
  input[NUM_NODES - 1:0] nodes,   //Binary node values
  input [PRECISION_BITS*NUM_NODES - 1:0] weights, //fixed point weight values

  output reg [(PRECISION_BITS + OVERFLOW_BITS) - 1:0] product //Vector multiplication output
);
//  localparam PAD_NODES = `pow2 (`log2(NUM_NODES));
  localparam PAD_BITS = PRECISION_BITS + OVERFLOW_BITS;
//  localparam EXTRA_BITS = (PAD_NODES - NUM_NODES)*PAD_BITS;

  localparam MAX_POS = {1'b0, {(PRECISION_BITS-1){1'b1}}};
  localparam MAX_NEG = {1'b1, {(PRECISION_BITS-2){1'b0}}, 1'b1};

  wire [PAD_BITS*NUM_NODES - 1:0] masked_weights;

  //Use binary mask on the weight vector input
  // Lot's of ugly indexing since 2D arrays can't be input/output for modules
  genvar i;
  generate
    // Mask and pad weights
    for (i = 0; i < NUM_NODES; i = i + 1) begin
          wire [PRECISION_BITS - 1:0] weight = weights[PRECISION_BITS*i +: PRECISION_BITS];
          wire [PAD_BITS - 1:0] pad_weight = $signed(weight);
          assign masked_weights[PAD_BITS*(i + 1) - 1: PAD_BITS*i] = (nodes[i]) ? pad_weight : {PAD_BITS{1'b0}};  //Choose either padded weight or 0
    end
  endgenerate
  //Zero pad the excess nodes
//  if (PAD_NODES > NUM_NODES)
//    assign masked_weights[PAD_BITS*PAD_NODES - 1:PAD_BITS*NUM_NODES] = {EXTRA_BITS {1'b0}};

  //Instantiate adder implementation for efficient computation
  //Forcing proper truncation for signed numbers
  wire [PAD_BITS - 1:0] pad_out;
  adder_tree  #(
    //Number of precision bits for fixed point calculations
    .PRECISION_BITS(PAD_BITS),
    // Number of nodes
    .NUM_NODES(NUM_NODES),
    // Machine clock frequency in Hz.
    .RBM_CLOCK_FREQ(RBM_CLOCK_FREQ)
  ) tree (
    .clk(clk),
    .addends(masked_weights),
    .result(pad_out)
  );
  
  //Registering the output of the adder tree (the tree has a lot of latency)

  reg [PAD_BITS - 1:0] pad_reg;
  always @(posedge clk) begin
    pad_reg <= pad_out;
  end
  
  always @* begin
    product = pad_reg;
  end
//    always @* begin
//        product = pad_out;
//    end


endmodule
