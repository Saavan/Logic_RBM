`timescale 1ns / 10ps
`include "util.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company: LEED
// Engineer: Philip Canoza
// 
// Create Date: 2/11/2019 1:05:35 AM
// Design Name: vanilla
// Module Name: adder_tree
// Project Name: RBM_FPGA
// Target Devices: Xilinx Virtex Ultrascale
// Tool Versions: 
// Description: Inputing a vector of values to add, optimized addition via
//				binary adder tree.  Assume power of 2 number of nodes.
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments: 
// 
//////////////////////////////////////////////////////////////////////////////////
module adder_tree # (

	//Number of precision bits for fixed point calculations
  	parameter PRECISION_BITS = 32,
  	// Number of nodes
  	parameter NUM_NODES = 4,
  	// Machine clock frequency in Hz.
  	parameter RBM_CLOCK_FREQ = 50_000_000

)(
  	input clk,      //Machine clock
  	input [PRECISION_BITS*NUM_NODES - 1:0] addends , //fixed point values to add
  	output reg [PRECISION_BITS - 1:0] result //Addition output
);

//    localparam MID = NUM_NODES/2;
//    reg [PRECISION_BITS*NUM_NODES - 1:0] addends2;
//    integer i, j;
//    reg [PRECISION_BITS - 1:0] temp_data;
//    reg [PRECISION_BITS - 1:0] mid;
//    reg [PRECISION_BITS - 1:0] temp_data2;
//    always @* begin
//         temp_data = 0;
//        for (i = 0; i < MID; i = i + 1) 
//            temp_data = temp_data + addends[i*PRECISION_BITS +: PRECISION_BITS];
//        temp_data2 = mid;
//        for (j = i; j < NUM_NODES; j = j + 1)
//            temp_data2 = temp_data2 + addends2[j*PRECISION_BITS +: PRECISION_BITS];
//        result = temp_data2;
//    end
    
//    always @(posedge clk) begin
//        mid <= temp_data;
//        addends2 <= addends;
//    end


    integer i;
    reg [PRECISION_BITS - 1:0] temp_data;
    always @* begin
         temp_data = 0;
        for (i = 0; i < NUM_NODES; i = i + 1) 
            temp_data = temp_data + addends[i*PRECISION_BITS +: PRECISION_BITS];
        result = temp_data;
    end

    
endmodule


