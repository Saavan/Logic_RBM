`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: LEED
// Engineer: Philip Canoza
//
// Create Date: 5/06/2019 5:36:30 AM
// Design Name: vanilla
// Module Name: lfs_lookahead
// Project Name: RBM_FPGA
// Target Devices: Xilinx Virtex Ultrascale
// Tool Versions:
// Description: Implementation of LFSR pseudorandom number generator. Generates
//				n random bits with each cycle by predicting n clock cycles of LFSR generation
// Dependencies:
//
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
//
//////////////////////////////////////////////////////////////////////////////////
module lfsr_lookahead # (
	//Length of LFSR
	parameter LENGTH = 16,
	//Number of bits out each cycle
	parameter N = 1,
	parameter [LENGTH-1:0] TAPS = 16'b1011010000000000,
	parameter [LENGTH-1:0] SEED = 16'hcafe
) (
	input clk, rst,
	output[N-1:0] out
);

	reg[LENGTH-1:0] s_reg;
	initial s_reg = SEED;

	
	
	reg[N-1:0] next;
	wire [LENGTH + N-1:0] long_taps = {TAPS, {N{1'b0}}}; 
	integer j;
	//Dynamically generating the next bits
	always @(*) begin
        for(j = 0; j < N; j = j+1) begin
            next[N-j-1] = ~^({s_reg, next} & (long_taps>>j));
        end	   
	end
	
	//Note the nonblocking assignments so the feedback is correct
	always @(posedge clk) begin
		if (rst)
			s_reg <= SEED;
		else begin
		    //Move over by N bits instead of 1
			s_reg[N +: LENGTH - N] <= s_reg[0 +: LENGTH-N];
			s_reg[0 +: N] <= next;
		end
	end
    //The MSBs are set to the output
	assign out = s_reg[LENGTH-1 -: N];

endmodule
