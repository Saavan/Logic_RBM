`timescale 1ns / 10ps
//////////////////////////////////////////////////////////////////////////////////
// Company: LEED
// Engineer: Philip Canoza
// 
// Create Date: 1/09/2019 1:18:54 PM
// Design Name: vanilla
// Module Name: sigmoid_func
// Project Name: RBM_FPGA
// Target Devices: Xilinx Virtex Ultrascale
// Tool Versions: 
// Description: An implementation of the sigmoid function.  Wrapper for a 7 bit
//              LUT unsigned sigmoid, with implementations for soft and hard 
//				saturations. Note it is very much hard coded.
// 				This version will assume 10 bit numbers, fpoint at 5th bit.
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments: 
// 
//////////////////////////////////////////////////////////////////////////////////
module sigmoid_func # (
	//Number of precision bits for fixed point calculations
    parameter PRECISION_BITS = 10,
    parameter N_IN = 8,
    parameter P_IN = 5,
    parameter N_OUT = 16,
    parameter P_OUT = 15,
    parameter P_INDEX = 5
) (
	input[PRECISION_BITS - 1:0] x,
	output reg[N_OUT - 1:0] y
); 

	//Values for saturation
	localparam SOFTSAT_VAL = {{(N_OUT-P_OUT){1'b0}},{P_OUT{1'b1}}};	// 0.999512 in convention
	localparam LSB_VAL = {{(N_OUT-1){1'b0}}, 1'b1}; 		// 0.000488 in convention
	localparam HARDSAT_VAL = {{(N_OUT-P_OUT-1){1'b0}},1'b1,{P_OUT{1'b0}}};	// 1 in convention

    localparam HARDSAT_CUTOFF = {{(PRECISION_BITS - N_IN - (P_INDEX - P_IN) - 1){1'b0}}, {(N_IN + P_INDEX - P_IN){1'b1}}};
    localparam SOFTSAT_CUTOFF = HARDSAT_CUTOFF;

	//Toggle bit to select negative number
	wire isneg;
	assign isneg = x[PRECISION_BITS - 1];

	//Toggle bits for hard and soft saturation values
	reg[PRECISION_BITS-2:0] x_trunc;	//Unsigned x
	always @(*) begin
		x_trunc = (isneg) ? -x : x;		//Take the magnitude of x
	end
	wire softsat, hardsat;
	assign hardsat = x_trunc >= HARDSAT_CUTOFF;  // Hard cutoff at 8
	assign softsat = x_trunc >= SOFTSAT_CUTOFF;	// Soft saturaton starts at 8


	//LUT is unsigned, so use our truncated x
	
	wire[N_OUT -1 :0] outLUT;
	wire[N_IN - 1:0] inLUT; 
	assign inLUT = x_trunc[P_INDEX+(N_IN-P_IN)-1:P_INDEX-P_IN];


 	//Declaration of sigmoid LUT
    sigmoidLUT #(
        .N_IN(N_IN),
        .P_IN(P_IN),
        .N_OUT(N_OUT),
        .P_OUT(P_OUT)
    ) sigLUT (
        .in(inLUT), .out(outLUT)
    );

	//Reversion back from negative numbers
	reg[N_OUT - 1:0] nonsat_val;
	always @(*) begin
		case (isneg)
			1'b0: nonsat_val = outLUT;
			1'b1: nonsat_val = HARDSAT_VAL - outLUT;   //1 - x in our conv.
		endcase
	end

	//Choose the correct value
	always @(*) begin
		if (hardsat) begin
			y = (isneg) ? {N_OUT{1'b0}} : HARDSAT_VAL;
		end else if (softsat) begin
			y = (isneg) ? LSB_VAL : SOFTSAT_VAL;
		end else begin
			y = nonsat_val;
		end
	end

endmodule