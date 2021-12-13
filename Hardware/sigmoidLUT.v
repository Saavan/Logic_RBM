`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 07/08/2019 01:11:55 PM
// Design Name: 
// Module Name: sigmoidLUT
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: This performs all back end modeling of sigmoid sizing so you don't have to modify
//                it internally everytime precision is changed
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module sigmoidLUT # (
	//Number of precision bits for fixed point calculations
    parameter N_IN = 8,
    parameter P_IN = 4,
    parameter N_OUT = 32,
    parameter P_OUT = 31
) (
    input[N_IN-1:0] in,
    output[N_OUT-1:0] out
    );
    
    if (N_IN == 8 & P_IN == 4) begin
        if (N_OUT == 16 & P_OUT == 15) begin
           sigmoidLUT_in8b4p_out16b15p sigLUT (
		      .sigmoid_in(in), .sigmoid_out(out)
	       );
        end else if (N_OUT == 32 & P_OUT == 31) begin
	       sigmoidLUT_in8b4p_out32b31p sigLUT (
		      .sigmoid_in(in), .sigmoid_out(out)
	       );
        end else if (N_OUT == 64 & P_OUT == 63) begin
           sigmoidLUT_in8b4p_out64b63p sigLUT (
		      .sigmoid_in(in), .sigmoid_out(out)
	       );
        end else begin
            assign out = in;
            initial 
                $display("WARNING:NOT USING VALID COMBINATION FOR SIGMOID LUT");
        end
    end else if (N_IN == 7 & P_IN==4) begin
        if (N_OUT == 8 & P_OUT == 7) begin 
           sigmoidLUT_in7b4p_out8b7p sigLUT (
		      .sigmoid_in(in), .sigmoid_out(out)
	       );
       end else if (N_OUT == 16 & P_OUT == 15) begin
           sigmoidLUT_in7b4p_out16b15p sigLUT (
		      .sigmoid_in(in), .sigmoid_out(out)
	       );
        end else begin
            assign out = in;
            initial 
                $display("WARNING:NOT USING VALID COMBINATION FOR SIGMOID LUT");
        end
    end else if (N_IN == 7 & P_IN==5) begin
        if (N_OUT == 16 & P_OUT == 15) begin 
           sigmoidLUT_in7b5p_out16b15p sigLUT (
		      .sigmoid_in(in), .sigmoid_out(out)
	       );
        end else begin
            assign out = in;
            initial 
                $display("WARNING:NOT USING VALID COMBINATION FOR SIGMOID LUT");
        end
    end else if (N_IN == 6 & P_IN==4) begin
        if (N_OUT == 16 & P_OUT == 15) begin 
           sigmoidLUT_in6b4p_out16b15p sigLUT (
		      .sigmoid_in(in), .sigmoid_out(out)
	       );
        end else begin
            assign out = in;
            initial 
                $display("WARNING:NOT USING VALID COMBINATION FOR SIGMOID LUT");
        end
    end else begin
        assign out = in;
        initial begin
            $display("WARNING:NOT USING VALID COMBINATION FOR SIGMOID LUT");
        end
    end
    
endmodule
