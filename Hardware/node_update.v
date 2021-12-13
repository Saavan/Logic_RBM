//////////////////////////////////////////////////////////////////////////////////
// Company: LEED
// Engineer: Philip Canoza
// 
// Create Date: 1/09/2019 1:12:20 PM
// Design Name: vanilla
// Module Name: node_update
// Project Name: RBM_FPGA
// Target Devices: Xilinx Virtex Ultrascale
// Tool Versions: 
// Description: Updates node values given previous node, weights, and biases
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments: 
// 
//////////////////////////////////////////////////////////////////////////////////
module node_update # (

  	//Number of precision bits for fixed point calculations
  	parameter PRECISION_BITS = 10,
  	// Number of bits below fixed point
  	parameter P_INDEX = 5,
  	//Extra bits to prevent overflow in additions
  	parameter OVERFLOW_BITS = 8,
  	// Number of visible nodes
  	parameter NUM_NODES = 3,
  	// Machine clock frequency in Hz.
  	parameter RBM_CLOCK_FREQ = 50_000_000,
  	//Decimal and point location for sigmoid input
  	parameter N_IN = 7,
  	parameter P_IN = 4,
  	//Decimal and point location for sigmoid output
	parameter N_OUT = 16,
	parameter P_OUT = 15,
    //How long should the LFSR random number generator be
    parameter RNG_LEN = 32

) (
  	input clk,      //Machine clock
  	input[PRECISION_BITS - 1:0] bias,
  	input[NUM_NODES - 1:0] nodes,   //Previous node values
	input[PRECISION_BITS*NUM_NODES - 1:0] weights,  //Fixed point weights
	input[P_OUT -1 :0] rand, // Input random number
	output reg [PRECISION_BITS + OVERFLOW_BITS - 1:0] sigmoid_input, //For use with hitting time engine
	output new_val 	//Binary result, new node value
);


	//Perform vector multiplication of weight row and input nodes
	wire [PRECISION_BITS + OVERFLOW_BITS - 1:0] vecmul_out;
	vecmul  #(
		.PRECISION_BITS(PRECISION_BITS),
		.OVERFLOW_BITS(OVERFLOW_BITS),
		.NUM_NODES(NUM_NODES),
		.RBM_CLOCK_FREQ(RBM_CLOCK_FREQ)
	) vec_multiplier (
		.clk(clk), 
  		.nodes(nodes),
		.weights(weights),
 		.product(vecmul_out) 
	);

	//Add the bias value h
	//reg [PRECISION_BITS + OVERFLOW_BITS - 1:0] sigmoid_input;
	
	//Doing this ensures nothing overflows, as we keep everything in the higher bit count domain
	wire [PRECISION_BITS + OVERFLOW_BITS - 1:0] bias_extended = $signed(bias);
	always @* begin
	   sigmoid_input = vecmul_out + bias_extended;
	end

	//Pass through non linear squishing function
	//sigmoid function now operates with more bits
	wire [N_OUT - 1:0] sigmoid_output;
	sigmoid_func #(
	.N_IN(N_IN),
	.P_IN(P_IN),
	.N_OUT(N_OUT),
	.P_OUT(P_OUT),
	.P_INDEX(P_INDEX),
	.PRECISION_BITS(PRECISION_BITS+OVERFLOW_BITS)
	) nonlinear_machine (
		.x(sigmoid_input),
		.y(sigmoid_output)
	);

    //Comparing against a random number
	wire [N_OUT - 1:0] random_num;
	assign random_num = { {(N_OUT-P_OUT){1'b0}}, rand};
	assign new_val = sigmoid_output > random_num;

endmodule

