`include "rng.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company: LEED
// Engineer: Philip Canoza
// 
// Create Date: 1/09/2019 1:08:43 PM
// Design Name: vanilla
// Module Name: RBM
// Project Name: RBM_FPGA
// Target Devices: Xilinx VCU118
// Tool Versions: 
// Description: Boltzmann machine that houses the main memory hierarchy
//				and computation modules. Instantiated by upper level for a specific
//              RBM instance (AND, SAT, etc.)
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments: 
// 
//////////////////////////////////////////////////////////////////////////////////
module RBM # (

  	// Machine clock frequency in Hz.
  	parameter RBM_CLOCK_FREQ = 50_000_000,
  	//Number of precision bits for fixed point calculations
  	parameter PRECISION_BITS = 10,
  	// Number of extra bits during calculation to prevent overflow
  	parameter OVERFLOW_BITS = 2,
    // Number of bits below fixed point
  	parameter P_INDEX = 5,
  	// Number of visible nodes
  	parameter NUM_VNODES = 3,
  	// Number of hidden nodes 
  	parameter NUM_HNODES = 3,
  	//Decimal and point location for sigmoid input
  	parameter N_IN = 8,
  	parameter P_IN = 4,
    //Decimal and point location for sigmoid output
    parameter N_OUT = 16,
    parameter P_OUT = N_OUT - 1,
    //How long should the LFSR random number generator be
    parameter RNG_LEN = 32

) (
	input reset,		//Machine reset
	input clk,			//Machine clock
	input stall,        //Stall data generation
	//input weights, assumes rows of visible nodes. 
	//Array should be NUM_HNODES columns and NUM_VNODES rows
	input[PRECISION_BITS*NUM_HNODES*NUM_VNODES - 1:0] weights, 
	//input[PRECISION_BITS*NUM_HNODES*NUM_VNODES - 1:0] weights_transpose, 
	input[PRECISION_BITS*NUM_VNODES - 1:0] visible_bias,
	input[PRECISION_BITS*NUM_HNODES - 1:0] hidden_bias, 
	//Used to clamp output to either 0 or 1
	input[NUM_VNODES - 1:0] clamp0,
	input[NUM_VNODES - 1:0] clamp1,
	output reg[NUM_VNODES - 1:0]  node_data,	//Output data to computer
    output [(PRECISION_BITS+OVERFLOW_BITS)*NUM_HNODES - 1:0] hidden_calc //Hidden calculations for hitting time engine
);

	//////////////////////////////////////////////////////////////////////////////////
	//Node update
	//////////////////////////////////////////////////////////////////////////////////

    reg update_visA;

    
    
    
    //Creating two crossed chains to (explicitly) utilize full RBM compute architecture
    reg[NUM_VNODES - 1:0] visA;
    reg[NUM_VNODES - 1:0] visB;
    wire[NUM_VNODES - 1:0] vis_out;
    reg[NUM_HNODES - 1:0] hidA;
    reg[NUM_HNODES - 1:0] hidB;
    wire[NUM_HNODES - 1:0] hid_out;
    
    //Creating transposed weights matrix
    wire[PRECISION_BITS*NUM_HNODES*NUM_VNODES - 1:0] weight_trans;
//    assign weight_trans = weights_transpose; 
    genvar l, m;
    for (l = 0; l < NUM_HNODES; l=l+1)
    begin : trans_gen1
        for (m = 0; m < NUM_VNODES; m=m+1)
        begin :trans_gen2
             assign weight_trans[PRECISION_BITS*((l*NUM_VNODES)+m) +: PRECISION_BITS] = weights[PRECISION_BITS*((m*NUM_HNODES)+l) +: PRECISION_BITS];
        end 
    end

    //Generating the visible nodes
    genvar j;
    for (j = 0; j < NUM_VNODES; j = j+1)
    begin : vis_nodes
        wire[P_OUT -1:0] rand;
    	lfsr_lookahead #(
    	    .N(P_OUT),
			.LENGTH(RNG_LEN),
			.TAPS(`RNG_TAPS(RNG_LEN)),
			.SEED(j+1)
        ) visrng (
            .clk(clk), .rst(reset),
            .out(rand)
        );
        node_update #(
            .PRECISION_BITS(PRECISION_BITS),
            .OVERFLOW_BITS(OVERFLOW_BITS),
            .P_INDEX(P_INDEX),
            //Note this is the number of input nodes
            .NUM_NODES(NUM_HNODES), 
            .RBM_CLOCK_FREQ(RBM_CLOCK_FREQ),
            .N_IN(N_IN),
            .P_IN(P_IN),
            .N_OUT(N_OUT),
            .P_OUT(P_OUT)
            ) fire_vis (
            .clk(clk),      //Machine clock
            .bias(visible_bias[PRECISION_BITS*j +: PRECISION_BITS]),
            .nodes(update_visA ? hidA : hidB),   //Previous node values
            .weights(weights[PRECISION_BITS*NUM_HNODES*j +: NUM_HNODES*PRECISION_BITS]),  //Floating point weights
            .rand(rand), //Random number generator output
            .new_val(vis_out[j]), 	//Binary result
            .sigmoid_input()
        );
	end
	wire [NUM_VNODES - 1:0] hid_in; 
	assign hid_in = update_visA ? visB : visA;
	//Generating the hidden nodes
    genvar k;
    for (k = 0; k < NUM_HNODES; k = k+1)
    begin : hid_nodes
        wire[P_OUT -1:0] rand;
    	lfsr_lookahead #(
    	    .N(P_OUT),
			.LENGTH(RNG_LEN),
			.TAPS(`RNG_TAPS(RNG_LEN)),
			.SEED(k+NUM_VNODES+1)
        ) visrng (
            .clk(clk), .rst(reset),
            .out(rand)
        );
        node_update #(
            .PRECISION_BITS(PRECISION_BITS),
            .OVERFLOW_BITS(OVERFLOW_BITS),
            .P_INDEX(P_INDEX),
            //Note this is the number of input nodes
            .NUM_NODES(NUM_VNODES), 
            .RBM_CLOCK_FREQ(RBM_CLOCK_FREQ),
            .N_IN(N_IN),
            .P_IN(P_IN),
            .N_OUT(N_OUT),
            .P_OUT(P_OUT)
            ) fire_hid (
            .clk(clk),      //Machine clock
            .bias(hidden_bias[PRECISION_BITS*k +: PRECISION_BITS]),
            .nodes(hid_in),   //Previous node values
            .weights(weight_trans[PRECISION_BITS*NUM_VNODES*k +: NUM_VNODES*PRECISION_BITS]),  //Floating point weights
            .rand(rand), //Random number generator output
            .new_val(hid_out[k]), 	//Binary result
            .sigmoid_input(hidden_calc[(PRECISION_BITS+OVERFLOW_BITS)*k +: (PRECISION_BITS+OVERFLOW_BITS)]) //output for hitting engine
        );
	end
	wire [NUM_VNODES - 1 :0] vis_update;
	assign vis_update = (vis_out | clamp1) & (~(clamp0));
	//Logic for node updates
    always @ (posedge clk) begin
        if (reset) begin
            update_visA <= 1'b0;
            visA <= {NUM_VNODES{1'b0}};
            hidA <= {NUM_HNODES{1'b0}};
            visB <= {NUM_VNODES{1'b0}};
            hidB <= {NUM_HNODES{1'b0}};
            node_data <= {NUM_VNODES{1'b0}};
        end 
        else if (stall) begin
            update_visA <= update_visA;
            visA <= visA;
            hidA <= hidA;
            visB <= visB;
            hidB <= hidB;
        end
        else begin
            if (update_visA) begin
                visA <= vis_update; 
                visB <= visB;
                hidA <= hidA;
                hidB <= hid_out;
                node_data <= vis_update;
            end else begin
                visA <= visA;
                visB <= vis_update;
                hidA <= hid_out;
                hidB <= hidB;
                node_data <= vis_update;
            end
            update_visA <= ~update_visA;
        end
    end

endmodule


