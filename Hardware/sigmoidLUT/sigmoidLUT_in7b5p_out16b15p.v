//////////////////////////////////////////////////////////////////////////////////
// Company: LEED
// Engineer: Philip Canoza
//
// Create Date: 2019-10-17 17:13:46.377687
// Design Name: vanilla
// Module Name: sigmoidLUT_in7b5p_out16b15p
// Project Name: RBM_FPGA
// Description: An implementation of a sigmoid function via 7 bit LUT.
//              It is assumed that all inputs are unsigned fixed point values.
//
// Additional Comments: Generated by LUT_generator_sigmoid.py
//
//////////////////////////////////////////////////////////////////////////////////
module sigmoidLUT_in7b5p_out16b15p #(
    parameter PRECISION_INPUT_BITS = 7,
    parameter PRECISION_OUTPUT_BITS = 16
)(
    input[PRECISION_INPUT_BITS - 1: 0] sigmoid_in,
    output reg[PRECISION_OUTPUT_BITS - 1: 0] sigmoid_out
);

    always @(sigmoid_in) begin
        case(sigmoid_in)
			7'b0000000: sigmoid_out <= 16'b0100000000000000;    //sigmoid(0.000000) ≈ 0.500000
			7'b0000001: sigmoid_out <= 16'b0100000100000000;    //sigmoid(0.031250) ≈ 0.507812
			7'b0000010: sigmoid_out <= 16'b0100001000000000;    //sigmoid(0.062500) ≈ 0.515625
			7'b0000011: sigmoid_out <= 16'b0100001011111111;    //sigmoid(0.093750) ≈ 0.523407
			7'b0000100: sigmoid_out <= 16'b0100001111111111;    //sigmoid(0.125000) ≈ 0.531219
			7'b0000101: sigmoid_out <= 16'b0100010011111101;    //sigmoid(0.156250) ≈ 0.538971
			7'b0000110: sigmoid_out <= 16'b0100010111111100;    //sigmoid(0.187500) ≈ 0.546753
			7'b0000111: sigmoid_out <= 16'b0100011011111001;    //sigmoid(0.218750) ≈ 0.554474
			7'b0001000: sigmoid_out <= 16'b0100011111110101;    //sigmoid(0.250000) ≈ 0.562164
			7'b0001001: sigmoid_out <= 16'b0100100011110001;    //sigmoid(0.281250) ≈ 0.569855
			7'b0001010: sigmoid_out <= 16'b0100100111101011;    //sigmoid(0.312500) ≈ 0.577484
			7'b0001011: sigmoid_out <= 16'b0100101011100101;    //sigmoid(0.343750) ≈ 0.585114
			7'b0001100: sigmoid_out <= 16'b0100101111011100;    //sigmoid(0.375000) ≈ 0.592651
			7'b0001101: sigmoid_out <= 16'b0100110011010011;    //sigmoid(0.406250) ≈ 0.600189
			7'b0001110: sigmoid_out <= 16'b0100110111001000;    //sigmoid(0.437500) ≈ 0.607666
			7'b0001111: sigmoid_out <= 16'b0100111010111011;    //sigmoid(0.468750) ≈ 0.615082
			7'b0010000: sigmoid_out <= 16'b0100111110101101;    //sigmoid(0.500000) ≈ 0.622467
			7'b0010001: sigmoid_out <= 16'b0101000010011100;    //sigmoid(0.531250) ≈ 0.629761
			7'b0010010: sigmoid_out <= 16'b0101000110001010;    //sigmoid(0.562500) ≈ 0.637024
			7'b0010011: sigmoid_out <= 16'b0101001001110110;    //sigmoid(0.593750) ≈ 0.644226
			7'b0010100: sigmoid_out <= 16'b0101001101100000;    //sigmoid(0.625000) ≈ 0.651367
			7'b0010101: sigmoid_out <= 16'b0101010001000111;    //sigmoid(0.656250) ≈ 0.658417
			7'b0010110: sigmoid_out <= 16'b0101010100101100;    //sigmoid(0.687500) ≈ 0.665405
			7'b0010111: sigmoid_out <= 16'b0101011000001111;    //sigmoid(0.718750) ≈ 0.672333
			7'b0011000: sigmoid_out <= 16'b0101011011101111;    //sigmoid(0.750000) ≈ 0.679169
			7'b0011001: sigmoid_out <= 16'b0101011111001101;    //sigmoid(0.781250) ≈ 0.685944
			7'b0011010: sigmoid_out <= 16'b0101100010101000;    //sigmoid(0.812500) ≈ 0.692627
			7'b0011011: sigmoid_out <= 16'b0101100110000001;    //sigmoid(0.843750) ≈ 0.699249
			7'b0011100: sigmoid_out <= 16'b0101101001010111;    //sigmoid(0.875000) ≈ 0.705780
			7'b0011101: sigmoid_out <= 16'b0101101100101010;    //sigmoid(0.906250) ≈ 0.712219
			7'b0011110: sigmoid_out <= 16'b0101101111111011;    //sigmoid(0.937500) ≈ 0.718597
			7'b0011111: sigmoid_out <= 16'b0101110011001001;    //sigmoid(0.968750) ≈ 0.724884
			7'b0100000: sigmoid_out <= 16'b0101110110010011;    //sigmoid(1.000000) ≈ 0.731049
			7'b0100001: sigmoid_out <= 16'b0101111001011011;    //sigmoid(1.031250) ≈ 0.737152
			7'b0100010: sigmoid_out <= 16'b0101111100100000;    //sigmoid(1.062500) ≈ 0.743164
			7'b0100011: sigmoid_out <= 16'b0101111111100010;    //sigmoid(1.093750) ≈ 0.749084
			7'b0100100: sigmoid_out <= 16'b0110000010100001;    //sigmoid(1.125000) ≈ 0.754913
			7'b0100101: sigmoid_out <= 16'b0110000101011101;    //sigmoid(1.156250) ≈ 0.760651
			7'b0100110: sigmoid_out <= 16'b0110001000010110;    //sigmoid(1.187500) ≈ 0.766296
			7'b0100111: sigmoid_out <= 16'b0110001011001100;    //sigmoid(1.218750) ≈ 0.771851
			7'b0101000: sigmoid_out <= 16'b0110001101111111;    //sigmoid(1.250000) ≈ 0.777313
			7'b0101001: sigmoid_out <= 16'b0110010000101110;    //sigmoid(1.281250) ≈ 0.782654
			7'b0101010: sigmoid_out <= 16'b0110010011011011;    //sigmoid(1.312500) ≈ 0.787933
			7'b0101011: sigmoid_out <= 16'b0110010110000100;    //sigmoid(1.343750) ≈ 0.793091
			7'b0101100: sigmoid_out <= 16'b0110011000101011;    //sigmoid(1.375000) ≈ 0.798187
			7'b0101101: sigmoid_out <= 16'b0110011011001110;    //sigmoid(1.406250) ≈ 0.803162
			7'b0101110: sigmoid_out <= 16'b0110011101101111;    //sigmoid(1.437500) ≈ 0.808075
			7'b0101111: sigmoid_out <= 16'b0110100000001100;    //sigmoid(1.468750) ≈ 0.812866
			7'b0110000: sigmoid_out <= 16'b0110100010100110;    //sigmoid(1.500000) ≈ 0.817566
			7'b0110001: sigmoid_out <= 16'b0110100100111101;    //sigmoid(1.531250) ≈ 0.822174
			7'b0110010: sigmoid_out <= 16'b0110100111010010;    //sigmoid(1.562500) ≈ 0.826721
			7'b0110011: sigmoid_out <= 16'b0110101001100011;    //sigmoid(1.593750) ≈ 0.831146
			7'b0110100: sigmoid_out <= 16'b0110101011110001;    //sigmoid(1.625000) ≈ 0.835480
			7'b0110101: sigmoid_out <= 16'b0110101101111100;    //sigmoid(1.656250) ≈ 0.839722
			7'b0110110: sigmoid_out <= 16'b0110110000000101;    //sigmoid(1.687500) ≈ 0.843903
			7'b0110111: sigmoid_out <= 16'b0110110010001010;    //sigmoid(1.718750) ≈ 0.847961
			7'b0111000: sigmoid_out <= 16'b0110110100001101;    //sigmoid(1.750000) ≈ 0.851959
			7'b0111001: sigmoid_out <= 16'b0110110110001101;    //sigmoid(1.781250) ≈ 0.855865
			7'b0111010: sigmoid_out <= 16'b0110111000001001;    //sigmoid(1.812500) ≈ 0.859650
			7'b0111011: sigmoid_out <= 16'b0110111010000100;    //sigmoid(1.843750) ≈ 0.863403
			7'b0111100: sigmoid_out <= 16'b0110111011111011;    //sigmoid(1.875000) ≈ 0.867035
			7'b0111101: sigmoid_out <= 16'b0110111101110000;    //sigmoid(1.906250) ≈ 0.870605
			7'b0111110: sigmoid_out <= 16'b0110111111100010;    //sigmoid(1.937500) ≈ 0.874084
			7'b0111111: sigmoid_out <= 16'b0111000001010001;    //sigmoid(1.968750) ≈ 0.877472
			7'b1000000: sigmoid_out <= 16'b0111000010111110;    //sigmoid(2.000000) ≈ 0.880798
			7'b1000001: sigmoid_out <= 16'b0111000100101000;    //sigmoid(2.031250) ≈ 0.884033
			7'b1000010: sigmoid_out <= 16'b0111000110010000;    //sigmoid(2.062500) ≈ 0.887207
			7'b1000011: sigmoid_out <= 16'b0111000111110101;    //sigmoid(2.093750) ≈ 0.890289
			7'b1000100: sigmoid_out <= 16'b0111001001011000;    //sigmoid(2.125000) ≈ 0.893311
			7'b1000101: sigmoid_out <= 16'b0111001010111000;    //sigmoid(2.156250) ≈ 0.896240
			7'b1000110: sigmoid_out <= 16'b0111001100010110;    //sigmoid(2.187500) ≈ 0.899109
			7'b1000111: sigmoid_out <= 16'b0111001101110010;    //sigmoid(2.218750) ≈ 0.901917
			7'b1001000: sigmoid_out <= 16'b0111001111001100;    //sigmoid(2.250000) ≈ 0.904663
			7'b1001001: sigmoid_out <= 16'b0111010000100011;    //sigmoid(2.281250) ≈ 0.907318
			7'b1001010: sigmoid_out <= 16'b0111010001111000;    //sigmoid(2.312500) ≈ 0.909912
			7'b1001011: sigmoid_out <= 16'b0111010011001011;    //sigmoid(2.343750) ≈ 0.912445
			7'b1001100: sigmoid_out <= 16'b0111010100011011;    //sigmoid(2.375000) ≈ 0.914886
			7'b1001101: sigmoid_out <= 16'b0111010101101010;    //sigmoid(2.406250) ≈ 0.917297
			7'b1001110: sigmoid_out <= 16'b0111010110110111;    //sigmoid(2.437500) ≈ 0.919647
			7'b1001111: sigmoid_out <= 16'b0111011000000010;    //sigmoid(2.468750) ≈ 0.921936
			7'b1010000: sigmoid_out <= 16'b0111011001001010;    //sigmoid(2.500000) ≈ 0.924133
			7'b1010001: sigmoid_out <= 16'b0111011010010001;    //sigmoid(2.531250) ≈ 0.926300
			7'b1010010: sigmoid_out <= 16'b0111011011010110;    //sigmoid(2.562500) ≈ 0.928406
			7'b1010011: sigmoid_out <= 16'b0111011100011001;    //sigmoid(2.593750) ≈ 0.930450
			7'b1010100: sigmoid_out <= 16'b0111011101011011;    //sigmoid(2.625000) ≈ 0.932465
			7'b1010101: sigmoid_out <= 16'b0111011110011010;    //sigmoid(2.656250) ≈ 0.934387
			7'b1010110: sigmoid_out <= 16'b0111011111011000;    //sigmoid(2.687500) ≈ 0.936279
			7'b1010111: sigmoid_out <= 16'b0111100000010100;    //sigmoid(2.718750) ≈ 0.938110
			7'b1011000: sigmoid_out <= 16'b0111100001001111;    //sigmoid(2.750000) ≈ 0.939911
			7'b1011001: sigmoid_out <= 16'b0111100010001000;    //sigmoid(2.781250) ≈ 0.941650
			7'b1011010: sigmoid_out <= 16'b0111100011000000;    //sigmoid(2.812500) ≈ 0.943359
			7'b1011011: sigmoid_out <= 16'b0111100011110110;    //sigmoid(2.843750) ≈ 0.945007
			7'b1011100: sigmoid_out <= 16'b0111100100101010;    //sigmoid(2.875000) ≈ 0.946594
			7'b1011101: sigmoid_out <= 16'b0111100101011101;    //sigmoid(2.906250) ≈ 0.948151
			7'b1011110: sigmoid_out <= 16'b0111100110001111;    //sigmoid(2.937500) ≈ 0.949677
			7'b1011111: sigmoid_out <= 16'b0111100110111111;    //sigmoid(2.968750) ≈ 0.951141
			7'b1100000: sigmoid_out <= 16'b0111100111101110;    //sigmoid(3.000000) ≈ 0.952576
			7'b1100001: sigmoid_out <= 16'b0111101000011100;    //sigmoid(3.031250) ≈ 0.953979
			7'b1100010: sigmoid_out <= 16'b0111101001001000;    //sigmoid(3.062500) ≈ 0.955322
			7'b1100011: sigmoid_out <= 16'b0111101001110011;    //sigmoid(3.093750) ≈ 0.956635
			7'b1100100: sigmoid_out <= 16'b0111101010011101;    //sigmoid(3.125000) ≈ 0.957916
			7'b1100101: sigmoid_out <= 16'b0111101011000110;    //sigmoid(3.156250) ≈ 0.959167
			7'b1100110: sigmoid_out <= 16'b0111101011101101;    //sigmoid(3.187500) ≈ 0.960358
			7'b1100111: sigmoid_out <= 16'b0111101100010100;    //sigmoid(3.218750) ≈ 0.961548
			7'b1101000: sigmoid_out <= 16'b0111101100111001;    //sigmoid(3.250000) ≈ 0.962677
			7'b1101001: sigmoid_out <= 16'b0111101101011101;    //sigmoid(3.281250) ≈ 0.963776
			7'b1101010: sigmoid_out <= 16'b0111101110000000;    //sigmoid(3.312500) ≈ 0.964844
			7'b1101011: sigmoid_out <= 16'b0111101110100011;    //sigmoid(3.343750) ≈ 0.965912
			7'b1101100: sigmoid_out <= 16'b0111101111000100;    //sigmoid(3.375000) ≈ 0.966919
			7'b1101101: sigmoid_out <= 16'b0111101111100100;    //sigmoid(3.406250) ≈ 0.967896
			7'b1101110: sigmoid_out <= 16'b0111110000000011;    //sigmoid(3.437500) ≈ 0.968842
			7'b1101111: sigmoid_out <= 16'b0111110000100010;    //sigmoid(3.468750) ≈ 0.969788
			7'b1110000: sigmoid_out <= 16'b0111110000111111;    //sigmoid(3.500000) ≈ 0.970673
			7'b1110001: sigmoid_out <= 16'b0111110001011100;    //sigmoid(3.531250) ≈ 0.971558
			7'b1110010: sigmoid_out <= 16'b0111110001111000;    //sigmoid(3.562500) ≈ 0.972412
			7'b1110011: sigmoid_out <= 16'b0111110010010011;    //sigmoid(3.593750) ≈ 0.973236
			7'b1110100: sigmoid_out <= 16'b0111110010101101;    //sigmoid(3.625000) ≈ 0.974030
			7'b1110101: sigmoid_out <= 16'b0111110011000111;    //sigmoid(3.656250) ≈ 0.974823
			7'b1110110: sigmoid_out <= 16'b0111110011100000;    //sigmoid(3.687500) ≈ 0.975586
			7'b1110111: sigmoid_out <= 16'b0111110011111000;    //sigmoid(3.718750) ≈ 0.976318
			7'b1111000: sigmoid_out <= 16'b0111110100001111;    //sigmoid(3.750000) ≈ 0.977020
			7'b1111001: sigmoid_out <= 16'b0111110100100110;    //sigmoid(3.781250) ≈ 0.977722
			7'b1111010: sigmoid_out <= 16'b0111110100111100;    //sigmoid(3.812500) ≈ 0.978394
			7'b1111011: sigmoid_out <= 16'b0111110101010001;    //sigmoid(3.843750) ≈ 0.979034
			7'b1111100: sigmoid_out <= 16'b0111110101100110;    //sigmoid(3.875000) ≈ 0.979675
			7'b1111101: sigmoid_out <= 16'b0111110101111010;    //sigmoid(3.906250) ≈ 0.980286
			7'b1111110: sigmoid_out <= 16'b0111110110001101;    //sigmoid(3.937500) ≈ 0.980865
			7'b1111111: sigmoid_out <= 16'b0111110110100000;    //sigmoid(3.968750) ≈ 0.981445

        endcase
    end
endmodule