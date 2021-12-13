//////////////////////////////////////////////////////////////////////////////////
// Company: LEED
// Engineer: Philip Canoza
//
// Create Date: 2019-07-01 16:10:11.154939
// Design Name: vanilla
// Module Name: sigmoidLUT_in8b4p_out17b16p
// Project Name: RBM_FPGA
// Description: An implementation of a sigmoid function via 8 bit LUT.
//              It is assumed that all inputs are unsigned fixed point values.
//
// Additional Comments: Generated by LUT_generator_sigmoid.py
//
//////////////////////////////////////////////////////////////////////////////////
module sigmoidLUT_in8b4p_out17b16p #(
    parameter PRECISION_INPUT_BITS = 8,
    parameter PRECISION_OUTPUT_BITS = 17
)(
    input[PRECISION_INPUT_BITS - 1: 0] sigmoid_in,
    output reg[PRECISION_OUTPUT_BITS - 1: 0] sigmoid_out
);

    always @(sigmoid_in) begin
        case(sigmoid_in)
			8'b00000000: sigmoid_out <= 17'b01000000000000000;    //sigmoid(0.000000) ≈ 0.500000
			8'b00000001: sigmoid_out <= 17'b01000010000000000;    //sigmoid(0.062500) ≈ 0.515625
			8'b00000010: sigmoid_out <= 17'b01000011111111101;    //sigmoid(0.125000) ≈ 0.531204
			8'b00000011: sigmoid_out <= 17'b01000101111110111;    //sigmoid(0.187500) ≈ 0.546738
			8'b00000100: sigmoid_out <= 17'b01000111111101011;    //sigmoid(0.250000) ≈ 0.562180
			8'b00000101: sigmoid_out <= 17'b01001001111010111;    //sigmoid(0.312500) ≈ 0.577499
			8'b00000110: sigmoid_out <= 17'b01001011110111001;    //sigmoid(0.375000) ≈ 0.592667
			8'b00000111: sigmoid_out <= 17'b01001101110010000;    //sigmoid(0.437500) ≈ 0.607666
			8'b00001000: sigmoid_out <= 17'b01001111101011001;    //sigmoid(0.500000) ≈ 0.622452
			8'b00001001: sigmoid_out <= 17'b01010001100010100;    //sigmoid(0.562500) ≈ 0.637024
			8'b00001010: sigmoid_out <= 17'b01010011010111111;    //sigmoid(0.625000) ≈ 0.651352
			8'b00001011: sigmoid_out <= 17'b01010101001011000;    //sigmoid(0.687500) ≈ 0.665405
			8'b00001100: sigmoid_out <= 17'b01010110111011111;    //sigmoid(0.750000) ≈ 0.679184
			8'b00001101: sigmoid_out <= 17'b01011000101010001;    //sigmoid(0.812500) ≈ 0.692642
			8'b00001110: sigmoid_out <= 17'b01011010010101110;    //sigmoid(0.875000) ≈ 0.705780
			8'b00001111: sigmoid_out <= 17'b01011011111110110;    //sigmoid(0.937500) ≈ 0.718597
			8'b00010000: sigmoid_out <= 17'b01011101100100111;    //sigmoid(1.000000) ≈ 0.731064
			8'b00010001: sigmoid_out <= 17'b01011111001000000;    //sigmoid(1.062500) ≈ 0.743164
			8'b00010010: sigmoid_out <= 17'b01100000101000010;    //sigmoid(1.125000) ≈ 0.754913
			8'b00010011: sigmoid_out <= 17'b01100010000101100;    //sigmoid(1.187500) ≈ 0.766296
			8'b00010100: sigmoid_out <= 17'b01100011011111101;    //sigmoid(1.250000) ≈ 0.777298
			8'b00010101: sigmoid_out <= 17'b01100100110110110;    //sigmoid(1.312500) ≈ 0.787933
			8'b00010110: sigmoid_out <= 17'b01100110001010110;    //sigmoid(1.375000) ≈ 0.798187
			8'b00010111: sigmoid_out <= 17'b01100111011011101;    //sigmoid(1.437500) ≈ 0.808060
			8'b00011000: sigmoid_out <= 17'b01101000101001101;    //sigmoid(1.500000) ≈ 0.817581
			8'b00011001: sigmoid_out <= 17'b01101001110100011;    //sigmoid(1.562500) ≈ 0.826706
			8'b00011010: sigmoid_out <= 17'b01101010111100010;    //sigmoid(1.625000) ≈ 0.835480
			8'b00011011: sigmoid_out <= 17'b01101100000001010;    //sigmoid(1.687500) ≈ 0.843903
			8'b00011100: sigmoid_out <= 17'b01101101000011010;    //sigmoid(1.750000) ≈ 0.851959
			8'b00011101: sigmoid_out <= 17'b01101110000010011;    //sigmoid(1.812500) ≈ 0.859665
			8'b00011110: sigmoid_out <= 17'b01101110111110110;    //sigmoid(1.875000) ≈ 0.867035
			8'b00011111: sigmoid_out <= 17'b01101111111000100;    //sigmoid(1.937500) ≈ 0.874084
			8'b00100000: sigmoid_out <= 17'b01110000101111100;    //sigmoid(2.000000) ≈ 0.880798
			8'b00100001: sigmoid_out <= 17'b01110001100100000;    //sigmoid(2.062500) ≈ 0.887207
			8'b00100010: sigmoid_out <= 17'b01110010010110000;    //sigmoid(2.125000) ≈ 0.893311
			8'b00100011: sigmoid_out <= 17'b01110011000101101;    //sigmoid(2.187500) ≈ 0.899124
			8'b00100100: sigmoid_out <= 17'b01110011110010111;    //sigmoid(2.250000) ≈ 0.904648
			8'b00100101: sigmoid_out <= 17'b01110100011110000;    //sigmoid(2.312500) ≈ 0.909912
			8'b00100110: sigmoid_out <= 17'b01110101000110111;    //sigmoid(2.375000) ≈ 0.914902
			8'b00100111: sigmoid_out <= 17'b01110101101101110;    //sigmoid(2.437500) ≈ 0.919647
			8'b00101000: sigmoid_out <= 17'b01110110010010101;    //sigmoid(2.500000) ≈ 0.924149
			8'b00101001: sigmoid_out <= 17'b01110110110101100;    //sigmoid(2.562500) ≈ 0.928406
			8'b00101010: sigmoid_out <= 17'b01110111010110101;    //sigmoid(2.625000) ≈ 0.932449
			8'b00101011: sigmoid_out <= 17'b01110111110110000;    //sigmoid(2.687500) ≈ 0.936279
			8'b00101100: sigmoid_out <= 17'b01111000010011110;    //sigmoid(2.750000) ≈ 0.939911
			8'b00101101: sigmoid_out <= 17'b01111000101111111;    //sigmoid(2.812500) ≈ 0.943344
			8'b00101110: sigmoid_out <= 17'b01111001001010100;    //sigmoid(2.875000) ≈ 0.946594
			8'b00101111: sigmoid_out <= 17'b01111001100011110;    //sigmoid(2.937500) ≈ 0.949677
			8'b00110000: sigmoid_out <= 17'b01111001111011100;    //sigmoid(3.000000) ≈ 0.952576
			8'b00110001: sigmoid_out <= 17'b01111010010010000;    //sigmoid(3.062500) ≈ 0.955322
			8'b00110010: sigmoid_out <= 17'b01111010100111010;    //sigmoid(3.125000) ≈ 0.957916
			8'b00110011: sigmoid_out <= 17'b01111010111011010;    //sigmoid(3.187500) ≈ 0.960358
			8'b00110100: sigmoid_out <= 17'b01111011001110010;    //sigmoid(3.250000) ≈ 0.962677
			8'b00110101: sigmoid_out <= 17'b01111011100000001;    //sigmoid(3.312500) ≈ 0.964859
			8'b00110110: sigmoid_out <= 17'b01111011110001000;    //sigmoid(3.375000) ≈ 0.966919
			8'b00110111: sigmoid_out <= 17'b01111100000000111;    //sigmoid(3.437500) ≈ 0.968857
			8'b00111000: sigmoid_out <= 17'b01111100001111111;    //sigmoid(3.500000) ≈ 0.970688
			8'b00111001: sigmoid_out <= 17'b01111100011110000;    //sigmoid(3.562500) ≈ 0.972412
			8'b00111010: sigmoid_out <= 17'b01111100101011011;    //sigmoid(3.625000) ≈ 0.974045
			8'b00111011: sigmoid_out <= 17'b01111100110111111;    //sigmoid(3.687500) ≈ 0.975571
			8'b00111100: sigmoid_out <= 17'b01111101000011110;    //sigmoid(3.750000) ≈ 0.977020
			8'b00111101: sigmoid_out <= 17'b01111101001110111;    //sigmoid(3.812500) ≈ 0.978378
			8'b00111110: sigmoid_out <= 17'b01111101011001011;    //sigmoid(3.875000) ≈ 0.979660
			8'b00111111: sigmoid_out <= 17'b01111101100011011;    //sigmoid(3.937500) ≈ 0.980881
			8'b01000000: sigmoid_out <= 17'b01111101101100101;    //sigmoid(4.000000) ≈ 0.982010
			8'b01000001: sigmoid_out <= 17'b01111101110101011;    //sigmoid(4.062500) ≈ 0.983078
			8'b01000010: sigmoid_out <= 17'b01111101111101110;    //sigmoid(4.125000) ≈ 0.984100
			8'b01000011: sigmoid_out <= 17'b01111110000101100;    //sigmoid(4.187500) ≈ 0.985046
			8'b01000100: sigmoid_out <= 17'b01111110001100110;    //sigmoid(4.250000) ≈ 0.985931
			8'b01000101: sigmoid_out <= 17'b01111110010011101;    //sigmoid(4.312500) ≈ 0.986771
			8'b01000110: sigmoid_out <= 17'b01111110011010001;    //sigmoid(4.375000) ≈ 0.987564
			8'b01000111: sigmoid_out <= 17'b01111110100000010;    //sigmoid(4.437500) ≈ 0.988312
			8'b01001000: sigmoid_out <= 17'b01111110100110000;    //sigmoid(4.500000) ≈ 0.989014
			8'b01001001: sigmoid_out <= 17'b01111110101011011;    //sigmoid(4.562500) ≈ 0.989670
			8'b01001010: sigmoid_out <= 17'b01111110110000100;    //sigmoid(4.625000) ≈ 0.990295
			8'b01001011: sigmoid_out <= 17'b01111110110101010;    //sigmoid(4.687500) ≈ 0.990875
			8'b01001100: sigmoid_out <= 17'b01111110111001110;    //sigmoid(4.750000) ≈ 0.991425
			8'b01001101: sigmoid_out <= 17'b01111110111110000;    //sigmoid(4.812500) ≈ 0.991943
			8'b01001110: sigmoid_out <= 17'b01111111000001111;    //sigmoid(4.875000) ≈ 0.992416
			8'b01001111: sigmoid_out <= 17'b01111111000101101;    //sigmoid(4.937500) ≈ 0.992874
			8'b01010000: sigmoid_out <= 17'b01111111001001001;    //sigmoid(5.000000) ≈ 0.993301
			8'b01010001: sigmoid_out <= 17'b01111111001100100;    //sigmoid(5.062500) ≈ 0.993713
			8'b01010010: sigmoid_out <= 17'b01111111001111101;    //sigmoid(5.125000) ≈ 0.994095
			8'b01010011: sigmoid_out <= 17'b01111111010010100;    //sigmoid(5.187500) ≈ 0.994446
			8'b01010100: sigmoid_out <= 17'b01111111010101010;    //sigmoid(5.250000) ≈ 0.994781
			8'b01010101: sigmoid_out <= 17'b01111111010111111;    //sigmoid(5.312500) ≈ 0.995102
			8'b01010110: sigmoid_out <= 17'b01111111011010010;    //sigmoid(5.375000) ≈ 0.995392
			8'b01010111: sigmoid_out <= 17'b01111111011100100;    //sigmoid(5.437500) ≈ 0.995667
			8'b01011000: sigmoid_out <= 17'b01111111011110101;    //sigmoid(5.500000) ≈ 0.995926
			8'b01011001: sigmoid_out <= 17'b01111111100000101;    //sigmoid(5.562500) ≈ 0.996170
			8'b01011010: sigmoid_out <= 17'b01111111100010100;    //sigmoid(5.625000) ≈ 0.996399
			8'b01011011: sigmoid_out <= 17'b01111111100100011;    //sigmoid(5.687500) ≈ 0.996628
			8'b01011100: sigmoid_out <= 17'b01111111100110000;    //sigmoid(5.750000) ≈ 0.996826
			8'b01011101: sigmoid_out <= 17'b01111111100111101;    //sigmoid(5.812500) ≈ 0.997025
			8'b01011110: sigmoid_out <= 17'b01111111101001000;    //sigmoid(5.875000) ≈ 0.997192
			8'b01011111: sigmoid_out <= 17'b01111111101010100;    //sigmoid(5.937500) ≈ 0.997375
			8'b01100000: sigmoid_out <= 17'b01111111101011110;    //sigmoid(6.000000) ≈ 0.997528
			8'b01100001: sigmoid_out <= 17'b01111111101101000;    //sigmoid(6.062500) ≈ 0.997681
			8'b01100010: sigmoid_out <= 17'b01111111101110001;    //sigmoid(6.125000) ≈ 0.997818
			8'b01100011: sigmoid_out <= 17'b01111111101111010;    //sigmoid(6.187500) ≈ 0.997955
			8'b01100100: sigmoid_out <= 17'b01111111110000010;    //sigmoid(6.250000) ≈ 0.998077
			8'b01100101: sigmoid_out <= 17'b01111111110001001;    //sigmoid(6.312500) ≈ 0.998184
			8'b01100110: sigmoid_out <= 17'b01111111110010001;    //sigmoid(6.375000) ≈ 0.998306
			8'b01100111: sigmoid_out <= 17'b01111111110010111;    //sigmoid(6.437500) ≈ 0.998398
			8'b01101000: sigmoid_out <= 17'b01111111110011110;    //sigmoid(6.500000) ≈ 0.998505
			8'b01101001: sigmoid_out <= 17'b01111111110100100;    //sigmoid(6.562500) ≈ 0.998596
			8'b01101010: sigmoid_out <= 17'b01111111110101001;    //sigmoid(6.625000) ≈ 0.998672
			8'b01101011: sigmoid_out <= 17'b01111111110101110;    //sigmoid(6.687500) ≈ 0.998749
			8'b01101100: sigmoid_out <= 17'b01111111110110011;    //sigmoid(6.750000) ≈ 0.998825
			8'b01101101: sigmoid_out <= 17'b01111111110111000;    //sigmoid(6.812500) ≈ 0.998901
			8'b01101110: sigmoid_out <= 17'b01111111110111100;    //sigmoid(6.875000) ≈ 0.998962
			8'b01101111: sigmoid_out <= 17'b01111111111000000;    //sigmoid(6.937500) ≈ 0.999023
			8'b01110000: sigmoid_out <= 17'b01111111111000100;    //sigmoid(7.000000) ≈ 0.999084
			8'b01110001: sigmoid_out <= 17'b01111111111001000;    //sigmoid(7.062500) ≈ 0.999146
			8'b01110010: sigmoid_out <= 17'b01111111111001011;    //sigmoid(7.125000) ≈ 0.999191
			8'b01110011: sigmoid_out <= 17'b01111111111001110;    //sigmoid(7.187500) ≈ 0.999237
			8'b01110100: sigmoid_out <= 17'b01111111111010001;    //sigmoid(7.250000) ≈ 0.999283
			8'b01110101: sigmoid_out <= 17'b01111111111010100;    //sigmoid(7.312500) ≈ 0.999329
			8'b01110110: sigmoid_out <= 17'b01111111111010111;    //sigmoid(7.375000) ≈ 0.999374
			8'b01110111: sigmoid_out <= 17'b01111111111011001;    //sigmoid(7.437500) ≈ 0.999405
			8'b01111000: sigmoid_out <= 17'b01111111111011100;    //sigmoid(7.500000) ≈ 0.999451
			8'b01111001: sigmoid_out <= 17'b01111111111011110;    //sigmoid(7.562500) ≈ 0.999481
			8'b01111010: sigmoid_out <= 17'b01111111111100000;    //sigmoid(7.625000) ≈ 0.999512
			8'b01111011: sigmoid_out <= 17'b01111111111100010;    //sigmoid(7.687500) ≈ 0.999542
			8'b01111100: sigmoid_out <= 17'b01111111111100100;    //sigmoid(7.750000) ≈ 0.999573
			8'b01111101: sigmoid_out <= 17'b01111111111100101;    //sigmoid(7.812500) ≈ 0.999588
			8'b01111110: sigmoid_out <= 17'b01111111111100111;    //sigmoid(7.875000) ≈ 0.999619
			8'b01111111: sigmoid_out <= 17'b01111111111101001;    //sigmoid(7.937500) ≈ 0.999649
			8'b10000000: sigmoid_out <= 17'b01111111111101010;    //sigmoid(8.000000) ≈ 0.999664
			8'b10000001: sigmoid_out <= 17'b01111111111101011;    //sigmoid(8.062500) ≈ 0.999680
			8'b10000010: sigmoid_out <= 17'b01111111111101101;    //sigmoid(8.125000) ≈ 0.999710
			8'b10000011: sigmoid_out <= 17'b01111111111101110;    //sigmoid(8.187500) ≈ 0.999725
			8'b10000100: sigmoid_out <= 17'b01111111111101111;    //sigmoid(8.250000) ≈ 0.999741
			8'b10000101: sigmoid_out <= 17'b01111111111110000;    //sigmoid(8.312500) ≈ 0.999756
			8'b10000110: sigmoid_out <= 17'b01111111111110001;    //sigmoid(8.375000) ≈ 0.999771
			8'b10000111: sigmoid_out <= 17'b01111111111110010;    //sigmoid(8.437500) ≈ 0.999786
			8'b10001000: sigmoid_out <= 17'b01111111111110011;    //sigmoid(8.500000) ≈ 0.999802
			8'b10001001: sigmoid_out <= 17'b01111111111110011;    //sigmoid(8.562500) ≈ 0.999802
			8'b10001010: sigmoid_out <= 17'b01111111111110100;    //sigmoid(8.625000) ≈ 0.999817
			8'b10001011: sigmoid_out <= 17'b01111111111110101;    //sigmoid(8.687500) ≈ 0.999832
			8'b10001100: sigmoid_out <= 17'b01111111111110110;    //sigmoid(8.750000) ≈ 0.999847
			8'b10001101: sigmoid_out <= 17'b01111111111110110;    //sigmoid(8.812500) ≈ 0.999847
			8'b10001110: sigmoid_out <= 17'b01111111111110111;    //sigmoid(8.875000) ≈ 0.999863
			8'b10001111: sigmoid_out <= 17'b01111111111110111;    //sigmoid(8.937500) ≈ 0.999863
			8'b10010000: sigmoid_out <= 17'b01111111111111000;    //sigmoid(9.000000) ≈ 0.999878
			8'b10010001: sigmoid_out <= 17'b01111111111111000;    //sigmoid(9.062500) ≈ 0.999878
			8'b10010010: sigmoid_out <= 17'b01111111111111001;    //sigmoid(9.125000) ≈ 0.999893
			8'b10010011: sigmoid_out <= 17'b01111111111111001;    //sigmoid(9.187500) ≈ 0.999893
			8'b10010100: sigmoid_out <= 17'b01111111111111010;    //sigmoid(9.250000) ≈ 0.999908
			8'b10010101: sigmoid_out <= 17'b01111111111111010;    //sigmoid(9.312500) ≈ 0.999908
			8'b10010110: sigmoid_out <= 17'b01111111111111010;    //sigmoid(9.375000) ≈ 0.999908
			8'b10010111: sigmoid_out <= 17'b01111111111111011;    //sigmoid(9.437500) ≈ 0.999924
			8'b10011000: sigmoid_out <= 17'b01111111111111011;    //sigmoid(9.500000) ≈ 0.999924
			8'b10011001: sigmoid_out <= 17'b01111111111111011;    //sigmoid(9.562500) ≈ 0.999924
			8'b10011010: sigmoid_out <= 17'b01111111111111100;    //sigmoid(9.625000) ≈ 0.999939
			8'b10011011: sigmoid_out <= 17'b01111111111111100;    //sigmoid(9.687500) ≈ 0.999939
			8'b10011100: sigmoid_out <= 17'b01111111111111100;    //sigmoid(9.750000) ≈ 0.999939
			8'b10011101: sigmoid_out <= 17'b01111111111111100;    //sigmoid(9.812500) ≈ 0.999939
			8'b10011110: sigmoid_out <= 17'b01111111111111101;    //sigmoid(9.875000) ≈ 0.999954
			8'b10011111: sigmoid_out <= 17'b01111111111111101;    //sigmoid(9.937500) ≈ 0.999954
			8'b10100000: sigmoid_out <= 17'b01111111111111101;    //sigmoid(10.000000) ≈ 0.999954
			8'b10100001: sigmoid_out <= 17'b01111111111111101;    //sigmoid(10.062500) ≈ 0.999954
			8'b10100010: sigmoid_out <= 17'b01111111111111101;    //sigmoid(10.125000) ≈ 0.999954
			8'b10100011: sigmoid_out <= 17'b01111111111111110;    //sigmoid(10.187500) ≈ 0.999969
			8'b10100100: sigmoid_out <= 17'b01111111111111110;    //sigmoid(10.250000) ≈ 0.999969
			8'b10100101: sigmoid_out <= 17'b01111111111111110;    //sigmoid(10.312500) ≈ 0.999969
			8'b10100110: sigmoid_out <= 17'b01111111111111110;    //sigmoid(10.375000) ≈ 0.999969
			8'b10100111: sigmoid_out <= 17'b01111111111111110;    //sigmoid(10.437500) ≈ 0.999969
			8'b10101000: sigmoid_out <= 17'b01111111111111110;    //sigmoid(10.500000) ≈ 0.999969
			8'b10101001: sigmoid_out <= 17'b01111111111111110;    //sigmoid(10.562500) ≈ 0.999969
			8'b10101010: sigmoid_out <= 17'b01111111111111110;    //sigmoid(10.625000) ≈ 0.999969
			8'b10101011: sigmoid_out <= 17'b01111111111111111;    //sigmoid(10.687500) ≈ 0.999985
			8'b10101100: sigmoid_out <= 17'b01111111111111111;    //sigmoid(10.750000) ≈ 0.999985
			8'b10101101: sigmoid_out <= 17'b01111111111111111;    //sigmoid(10.812500) ≈ 0.999985
			8'b10101110: sigmoid_out <= 17'b01111111111111111;    //sigmoid(10.875000) ≈ 0.999985
			8'b10101111: sigmoid_out <= 17'b01111111111111111;    //sigmoid(10.937500) ≈ 0.999985
			8'b10110000: sigmoid_out <= 17'b01111111111111111;    //sigmoid(11.000000) ≈ 0.999985
			8'b10110001: sigmoid_out <= 17'b01111111111111111;    //sigmoid(11.062500) ≈ 0.999985
			8'b10110010: sigmoid_out <= 17'b01111111111111111;    //sigmoid(11.125000) ≈ 0.999985
			8'b10110011: sigmoid_out <= 17'b01111111111111111;    //sigmoid(11.187500) ≈ 0.999985
			8'b10110100: sigmoid_out <= 17'b01111111111111111;    //sigmoid(11.250000) ≈ 0.999985
			8'b10110101: sigmoid_out <= 17'b01111111111111111;    //sigmoid(11.312500) ≈ 0.999985
			8'b10110110: sigmoid_out <= 17'b01111111111111111;    //sigmoid(11.375000) ≈ 0.999985
			8'b10110111: sigmoid_out <= 17'b01111111111111111;    //sigmoid(11.437500) ≈ 0.999985
			8'b10111000: sigmoid_out <= 17'b01111111111111111;    //sigmoid(11.500000) ≈ 0.999985
			8'b10111001: sigmoid_out <= 17'b01111111111111111;    //sigmoid(11.562500) ≈ 0.999985
			8'b10111010: sigmoid_out <= 17'b01111111111111111;    //sigmoid(11.625000) ≈ 0.999985
			8'b10111011: sigmoid_out <= 17'b01111111111111111;    //sigmoid(11.687500) ≈ 0.999985
			8'b10111100: sigmoid_out <= 17'b01111111111111111;    //sigmoid(11.750000) ≈ 0.999985
			8'b10111101: sigmoid_out <= 17'b10000000000000000;    //sigmoid(11.812500) ≈ 1.000000
			8'b10111110: sigmoid_out <= 17'b10000000000000000;    //sigmoid(11.875000) ≈ 1.000000
			8'b10111111: sigmoid_out <= 17'b10000000000000000;    //sigmoid(11.937500) ≈ 1.000000
			8'b11000000: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.000000) ≈ 1.000000
			8'b11000001: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.062500) ≈ 1.000000
			8'b11000010: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.125000) ≈ 1.000000
			8'b11000011: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.187500) ≈ 1.000000
			8'b11000100: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.250000) ≈ 1.000000
			8'b11000101: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.312500) ≈ 1.000000
			8'b11000110: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.375000) ≈ 1.000000
			8'b11000111: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.437500) ≈ 1.000000
			8'b11001000: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.500000) ≈ 1.000000
			8'b11001001: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.562500) ≈ 1.000000
			8'b11001010: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.625000) ≈ 1.000000
			8'b11001011: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.687500) ≈ 1.000000
			8'b11001100: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.750000) ≈ 1.000000
			8'b11001101: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.812500) ≈ 1.000000
			8'b11001110: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.875000) ≈ 1.000000
			8'b11001111: sigmoid_out <= 17'b10000000000000000;    //sigmoid(12.937500) ≈ 1.000000
			8'b11010000: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.000000) ≈ 1.000000
			8'b11010001: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.062500) ≈ 1.000000
			8'b11010010: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.125000) ≈ 1.000000
			8'b11010011: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.187500) ≈ 1.000000
			8'b11010100: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.250000) ≈ 1.000000
			8'b11010101: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.312500) ≈ 1.000000
			8'b11010110: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.375000) ≈ 1.000000
			8'b11010111: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.437500) ≈ 1.000000
			8'b11011000: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.500000) ≈ 1.000000
			8'b11011001: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.562500) ≈ 1.000000
			8'b11011010: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.625000) ≈ 1.000000
			8'b11011011: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.687500) ≈ 1.000000
			8'b11011100: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.750000) ≈ 1.000000
			8'b11011101: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.812500) ≈ 1.000000
			8'b11011110: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.875000) ≈ 1.000000
			8'b11011111: sigmoid_out <= 17'b10000000000000000;    //sigmoid(13.937500) ≈ 1.000000
			8'b11100000: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.000000) ≈ 1.000000
			8'b11100001: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.062500) ≈ 1.000000
			8'b11100010: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.125000) ≈ 1.000000
			8'b11100011: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.187500) ≈ 1.000000
			8'b11100100: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.250000) ≈ 1.000000
			8'b11100101: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.312500) ≈ 1.000000
			8'b11100110: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.375000) ≈ 1.000000
			8'b11100111: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.437500) ≈ 1.000000
			8'b11101000: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.500000) ≈ 1.000000
			8'b11101001: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.562500) ≈ 1.000000
			8'b11101010: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.625000) ≈ 1.000000
			8'b11101011: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.687500) ≈ 1.000000
			8'b11101100: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.750000) ≈ 1.000000
			8'b11101101: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.812500) ≈ 1.000000
			8'b11101110: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.875000) ≈ 1.000000
			8'b11101111: sigmoid_out <= 17'b10000000000000000;    //sigmoid(14.937500) ≈ 1.000000
			8'b11110000: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.000000) ≈ 1.000000
			8'b11110001: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.062500) ≈ 1.000000
			8'b11110010: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.125000) ≈ 1.000000
			8'b11110011: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.187500) ≈ 1.000000
			8'b11110100: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.250000) ≈ 1.000000
			8'b11110101: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.312500) ≈ 1.000000
			8'b11110110: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.375000) ≈ 1.000000
			8'b11110111: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.437500) ≈ 1.000000
			8'b11111000: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.500000) ≈ 1.000000
			8'b11111001: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.562500) ≈ 1.000000
			8'b11111010: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.625000) ≈ 1.000000
			8'b11111011: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.687500) ≈ 1.000000
			8'b11111100: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.750000) ≈ 1.000000
			8'b11111101: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.812500) ≈ 1.000000
			8'b11111110: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.875000) ≈ 1.000000
			8'b11111111: sigmoid_out <= 17'b10000000000000000;    //sigmoid(15.937500) ≈ 1.000000

        endcase
    end
endmodule