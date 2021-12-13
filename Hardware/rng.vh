//////////////////////////////////////////////////////////////////////////////////
// Header file to set up seeds, taps, and lengths for LFSR PseudoRNG
//////////////////////////////////////////////////////////////////////////////////
`ifndef RNG

`define RNG

//------------------------------------------------------------------------------
//	Section:	Length Macro
//	Desc:		A macro to define the LENGTH of the feedback registers
//------------------------------------------------------------------------------
`define RNG_LENS(x)	 32

//------------------------------------------------------------------------------
//	Section:	Seed Macro
//	Desc:		A macro to denote the SEED parameters of the LFSRs.
//------------------------------------------------------------------------------
`define NUM_SEEDS 47
`define RNG_SEEDS(x)	\
			((x == 0) ? 32'hefaccafe : \
			(x == 1) ? 32'hbebeefef : \
			(x == 2) ? 32'ha93644b1 : \
			(x == 3) ? 32'h5a7cd60e : \
			(x == 4) ? 32'h362356d7 : \
			(x == 5) ? 32'hec3344ca : \
			(x == 6) ? 32'haf23aba5 : \
			(x == 7) ? 32'h236ffee5 : \
			(x == 8) ? 32'he1fabcde : \
			(x == 9) ? 32'h2ecccecc : \
            (x == 10) ? 32'hbf5ea3af : \
			(x == 11) ? 32'ha9cca471 : \
			(x == 12) ? 32'h54cff6fc : \
			(x == 13) ? 32'h32ad42d1 : \
			(x == 14) ? 32'h43bc4eca : \
			(x == 15) ? 32'haa3324a5 : \
			(x == 16) ? 32'h2f123435 : \
			(x == 17) ? 32'h1afce2fa : \
			(x == 18) ? 32'h2ddcddc : \
            (x == 19) ? 32'haf2ee125 : \
			(x == 20) ? 32'h2fffcee5 : \
			(x == 21) ? 32'he1fab5de : \
			(x == 22) ? 32'h2e7aaecc : \
            (x == 23) ? 32'hbf5ea2af : \
			(x == 24) ? 32'ha9c33471 : \
			(x == 25) ? 32'h5423f6fc : \
			(x == 26) ? 32'h32fdf2d1 : \
			(x == 27) ? 32'h4eeddeea : \
			(x == 28) ? 32'haa3124a5 : \
			(x == 29) ? 32'h2f168435 : \
			(x == 30) ? 32'h1af932fa : \
			(x == 31) ? 32'h233dcddc : \
			(x == 32) ? 32'h2f12ff35 : \
			(x == 33) ? 32'h1af232fa : \
			(x == 34) ? 32'h2d11ddc : \
            (x == 35) ? 32'haf33e125 : \
			(x == 36) ? 32'h2fef3ee5 : \
			(x == 37) ? 32'he12abbde : \
			(x == 37) ? 32'h2efeaecc : \
            (x == 38) ? 32'hbfaba2af : \
			(x == 39) ? 32'ha9c33ab1 : \
			(x == 40) ? 32'h5323f6fc : \
			(x == 41) ? 32'h32f987d1 : \
			(x == 42) ? 32'h4ee99eea : \
			(x == 43) ? 32'haa312045 : \
			(x == 44) ? 32'h23348435 : \
			(x == 45) ? 32'h1afe62fa : \
			(x == 46) ? 32'h2f3dc2dc : \
			32'hdeaddead)


//------------------------------------------------------------------------------
//	Section:	Taps Macro
//	Desc:		A macro to define TAPS parameters of LFSRs. 
//              See https://www.xilinx.com/support/documentation/application_notes/xapp052.pdf
//              For descritipon of good taps to use
//------------------------------------------------------------------------------
`define RNG_TAPS(x) 	\
			((x== 16) ? 16'hb400 : \
			(x == 17) ? 17'h12000 : \
			(x == 32) ? 32'h80200003 : \
			(x == 33) ? 33'h100080000 : \
			(x == 48) ? 48'hc00000180000 :\
			(x == 49) ? 49'h1008000000000 : \
			(x == 64) ? 64'hd800000000000000 : \
			32'hdeaddead)

`endif

