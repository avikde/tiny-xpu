module pe (
	clk,
	rst_n,
	weight_ld,
	data_in,
	data_out,
	acc_in,
	acc_out
);
	parameter DATA_WIDTH = 8;
	parameter ACC_WIDTH = 32;
	input wire clk;
	input wire rst_n;
	input wire weight_ld;
	input wire signed [DATA_WIDTH - 1:0] data_in;
	output reg signed [DATA_WIDTH - 1:0] data_out;
	input wire signed [ACC_WIDTH - 1:0] acc_in;
	output reg signed [ACC_WIDTH - 1:0] acc_out;
	reg signed [DATA_WIDTH - 1:0] weight_r;
	wire signed [ACC_WIDTH - 1:0] mult_result;
	assign mult_result = weight_r * data_in;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			weight_r <= 1'sb0;
			data_out <= 1'sb0;
			acc_out <= 1'sb0;
		end
		else if (weight_ld) begin
			weight_r <= acc_in[DATA_WIDTH - 1:0];
			acc_out <= {{ACC_WIDTH - DATA_WIDTH {acc_in[DATA_WIDTH - 1]}}, acc_in[DATA_WIDTH - 1:0]};
			data_out <= 1'sb0;
		end
		else begin
			data_out <= data_in;
			acc_out <= acc_in + mult_result;
		end
endmodule
module pe_col (
	clk,
	rst_n,
	weight_ld,
	data_in,
	data_out,
	acc_in_top,
	acc_out_bottom
);
	parameter signed [31:0] ROWS = 16;
	parameter signed [31:0] DATA_WIDTH = 8;
	parameter signed [31:0] ACC_WIDTH = 32;
	input wire clk;
	input wire rst_n;
	input wire weight_ld;
	input wire signed [(ROWS * DATA_WIDTH) - 1:0] data_in;
	output wire signed [(ROWS * DATA_WIDTH) - 1:0] data_out;
	input wire signed [ACC_WIDTH - 1:0] acc_in_top;
	output wire signed [ACC_WIDTH - 1:0] acc_out_bottom;
	wire signed [ACC_WIDTH - 1:0] acc_wire [0:ROWS + 0];
	assign acc_wire[0] = acc_in_top;
	assign acc_out_bottom = acc_wire[ROWS];
	genvar _gv_r_1;
	generate
		for (_gv_r_1 = 0; _gv_r_1 < ROWS; _gv_r_1 = _gv_r_1 + 1) begin : gen_pe
			localparam r = _gv_r_1;
			pe #(
				.DATA_WIDTH(DATA_WIDTH),
				.ACC_WIDTH(ACC_WIDTH)
			) u_pe(
				.clk(clk),
				.rst_n(rst_n),
				.weight_ld(weight_ld),
				.data_in(data_in[((ROWS - 1) - r) * DATA_WIDTH+:DATA_WIDTH]),
				.data_out(data_out[((ROWS - 1) - r) * DATA_WIDTH+:DATA_WIDTH]),
				.acc_in(acc_wire[r]),
				.acc_out(acc_wire[r + 1])
			);
		end
	endgenerate
endmodule
module array (
	clk,
	rst_n,
	data_in_left,
	data_out_right,
	weight_ld,
	acc_in_top,
	acc_out_bottom
);
	parameter signed [31:0] ROWS = 16;
	parameter signed [31:0] COLS = 16;
	parameter signed [31:0] DATA_WIDTH = 8;
	parameter signed [31:0] ACC_WIDTH = 32;
	input wire clk;
	input wire rst_n;
	input wire signed [(ROWS * DATA_WIDTH) - 1:0] data_in_left;
	output wire signed [(ROWS * DATA_WIDTH) - 1:0] data_out_right;
	input wire [0:COLS - 1] weight_ld;
	input wire signed [(COLS * ACC_WIDTH) - 1:0] acc_in_top;
	output wire signed [(COLS * ACC_WIDTH) - 1:0] acc_out_bottom;
	wire signed [DATA_WIDTH - 1:0] data_wire [0:COLS + 0][0:ROWS - 1];
	genvar _gv_r_2;
	generate
		for (_gv_r_2 = 0; _gv_r_2 < ROWS; _gv_r_2 = _gv_r_2 + 1) begin : gen_data_boundary
			localparam r = _gv_r_2;
			assign data_wire[0][r] = data_in_left[((ROWS - 1) - r) * DATA_WIDTH+:DATA_WIDTH];
			assign data_out_right[((ROWS - 1) - r) * DATA_WIDTH+:DATA_WIDTH] = data_wire[COLS][r];
		end
	endgenerate
	genvar _gv_c_1;
	genvar _gv_r_conn_1;
	generate
		for (_gv_c_1 = 0; _gv_c_1 < COLS; _gv_c_1 = _gv_c_1 + 1) begin : gen_col
			localparam c = _gv_c_1;
			wire signed [(ROWS * DATA_WIDTH) - 1:0] col_data_in;
			wire signed [(ROWS * DATA_WIDTH) - 1:0] col_data_out;
			for (_gv_r_conn_1 = 0; _gv_r_conn_1 < ROWS; _gv_r_conn_1 = _gv_r_conn_1 + 1) begin : gen_conn
				localparam r_conn = _gv_r_conn_1;
				assign col_data_in[((ROWS - 1) - r_conn) * DATA_WIDTH+:DATA_WIDTH] = data_wire[c][r_conn];
				assign data_wire[c + 1][r_conn] = col_data_out[((ROWS - 1) - r_conn) * DATA_WIDTH+:DATA_WIDTH];
			end
			pe_col #(
				.ROWS(ROWS),
				.DATA_WIDTH(DATA_WIDTH),
				.ACC_WIDTH(ACC_WIDTH)
			) u_pe_col(
				.clk(clk),
				.rst_n(rst_n),
				.weight_ld(weight_ld[c]),
				.data_in(col_data_in),
				.data_out(col_data_out),
				.acc_in_top(acc_in_top[((COLS - 1) - c) * ACC_WIDTH+:ACC_WIDTH]),
				.acc_out_bottom(acc_out_bottom[((COLS - 1) - c) * ACC_WIDTH+:ACC_WIDTH])
			);
		end
	endgenerate
endmodule
