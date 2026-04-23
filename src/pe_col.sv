`timescale 1ns / 1ps

// Single column of PEs: ROWS elements stacked vertically with a shared
// weight_ld and acc cascading from top to bottom.

module pe_col #(
    parameter int ROWS       = 16,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    weight_ld,

    // One data lane per row (streams east through each PE)
    input  logic signed [DATA_WIDTH-1:0] data_in  [ROWS],
    output logic signed [DATA_WIDTH-1:0] data_out [ROWS],

    // Accumulator cascade: single input at top, single output at bottom
    input  logic signed [ACC_WIDTH-1:0]  acc_in_top,
    output logic signed [ACC_WIDTH-1:0]  acc_out_bottom
);

    logic signed [ACC_WIDTH-1:0] acc_wire [ROWS+1];

    assign acc_wire[0]      = acc_in_top;
    assign acc_out_bottom   = acc_wire[ROWS];

    genvar r;
    generate
        for (r = 0; r < ROWS; r++) begin : gen_pe
            pe #(
                .DATA_WIDTH (DATA_WIDTH),
                .ACC_WIDTH  (ACC_WIDTH)
            ) u_pe (
                .clk       (clk),
                .rst_n     (rst_n),
                .weight_ld (weight_ld),
                .data_in   (data_in[r]),
                .data_out  (data_out[r]),
                .acc_in    (acc_wire[r]),
                .acc_out   (acc_wire[r+1])
            );
        end
    endgenerate

endmodule
