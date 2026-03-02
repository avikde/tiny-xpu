`timescale 1ns / 1ps

// Systolic array: ROWS x COLS grid of processing elements.
//
// Dataflow (weight-stationary, Kung 1982):
//   - Activations stream east  (→)  one lane per row
//   - Partial sums cascade south (↓) one lane per column
//   - Weights are loaded once per tile (weight_ld=1), then held stationary
//
// After K streaming cycles the bottom row's acc_out holds one output tile of
// C = A × B, where A is ROWS×K and B is K×COLS.
//
// Array size is set by ROWS and COLS parameters.  When built via Verilator the
// values are overridden at elaboration time with -GROWS=N -GCOLS=N so no
// source edits are needed to change the tile size.

module array #(
    parameter int ROWS       = 2,
    parameter int COLS       = 2,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
) (
    input  logic clk,
    input  logic rst_n,
    input  logic en,

    // Activation inputs — one per row, stream east through the array
    input  logic signed [DATA_WIDTH-1:0] data_in  [ROWS],
    // Activation outputs — rightmost column passthrough (useful for chaining)
    output logic signed [DATA_WIDTH-1:0] data_out [ROWS],

    // Weight loading — broadcast ld signal, individual value per PE
    input  logic                         weight_ld,
    input  logic signed [DATA_WIDTH-1:0] weight_in [ROWS][COLS],

    // Accumulated results from the bottom row, one per column
    output logic signed [ACC_WIDTH-1:0]  acc_out [COLS]
);

    // Horizontal data wires: data_wire[r][c] feeds PE(r,c).data_in
    // Column 0 = external input, column COLS = external output
    logic signed [DATA_WIDTH-1:0] data_wire [ROWS][COLS+1];

    // Vertical accumulator wires: acc_wire[r][c] feeds PE(r,c).acc_in
    // Row 0 = zero (top boundary), row ROWS = external output
    logic signed [ACC_WIDTH-1:0] acc_wire [ROWS+1][COLS];

    // ----------------------------------------------------------------
    // Boundary connections
    // ----------------------------------------------------------------

    genvar r_b, c_b;

    generate
        for (r_b = 0; r_b < ROWS; r_b++) begin : gen_data_boundary
            assign data_wire[r_b][0]    = data_in[r_b];
            assign data_out[r_b]        = data_wire[r_b][COLS];
        end

        for (c_b = 0; c_b < COLS; c_b++) begin : gen_acc_boundary
            assign acc_wire[0][c_b] = '0;
            assign acc_out[c_b]     = acc_wire[ROWS][c_b];
        end
    endgenerate

    // ----------------------------------------------------------------
    // PE array
    // ----------------------------------------------------------------

    genvar r, c;

    generate
        for (r = 0; r < ROWS; r++) begin : gen_row
            for (c = 0; c < COLS; c++) begin : gen_col
                pe #(
                    .DATA_WIDTH (DATA_WIDTH),
                    .ACC_WIDTH  (ACC_WIDTH)
                ) u_pe (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .en        (en),
                    .data_in   (data_wire[r][c]),
                    .data_out  (data_wire[r][c+1]),
                    .weight_in (weight_in[r][c]),
                    .weight_ld (weight_ld),
                    .acc_in    (acc_wire[r][c]),
                    .acc_out   (acc_wire[r+1][c])
                );
            end
        end
    endgenerate

endmodule
