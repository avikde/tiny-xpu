`timescale 1ns / 1ps

// Systolic array: COLS columns of pe_col, each containing ROWS PEs.
//
// Dataflow (weight-stationary, Kung 1982):
//   - Activations stream east  (→) one lane per row
//   - Partial sums cascade south (↓) one lane per column
//   - Weights are loaded once per tile (weight_ld[c]=1), then held stationary
//
// External pre-staggering: the driver is responsible for staggering inputs
// in time.  Row r should begin streaming r cycles after row 0, so that all
// K elements of output row i arrive at the leftmost column of row i's PE
// lane at the same effective cycle.  No internal delay lines.
//
// This design enables direct layer chaining: layer N's data_out feeds into
// layer N+1's data_in, and the inter-layer pipeline registers handle the
// staggering.  No de-skewing buffers are needed between layers.
//
// Output: acc_out_bottom[c] for output row i is valid at cycle i + ROWS + c.
// The driver reads each column at its own tick rather than waiting for alignment.

module array #(
    parameter int ROWS       = 16,
    parameter int COLS       = 16,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
) (
    input  logic clk,
    input  logic rst_n,

    // Activation inputs — one per row, stream east through the array.
    // The caller is responsible for staggering: row r should present its
    // first element r cycles after row 0 begins streaming.
    input  logic signed [DATA_WIDTH-1:0] data_in_left  [ROWS],
    // Activation outputs — rightmost column passthrough (useful for chaining)
    output logic signed [DATA_WIDTH-1:0] data_out_right [ROWS],

    // Weight loading — one signal per column, shared by all PEs in the column.
    input  logic                         weight_ld [COLS],

    // Accumulator cascade: one input per column at the top, one output per
    // column at the bottom.  When weight_ld[c]=1, acc_in_top[c] carries the
    // weight/bias to be loaded; when weight_ld[c]=0, it carries the initial
    // partial sum (typically 0).
    input  logic signed [ACC_WIDTH-1:0]  acc_in_top [COLS],
    output logic signed [ACC_WIDTH-1:0]  acc_out_bottom [COLS]
);

    // Inter-column data wires: data_wire[c][r] connects column c-1 output
    // to column c input.  data_wire[0] is the left input; data_wire[COLS]
    // is the right output.
    logic signed [DATA_WIDTH-1:0] data_wire [COLS+1][ROWS];

    // ----------------------------------------------------------------
    // Left/right boundaries
    // ----------------------------------------------------------------

    genvar r;
    generate
        for (r = 0; r < ROWS; r++) begin : gen_data_boundary
            assign data_wire[0][r]     = data_in_left[r];
            assign data_out_right[r]   = data_wire[COLS][r];
        end
    endgenerate

    // ----------------------------------------------------------------
    // PE columns
    // ----------------------------------------------------------------

    genvar c;
    generate
        for (c = 0; c < COLS; c++) begin : gen_col
            pe_col #(
                .ROWS       (ROWS),
                .DATA_WIDTH (DATA_WIDTH),
                .ACC_WIDTH  (ACC_WIDTH)
            ) u_pe_col (
                .clk            (clk),
                .rst_n          (rst_n),
                .weight_ld      (weight_ld[c]),
                .data_in        (data_wire[c]),
                .data_out       (data_wire[c+1]),
                .acc_in_top     (acc_in_top[c]),
                .acc_out_bottom (acc_out_bottom[c])
            );
        end
    endgenerate

endmodule
