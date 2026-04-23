`timescale 1ns / 1ps

// Systolic array: ROWS x COLS grid of processing elements.
//
// Dataflow (weight-stationary, Kung 1982):
//   - Activations stream east  (→)  one lane per row
//   - Partial sums cascade south (↓) one lane per column
//   - Weights are loaded once per tile (weight_in=1), then held stationary
//
// External pre-staggering: the driver (or previous layer) is responsible
// for staggering inputs in time.  Row r should begin streaming r cycles
// after row 0, so that all K elements of output row i arrive at the
// leftmost column of row i's PE lane at the same effective cycle.
// The module simply passes data_in[r] straight to data_wire[r][0] with no
// internal delay lines.
//
// This design enables direct layer chaining: layer N's data_out feeds into
// layer N+1's data_in, and the inter-layer pipeline registers handle the
// staggering.  No de-skewing buffers are needed between layers.
//
// Output: acc_out[c] is wired directly to acc_wire[ROWS][c].  Column c of
// output row i is valid at cycle i + ROWS + c (1-indexed).  The driver
// reads each column at its own tick rather than waiting for alignment.
//
// Array size is set by ROWS and COLS parameters.  When built via Verilator
// the values are overridden at elaboration time with -GROWS=N -GCOLS=N so
// no source edits are needed to change the tile size.

module array #(
    parameter int ROWS       = 16,
    parameter int COLS       = 16,
    parameter int DATA_WIDTH = 8,
    parameter int ACC_WIDTH  = 32
) (
    input  logic clk,
    input  logic rst_n,
    input  logic en,
    input  logic relu_en,   // when 1, clamp acc_out negatives to 0 (hardware ReLU)

    // Activation inputs — one per row, stream east through the array.
    // The caller is responsible for staggering: row r should present its
    // first element r cycles after row 0 begins streaming.
    input  logic signed [DATA_WIDTH-1:0] data_in  [ROWS],
    // Activation outputs — rightmost column passthrough (useful for chaining)
    output logic signed [DATA_WIDTH-1:0] data_out [ROWS],

    // Weight loading — systolic cascade from top edge.
    // When weight_in=1: weight_in_top[c] enters row 0, cascades down over ROWS cycles.
    // When weight_in=0: acc_wire[0] is '0 (normal accumulation).
    input  logic                         weight_in,
    input  logic signed [DATA_WIDTH-1:0] weight_in_top [COLS],

    // Accumulated results from the bottom row, one per column.
    // Not de-skewed: acc_out[c] for output row i is valid at cycle i + ROWS + c
    // (1-indexed).  Driver reads each column at its own tick.
    output logic signed [ACC_WIDTH-1:0]  acc_out [COLS],

    // Requantization output stage (active when requant_en=1).
    // Converts acc_out int32 → int8 using fixed-point multiply-shift-saturate.
    // q_out[c] is valid at the same cycle as acc_out[c].
    input  logic                         requant_en,
    input  logic signed [ACC_WIDTH-1:0]  bias_in  [COLS],  // per-column int32 bias
    input  logic        [30:0]           M0,                // fixed-point multiplier
    input  logic        [4:0]            rshift,            // right-shift amount
    input  logic signed [DATA_WIDTH-1:0] zero_pt,           // output zero-point
    output logic signed [DATA_WIDTH-1:0] q_out    [COLS]    // int8 requantized output
);

    // Horizontal data wires: data_wire[r][c] feeds PE(r,c).data_in
    // Column 0 = skewed input, column COLS = rightmost passthrough output
    logic signed [DATA_WIDTH-1:0] data_wire [ROWS][COLS+1];

    // Vertical accumulator wires: acc_wire[r][c] feeds PE(r,c).acc_in
    // Row 0 = zero (top boundary), row ROWS = raw PE output (before de-skew)
    logic signed [ACC_WIDTH-1:0] acc_wire [ROWS+1][COLS];

    // Vertical weight tag wires: weight_wire[r][c] feeds PE(r,c).weight_in
    // Row 0 = top-level weight_in, row ROWS+1 = unused (bottom boundary)
    logic weight_wire [ROWS+1][COLS];

    // ----------------------------------------------------------------
    // Boundary: data_out and top acc boundary
    // ----------------------------------------------------------------

    genvar r_b, c_b;
    generate
        for (r_b = 0; r_b < ROWS; r_b++) begin : gen_data_out
            assign data_out[r_b] = data_wire[r_b][COLS];
        end
        for (c_b = 0; c_b < COLS; c_b++) begin : gen_acc_top
            assign acc_wire[0][c_b] = weight_in
                ? {{(ACC_WIDTH-DATA_WIDTH){weight_in_top[c_b][DATA_WIDTH-1]}},
                   weight_in_top[c_b]}
                : '0;
            assign weight_wire[0][c_b] = weight_in;
        end
    endgenerate

    // ----------------------------------------------------------------
    // Input: direct connection to column 0 of the data wire.
    // External pre-staggering is the caller's responsibility.
    // ----------------------------------------------------------------

    genvar r_in;
    generate
        for (r_in = 0; r_in < ROWS; r_in++) begin : gen_data_in
            assign data_wire[r_in][0] = data_in[r_in];
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
                    .weight_in (weight_wire[r][c]),
                    .data_in   (data_wire[r][c]),
                    .data_out  (data_wire[r][c+1]),
                    .acc_in    (acc_wire[r][c]),
                    .acc_out   (acc_wire[r+1][c]),
                    .weight_out(weight_wire[r+1][c])
                );
            end
        end
    endgenerate

    // ----------------------------------------------------------------
    // Output: direct wires, no de-skew registers.
    // acc_out[c] = acc_wire[ROWS][c]; column c of row i is valid at
    // cycle i + ROWS + c (1-indexed).
    // relu_en without requant_en still clamps acc_out negatives to 0.
    // ----------------------------------------------------------------

    genvar c_ds;
    generate
        for (c_ds = 0; c_ds < COLS; c_ds++) begin : gen_acc_out
            assign acc_out[c_ds] = (relu_en && !requant_en && acc_wire[ROWS][c_ds][ACC_WIDTH-1])
                                   ? '0
                                   : acc_wire[ROWS][c_ds];
        end
    endgenerate

    // ----------------------------------------------------------------
    // Requantization stage (combinational, active when requant_en=1).
    // When requant_en=0 q_out is undefined — only acc_out should be read.
    // relu_en is forwarded to requant so activation and requant share one
    // control signal.
    //
    // acc_wire[ROWS] is a 2-D array slice; iverilog does not support passing
    // such slices directly as port connections, so copy into a flat 1-D wire.
    // ----------------------------------------------------------------

    logic signed [ACC_WIDTH-1:0] acc_bottom [COLS];
    genvar c_rq;
    generate
        for (c_rq = 0; c_rq < COLS; c_rq++) begin : gen_acc_bottom
            assign acc_bottom[c_rq] = acc_wire[ROWS][c_rq];
        end
    endgenerate

    requant #(
        .COLS      (COLS),
        .ACC_WIDTH (ACC_WIDTH),
        .DAT_WIDTH (DATA_WIDTH)
    ) u_requant (
        .acc_in  (acc_bottom),
        .bias_in (bias_in),
        .M0      (M0),
        .rshift  (rshift),
        .zero_pt (zero_pt),
        .relu_en (relu_en),
        .q_out   (q_out)
    );

endmodule
