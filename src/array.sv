`timescale 1ns / 1ps

// Systolic array: ROWS x COLS grid of processing elements.
//
// Dataflow (weight-stationary, Kung 1982):
//   - Activations stream east  (→)  one lane per row
//   - Partial sums cascade south (↓) one lane per column
//   - Weights are loaded once per tile (weight_ld=1), then held stationary
//
// Input skewing: row r receives its activation r cycles later than row 0,
// so that all K elements of output row i arrive at column 0 of their
// respective PE rows at the same effective cycle.  This is implemented
// with shift-register delay lines inside this module; the driver simply
// presents row i of A on data_in[*] at streaming cycle i with no manual
// staggering.
//
// Output: acc_out[c] is wired directly to acc_wire[ROWS][c] with no
// de-skew registers.  Column c of output row i is valid at cycle
// i + ROWS + c (1-indexed).  The driver reads each column at its own
// cycle rather than waiting for all columns to align.
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
    // Present row i of A on data_in[*] at streaming cycle i; the module
    // applies the per-row skew internally.
    input  logic signed [DATA_WIDTH-1:0] data_in  [ROWS],
    // Activation outputs — rightmost column passthrough (useful for chaining)
    output logic signed [DATA_WIDTH-1:0] data_out [ROWS],

    // Weight loading — broadcast ld signal, individual value per PE.
    // Flattened to 1-D (row-major: index = r*COLS + c) so VPI/cocotb can
    // address each element directly; 2-D unpacked ports get merged into a
    // packed vector by iverilog's VPI layer and can't be sub-indexed.
    input  logic                         weight_ld,
    input  logic signed [DATA_WIDTH-1:0] weight_in [ROWS*COLS],

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

    // ----------------------------------------------------------------
    // Boundary: data_out and top acc boundary
    // ----------------------------------------------------------------

    genvar r_b, c_b;
    generate
        for (r_b = 0; r_b < ROWS; r_b++) begin : gen_data_out
            assign data_out[r_b] = data_wire[r_b][COLS];
        end
        for (c_b = 0; c_b < COLS; c_b++) begin : gen_acc_top
            assign acc_wire[0][c_b] = '0;
        end
    endgenerate

    // ----------------------------------------------------------------
    // Input skewing: row r is delayed by r clock cycles.
    // skew_buf[r][k]: stage k of the shift register for row r.
    //   row 0 → direct wire (0 FFs)
    //   row 1 → 1 FF  (stage 0 only)
    //   row r → r FFs (stages 0..r-1)
    // ----------------------------------------------------------------

    logic signed [DATA_WIDTH-1:0] skew_buf [ROWS][ROWS]; // [row][stage]

    genvar r_sk, k_sk;
    generate
        // Row 0: no delay, connect directly
        assign data_wire[0][0] = data_in[0];

        // Stage 0 for rows 1..ROWS-1: latch data_in
        for (r_sk = 1; r_sk < ROWS; r_sk++) begin : gen_skew_s0
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) skew_buf[r_sk][0] <= '0;
                else        skew_buf[r_sk][0] <= data_in[r_sk];
            end
        end

        // Chain stages 1..r-1 for rows 2..ROWS-1
        for (r_sk = 2; r_sk < ROWS; r_sk++) begin : gen_skew_chain_row
            for (k_sk = 1; k_sk < r_sk; k_sk++) begin : gen_skew_chain_stage
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) skew_buf[r_sk][k_sk] <= '0;
                    else        skew_buf[r_sk][k_sk] <= skew_buf[r_sk][k_sk-1];
                end
            end
        end

        // Connect shift register outputs to data_wire column 0
        for (r_sk = 1; r_sk < ROWS; r_sk++) begin : gen_skew_connect
            assign data_wire[r_sk][0] = skew_buf[r_sk][r_sk-1];
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
                    .weight_in (weight_in[r*COLS + c]),
                    .weight_ld (weight_ld),
                    .acc_in    (acc_wire[r][c]),
                    .acc_out   (acc_wire[r+1][c])
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
