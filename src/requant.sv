`timescale 1ns / 1ps

// Requantization output stage: int32 accumulator → int8 activation.
//
// Implements the standard fixed-point requantization used by integer-only
// neural network inference:
//
//   biased    = acc_in[c] + bias_in[c]          (int32 + int32)
//   product   = biased * M0                     (int32 × uint31 → int63)
//   shifted   = product >>> rshift              (arithmetic right-shift)
//   adjusted  = shifted + zero_pt               (add output zero-point)
//   q_out[c]  = sat8(adjusted)                  (saturate to [-128, 127])
//               followed by max(q, 0) if relu_en
//
// M0 is a 31-bit unsigned fixed-point multiplier derived from the combined
// layer scale:  M0 = round(a_scale * w_scale / y_scale * 2^rshift)
// Typically rshift = 31, so M0 ∈ [0, 2^31).
//
// All logic is purely combinational; output is valid in the same cycle
// that acc_in changes.

module requant #(
    parameter int COLS      = 64,
    parameter int ACC_WIDTH = 32,
    parameter int DAT_WIDTH = 8
) (
    // Accumulated int32 results from the systolic array (one per column)
    input  logic signed [ACC_WIDTH-1:0] acc_in   [COLS],
    // Per-column int32 bias (constant for the lifetime of the layer)
    input  logic signed [ACC_WIDTH-1:0] bias_in  [COLS],
    // Scalar layer constants (same for all columns)
    input  logic        [30:0]          M0,       // fixed-point multiplier
    input  logic        [4:0]           rshift,   // right-shift amount (0-31)
    input  logic signed [DAT_WIDTH-1:0] zero_pt,  // output zero-point
    input  logic                        relu_en,  // 1 → clamp output to [0, 127]
    // Requantized int8 output
    output logic signed [DAT_WIDTH-1:0] q_out    [COLS]
);

    genvar c;
    generate
        for (c = 0; c < COLS; c++) begin : gen_requant_col

            // Step 1: add bias
            logic signed [ACC_WIDTH-1:0] biased;
            assign biased = acc_in[c] + bias_in[c];

            // Step 2: multiply by M0 — int32 × uint31 produces a 63-bit signed result.
            // Sign-extend biased to 63 bits and zero-extend M0 to 63 bits so the
            // tool sees explicit widths for the signed multiply.
            logic signed [62:0] product;
            assign product = $signed({{31{biased[ACC_WIDTH-1]}}, biased}) *
                             $signed({32'b0, M0});

            // Step 3: arithmetic right-shift (truncation toward negative infinity).
            // Explicit $signed() cast ensures arithmetic (sign-extending) shift in all tools.
            logic signed [62:0] shifted;
            assign shifted = $signed(product) >>> rshift;

            // Step 4: add output zero-point (narrow back to 32-bit range)
            // $signed(shifted[31:0]) re-applies sign so the addition is signed.
            logic signed [ACC_WIDTH-1:0] adjusted;
            assign adjusted = $signed(shifted[ACC_WIDTH-1:0]) + $signed({{(ACC_WIDTH-DAT_WIDTH){zero_pt[DAT_WIDTH-1]}}, zero_pt});

            // Step 5: saturate to [-128, 127]
            logic signed [DAT_WIDTH-1:0] saturated;
            assign saturated = (adjusted > $signed(32'sd127))  ?  8'sd127 :
                               (adjusted < $signed(-32'sd128)) ? -8'sd128 :
                               adjusted[DAT_WIDTH-1:0];

            // Step 6: optional ReLU — clamp negatives to 0
            assign q_out[c] = (relu_en && saturated[DAT_WIDTH-1]) ? '0 : saturated;

        end
    endgenerate

endmodule
