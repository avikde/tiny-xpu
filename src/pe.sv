`timescale 1ns / 1ps

// Processing Element (PE) for systolic array, named as in Kung (1982)
// See documentation in README

module pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
) (
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    en,
    input  logic                    weight_ld,

    input  logic signed [DATA_WIDTH-1:0] data_in,
    output logic signed [DATA_WIDTH-1:0] data_out,

    // Partial sum cascade (DUAL-USE: also carries weights during weight_ld=1)
    // When weight_ld=1: acc_in carries 8-bit weight from PE above, latched into
    // weight_r, and cascaded down through acc_out to PE below.
    // When weight_ld=0: normal int32 accumulator for MAC operation.
    input  logic signed [ACC_WIDTH-1:0]  acc_in,
    output logic signed [ACC_WIDTH-1:0]  acc_out
);

    // weight_r is signed [7:0], an 8-bit signed value (range -128 to +127, i.e. int8)
    logic signed [DATA_WIDTH-1:0] weight_r;
    logic weight_valid;  // Set when weight loaded, cleared when weight_ld goes low
    // data_in is signed [7:0] 
    logic signed [ACC_WIDTH-1:0]  mult_result;
    // The result is assigned to mult_result which is signed [31:0]
    assign mult_result = weight_r * data_in;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_r <= '0;
            weight_valid <= 1'b0;
            data_out <= '0;
            acc_out  <= '0;
        end else begin
            if (weight_ld) begin
                // Cascade weight down regardless
                acc_out <= {{(ACC_WIDTH-DATA_WIDTH){acc_in[DATA_WIDTH-1]}},
                           acc_in[DATA_WIDTH-1:0]};
                // Only latch weight on first cycle of weight_ld (when not yet valid)
                if (!weight_valid) begin
                    weight_r <= acc_in[DATA_WIDTH-1:0];
                    weight_valid <= 1'b1;
                end
            end else begin
                // weight_ld is low: clear valid flag for next load session
                weight_valid <= 1'b0;
                if (en) begin
                    data_out <= data_in;
                    acc_out  <= acc_in + mult_result;
                end
            end
        end
    end

endmodule
