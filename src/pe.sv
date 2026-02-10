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

    // Data flowing through the systolic array
    input  logic signed [DATA_WIDTH-1:0] data_in,
    output logic signed [DATA_WIDTH-1:0] data_out,

    // Weight (stationary)
    input  logic signed [DATA_WIDTH-1:0] weight_in,
    // control signal "weight load"
    // When high, the PE latches weight_in into its internal weight_r register.
    input  logic                         weight_ld,

    // Partial sum cascade
    input  logic signed [ACC_WIDTH-1:0]  acc_in,
    output logic signed [ACC_WIDTH-1:0]  acc_out
);

    // weight_r is signed [7:0], an 8-bit signed value (range -128 to +127, i.e. int8)
    logic signed [DATA_WIDTH-1:0] weight_r;
    // data_in is signed [7:0] 
    logic signed [ACC_WIDTH-1:0]  mult_result;
    // The result is assigned to mult_result which is signed [31:0]
    assign mult_result = weight_r * data_in;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_r <= '0;
            data_out <= '0;
            acc_out  <= '0;
        end else begin
            if (weight_ld)
                weight_r <= weight_in;

            if (en) begin
                data_out <= data_in;
                acc_out  <= acc_in + mult_result;
            end
        end
    end

endmodule
