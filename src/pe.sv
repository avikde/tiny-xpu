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

    // From the left
    input  logic signed [DATA_WIDTH-1:0] data_in,
    // To the right
    output logic signed [DATA_WIDTH-1:0] data_out,

    // Partial sum cascade (carries weights when weight_ld=1, bias/partial sums when weight_ld=0)
    input  logic signed [ACC_WIDTH-1:0]  acc_in,
    output logic signed [ACC_WIDTH-1:0]  acc_out
);

    logic signed [DATA_WIDTH-1:0] weight_r;
    logic signed [ACC_WIDTH-1:0]  mult_result;

    assign mult_result = weight_r * data_in;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_r <= '0;
            data_out <= '0;
            acc_out  <= '0;
        end else if (weight_ld) begin
            // Load weight from acc_in and cascade it south
            weight_r <= acc_in[DATA_WIDTH-1:0];
            acc_out  <= {{(ACC_WIDTH-DATA_WIDTH){acc_in[DATA_WIDTH-1]}},
                         acc_in[DATA_WIDTH-1:0]};
            data_out <= '0;
        end else if (en) begin
            data_out <= data_in;
            acc_out  <= acc_in + mult_result;
        end
    end

endmodule
