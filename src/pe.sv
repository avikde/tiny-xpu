`timescale 1ns / 1ps

// Processing Element (PE) for systolic array
// Performs multiply-accumulate: acc += weight * data_in
// Passes data through to neighboring PEs

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
    input  logic                         weight_ld,

    // Partial sum cascade
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
