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
    input  logic                    weight_in,

    // From the left
    input  logic signed [DATA_WIDTH-1:0] data_in,
    // To the right
    output logic signed [DATA_WIDTH-1:0] data_out,

    // Partial sum cascade (carries weights when weight_in=1, bias/partial sums when weight_in=0)
    input  logic signed [ACC_WIDTH-1:0]  acc_in,
    output logic signed [ACC_WIDTH-1:0]  acc_out,

    // Tag propagation to south PE (registered weight_in, one cycle delay)
    output logic                    weight_out
);

    logic signed [DATA_WIDTH-1:0] weight_r;
    logic weight_valid;  // Set when weight loaded, cleared when weight_in goes low
    logic signed [ACC_WIDTH-1:0]  mult_result;

    assign mult_result = weight_r * data_in;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_r     <= '0;
            weight_valid <= 1'b0;
            data_out     <= '0;
            acc_out      <= '0;
            weight_out   <= 1'b0;
        end else if (weight_in) begin
            // Tagged: cascade weight down regardless
            acc_out    <= {{(ACC_WIDTH-DATA_WIDTH){acc_in[DATA_WIDTH-1]}},
                           acc_in[DATA_WIDTH-1:0]};
            weight_out <= 1'b1;
            // Only latch weight on first cycle (when not yet valid)
            if (!weight_valid) begin
                weight_r     <= acc_in[DATA_WIDTH-1:0];
                weight_valid <= 1'b1;
            end
        end else begin
            // Untagged: clear valid flag for next load session
            weight_valid <= 1'b0;
            weight_out   <= 1'b0;
            if (en) begin
                data_out <= data_in;
                acc_out  <= acc_in + mult_result;
            end
        end
    end

endmodule
