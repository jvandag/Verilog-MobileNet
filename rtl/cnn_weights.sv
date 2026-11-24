// Immutable, initialized dual-read-port weight/bias ROM. No flip-flop weight bank.
module cnn_weights(
    input wire clk,host_en,
    input wire [17:0] host_addr,core_addr,
    output wire [31:0] host_q,core_word
);
`ifdef CNN_FAST_SIM
    // Cycle-equivalent model for long image regressions. The separate primitive
    // ROM test checks every word and both physical read ports before using this.
    reg [31:0] words [0:54897];
    reg [31:0] hq,cq;
    initial $readmemh("weights.mem",words);
    always @(posedge clk) begin
        if(host_en) hq<=words[host_addr[17:2]];
        cq<=words[core_addr[17:2]];
    end
    assign host_q=hq;
    assign core_word=cq;
`else
    wire [31:0] host_words [0:53];
    wire [31:0] core_words [0:53];
    reg [5:0] host_bank,core_bank;
    `include "cnn_weight_banks.vh"
    always @(posedge clk) begin
        if(host_en) host_bank<=host_addr[17:12];
        core_bank<=core_addr[17:12];
    end
    assign host_q=host_words[host_bank];
    assign core_word=core_words[core_bank];
`endif
endmodule
