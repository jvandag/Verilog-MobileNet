// Synchronous dual-port word RAM: host word port and engine byte port.
// Ownership is enforced by wb_cnn; no writes are permitted to collide.
module cnn_ram #(parameter WORDS=3072)(
    input wire clk,
    input wire host_en, input wire [3:0] host_we,
    input wire [17:0] host_addr, input wire [31:0] host_data,
    output reg [31:0] host_q,
    input wire [17:0] core_addr, input wire core_we,
    input wire [7:0] core_data, output reg [31:0] core_word,
    output wire [7:0] core_q
);
    reg [1:0] lane;
    assign core_q = core_word[lane*8 +: 8];
    for(genvar b=0;b<4;b=b+1) begin: lanes
        (* ram_style="block" *) reg [7:0] mem [0:WORDS-1];
        always @(posedge clk) begin
            if(host_en) begin
                host_q[b*8 +: 8] <= mem[host_addr >> 2];
                if(host_we[b]) mem[host_addr >> 2] <= host_data[b*8 +: 8];
            end
        end
        always @(posedge clk) begin
            core_word[b*8 +: 8] <= mem[core_addr >> 2];
            if(core_we && core_addr[1:0]==b) mem[core_addr >> 2] <= core_data;
        end
    end
    always @(posedge clk) lane <= core_addr[1:0];
endmodule
