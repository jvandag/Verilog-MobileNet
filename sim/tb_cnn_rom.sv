`timescale 1ns/1ps
module tb_cnn_rom;
    reg clk=0,en=0;
    reg [17:0] ha=0,ca=0;
    wire [31:0] hq,cq;
    reg [31:0] expected [0:54897];
    reg [31:0] held;
    always #5 clk=~clk;
    cnn_weights dut(.clk(clk),.host_en(en),.host_addr(ha),.core_addr(ca),.host_q(hq),.core_word(cq));
    initial begin
        $readmemh("weights.mem",expected);
        repeat(20) @(negedge clk);
        for(integer i=0;i<54898;i=i+1) begin
            @(negedge clk); en=1; ha=i*4; ca=(54897-i)*4;
            @(posedge clk); #1;
            if(hq!==expected[i] || cq!==expected[54897-i]) $fatal(1,"physical ROM word %0d %h %h",i,hq,cq);
        end
        held=hq;
        @(negedge clk); en=0; ha=0;
        repeat(4) begin @(posedge clk); #1; if(hq!==held) $fatal(1,"host hold"); end
        $display("ROM TEST PASSED: 54898 words on each physical port"); $finish;
    end
    initial begin #10000000; $fatal(1,"ROM timeout"); end
endmodule
