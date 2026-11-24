`timescale 1ns/1ps
module tb_wb_cnn;
    reg clk=0,reset=1,cyc=0,stb=0,we=0;
    reg [31:0] addr=0,data=0;
    reg [3:0] sel=15;
    wire ack,irq;
    wire [31:0] rd;
    always #5 clk=~clk;
    wb_cnn dut(.i_clk(clk),.i_reset(reset),.i_wb_cyc(cyc),.i_wb_stb(stb),.i_wb_we(we),
        .i_wb_addr(addr),.i_wb_data(data),.i_wb_sel(sel),.o_wb_ack(ack),.o_wb_data(rd),.o_irq(irq));
    reg [31:0] constants [0:54897];
    reg [31:0] image [0:3071];
    reg [7:0] golden [0:200000];
    integer checked=0,frame=0,count=1;
    string path;
    reg [31:0] q;
    task automatic transfer(input bit write_en,input [31:0] a,d,output [31:0] r);
        integer timeout;
        begin
            @(negedge clk); cyc=1; stb=1; we=write_en; addr=32'h80050000+a; data=d;
            @(negedge clk); stb=0; timeout=0;
            while(!ack) begin @(negedge clk); timeout=timeout+1; if(timeout>12) $fatal(1,"Wishbone timeout %h",a); end
            r=rd;
            @(negedge clk); cyc=0; we=0;
            @(negedge clk);
        end
    endtask
    task automatic wr(input [31:0] a,d);
        reg [31:0] unused;
        transfer(1,a,d,unused);
    endtask
    always @(posedge clk) if(dut.core.trace_valid) begin
        if(dut.core.trace_data !== golden[checked])
            $fatal(1,"frame=%0d layer=%0d addr=%0d expected=%h actual=%h flat=%0d",
                frame,dut.core.trace_layer,dut.core.trace_addr,golden[checked],dut.core.trace_data,checked);
        checked=checked+1;
    end
    initial begin
        if($value$plusargs("COUNT=%d",count)) begin end
        $readmemh("fixtures/constants.mem",constants);
        repeat(8) @(negedge clk); reset=0;
        transfer(0,0,0,q); if(q!==32'h434e4e33) $fatal(1,"ID %h",q);
        wr(4096,32'h12345678);
        sel=5; wr(4096,32'haabbccdd); sel=15;
        transfer(0,4096,0,q); if(q!==32'h12bb56dd) $fatal(1,"SEL %h",q);
        // A captured request cancelled before execution must not write or ACK.
        @(negedge clk); cyc=1; stb=1; we=1; addr=32'h80051000; data=0;
        @(negedge clk); cyc=0; stb=0;
        repeat(4) begin @(negedge clk); if(ack) $fatal(1,"cancelled ACK"); end
        transfer(0,4096,0,q); if(q!==32'h12bb56dd) $fatal(1,"cancelled write");
        transfer(0,8,0,q); if(!q[3]) $fatal(1,"built-in MODEL_VALID");
        wr(4,1); transfer(0,28,0,q); if(q!=5) $fatal(1,"START without complete image"); wr(16,3);
        wr(48,0);
        wr(52,0); transfer(0,28,0,q); if(q!=7) $fatal(1,"ROM write protection"); wr(16,3);
        wr(48,0);
        for(integer i=0;i<54898;i=i+1) begin
            transfer(0,52,0,q); if(q!==constants[i]) $fatal(1,"constant readback %0d %h",i,q);
        end
        wr(4,4); transfer(0,8,0,q); if(q[3]!==1) $fatal(1,"commit %h",q);
        wr(12,3);
        for(frame=0;frame<count;frame=frame+1) begin
            path=$sformatf("fixtures/image_%0d.mem",frame); $readmemh(path,image);
            path=$sformatf("fixtures/golden_%0d.mem",frame); $readmemh(path,golden);
            checked=0;
            for(integer i=0;i<3072;i=i+1) wr(4096+i*4,image[i]);
            wr(4,1);
            if(frame==0) begin
                // Protected write must ACK promptly without damaging inference.
                wr(4096,0); transfer(0,28,0,q); if(q!=3) $fatal(1,"busy write");
                wr(16,2);
            end
            wait(irq);
            transfer(0,8,0,q); if(q[2:0]!==2) $fatal(1,"completion status %h",q);
            for(integer i=0;i<4;i=i+1) begin
                transfer(0,256+i*4,0,q);
                if($signed(q)!=$signed(golden[checked-4+i])) $fatal(1,"score %0d",i);
            end
            $display("FRAME %0d PASS: %0d tensor bytes, %0d cycles",frame,checked,dut.cycles);
            wr(16,3); if(irq) $fatal(1,"IRQ W1C");
        end
        $display("TEST PASSED"); $finish;
    end
    initial begin #2000000000; $fatal(1,"global timeout"); end
endmodule
