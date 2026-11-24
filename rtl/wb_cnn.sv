// Wishbone slave at region 5. Byte offsets follow docs/hardware-plan.md.
// Weights and configuration are compiled into ROM; writes are rejected.
// Load raw RGB in ascending, full-word order before START. No model load is needed.
module wb_cnn #(
    parameter QPARAM_FILE="qparams.mem", PIXEL_FILE="pixels.mem"
)(
    input wire i_clk,i_reset,i_wb_cyc,i_wb_stb,i_wb_we,
    input wire [31:0] i_wb_addr,i_wb_data,
    input wire [3:0] i_wb_sel,
    output reg o_wb_ack, output reg [31:0] o_wb_data,
    output wire o_irq
);
    reg [1:0] phase;
    reg [15:0] addr;
    reg [31:0] data,pa;
    reg [3:0] sel;
    reg we;
    reg read_image,read_param;
    reg model_valid,start,abort;
    reg [1:0] irq_enable,irq_status;
    reg [31:0] error_code,image_loaded;
    wire busy,done;
    wire [31:0] cycles;
    wire [1:0] class_id;
    wire signed [7:0] scores [0:3];
    wire [17:0] ia,ca;
    wire iw;
    wire [7:0] id,iq;
    wire [31:0] cq,unused,host_image,host_param;
    wire execute = phase==1 && i_wb_cyc;
    wire image_access = addr>=16'h1000 && addr<16'h4000;
    wire param_access = addr==16'h34 && pa<219592 && pa[1:0]==0;
    wire image_host_en = execute && image_access && !busy && !i_reset && addr[1:0]==0;
    wire param_host_en = execute && param_access && !busy && !i_reset;
    cnn_ram #(.WORDS(3072)) image_ram (
        .clk(i_clk),.host_en(image_host_en),.host_we(we ? sel : 4'b0),
        .host_addr({2'b0,addr}-18'h1000),.host_data(data),.host_q(host_image),
        .core_addr(ia),.core_we(iw),.core_data(id),.core_word(unused),.core_q(iq));
    cnn_weights parameter_rom (
        .clk(i_clk),.host_en(param_host_en),.host_addr(pa[17:0]),.host_q(host_param),
        .core_addr(ca),.core_word(cq));
    cnn_core #(.QPARAM_FILE(QPARAM_FILE),.PIXEL_FILE(PIXEL_FILE)) core (
        .clk(i_clk),.reset(i_reset),.start(start),.abort(abort),.busy(busy),.done(done),
        .cycles(cycles),.class_id(class_id),.scores(scores),
        .image_addr(ia),.image_we(iw),.image_data(id),.image_q(iq),.param_addr(ca),.param_q(cq),
        .trace_valid(),.trace_layer(),.trace_addr(),.trace_data());
    assign o_irq = |(irq_enable & irq_status);
    task automatic fault(input [31:0] code);
        begin error_code<=code; irq_status[1]<=1; end
    endtask
    always @(posedge i_clk) begin
        o_wb_ack<=0; start<=0; abort<=0;
        if(i_reset) begin
            phase<=0; o_wb_data<=0; model_valid<=1; irq_enable<=0; irq_status<=0;
            error_code<=0; pa<=0; image_loaded<=0;
        end else begin
            if(!i_wb_cyc) phase<=0;
            else case(phase)
                0: if(i_wb_stb) begin addr<=i_wb_addr[15:0]; data<=i_wb_data; sel<=i_wb_sel; we<=i_wb_we; phase<=1; end
                1: begin
                    phase<=2;
                    read_image<=image_access && !busy;
                    read_param<=param_access && !busy;
                    if(!we && param_access && busy) fault(3);
                    if(addr[1:0]!=0) fault(1);
                    else if(image_access) begin
                        if(busy) fault(3);
                        else if(we) begin
                            if(addr-16'h1000==image_loaded && sel==15) image_loaded<=image_loaded+4;
                            else image_loaded<=0;
                        end
                    end else if(we && sel!=0) case(addr)
                        16'h04: if(!sel[0]) begin end
                            else if(data==2) begin abort<=1; irq_status[0]<=0; image_loaded<=0; end
                            else if(busy || start) fault(3);
                            else if(data==4) model_valid<=1; // built-in profile is already committed
                            else if(data==1) begin
                                if(!model_valid) fault(4);
                                else if(irq_status!=0 || image_loaded!=12288) fault(5);
                                else begin start<=1; image_loaded<=0; end
                            end else fault(2);
                        16'h0c: if(sel[0]) irq_enable<=data[1:0];
                        16'h10: if(sel[0]) begin
                            irq_status<=irq_status & ~data[1:0];
                            if(data[1]) error_code<=0;
                        end
                        16'h30: begin
                            for(integer b=0;b<4;b=b+1) if(sel[b]) pa[b*8 +: 8]<=data[b*8 +: 8];
                        end
                        16'h34: fault(7); // immutable model ROM
                        default: fault(1);
                    endcase
                    else if(!we && !(image_access || param_access || addr==0 || addr==8 || addr==12 || addr==16 ||
                        addr==20 || addr==24 || addr==28 || addr==32 || addr==36 || addr==40 || addr==44 || addr==48 ||
                        (addr>=256 && addr<=268))) fault(1);
                    if(!we && param_access && !busy) pa<=pa+4;
                end
                2: begin
                    phase<=3; o_wb_ack<=1; o_wb_data<=0;
                    if(!we && addr[1:0]==0) begin
                        if(read_image) o_wb_data<=host_image;
                        else if(read_param) o_wb_data<=host_param;
                        else case(addr)
                            0: o_wb_data<=32'h434e4e33;
                            8: o_wb_data<={28'b0,model_valid,irq_status[1],irq_status[0],busy};
                            12: o_wb_data<={30'b0,irq_enable};
                            16: o_wb_data<={30'b0,irq_status};
                            20: o_wb_data<={30'b0,class_id};
                            24: o_wb_data<=cycles;
                            28: o_wb_data<=error_code;
                            32: o_wb_data<=219592;
                            36: o_wb_data<=12288;
                            40: o_wb_data<=4;
                            44: o_wb_data<=0; // configuration is fixed in this bitstream
                            48: o_wb_data<=pa;
                            256,260,264,268: o_wb_data<={{24{scores[(addr-256)>>2][7]}},scores[(addr-256)>>2]};
                            default: o_wb_data<=0;
                        endcase
                    end
                end
                3: if(!i_wb_stb) phase<=0;
            endcase
            // New completion wins over W1C on this edge.
            if(done) irq_status[0]<=1;
        end
    end
endmodule
