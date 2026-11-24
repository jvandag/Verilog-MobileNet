// Fixed-profile sequential MobileNetV1-0.25 engine, 64x64 RGB.
// All memories have registered reads. One 9x8 MAC, no floating point.
module cnn_core #(
    parameter QPARAM_FILE="qparams.mem", PIXEL_FILE="pixels.mem"
)(
    input wire clk, reset, start, abort,
    output reg busy, done,
    output reg [31:0] cycles,
    output reg [1:0] class_id,
    output reg signed [7:0] scores [0:3],
    output reg [17:0] image_addr, output wire image_we,
    output wire [7:0] image_data, input wire [7:0] image_q,
    output reg [17:0] param_addr, input wire [31:0] param_q,
    output wire trace_valid, output wire [4:0] trace_layer,
    output wire [14:0] trace_addr, output wire [7:0] trace_data
);
    localparam IDLE=0, LOAD=1, INIT=2, BIAS=3, BWAIT=4, BGET=5,
        ISSUE=6, WAIT_READ=7, MAC=8, STORE=9, NEXT=10,
        PIXEL=11, PWAIT=12, PLUT=13, PSTORE=14, FINISH=15,
        RQMUL=16,RQCOMB=17,RQADD=18,RQHIGH=19,RQROUND=20,ACCUM=21,MULTIPLY=22,
        RQBIAS=23,RQCLAMP=24;
    reg [4:0] state;
    reg [4:0] layer;
    reg [2:0] kind;
    reg [1:0] src_bank,dst_bank,kh,kw,stride,pady,padx;
    reg [6:0] ih,iw,oh,ow;
    reg [8:0] ic,oc;
    reg signed [8:0] iz,oz,clo,chi;
    reg [17:0] wbase,bbase;
    reg [11:0] qbase;
    reg [3:0] lgiw,lgic,lgow,lgoc;
    reg [5:0] y,x;
    reg [7:0] ch,ci;
    reg [1:0] ky,kx;
    reg [13:0] idx;
    integer sy,sx,source_address,weight_address,output_address;
    reg padding;
    reg [17:0] weight_pointer,source_pointer;
    reg [9:0] channel_rows;
    reg [11:0] channel_kernel,weight_offset;
    reg [17:0] weight_base;
    reg [3:0] kernel_index;
    (* max_fanout=64 *) reg [14:0] output_pointer;
    reg pad_q;
    reg signed [31:0] acc;
    reg signed [16:0] mac_product;
    reg signed [8:0] activation_reg;
    reg signed [7:0] weight_reg;
    reg signed [7:0] stage_scores [0:3];
    (* ram_style="block" *) reg signed [7:0] bank_a [0:16383];
    (* ram_style="block" *) reg signed [7:0] bank_b [0:8191];
    reg signed [7:0] aq,bq;
    wire signed [7:0] source_q = src_bank==0 ? $signed(image_q) : src_bank==1 ? aq : bq;
    // LUT ROM preserves scarce block RAM for weights and MicroBlaze memory.
    (* rom_style="distributed" *) reg [39:0] qparams [0:2739];
    reg [39:0] qp;
    reg [7:0] pixels [0:255];
    reg [7:0] pixel;
    reg signed [33:0] p00,p01,p10,p11;
    reg signed [63:0] product,rounded;
    reg signed [31:0] high,scaled;
    reg signed [31:0] biased;
    reg [9:0] pool_round;
    reg signed [7:0] final_byte;
    reg [5:0] shift_count;
    reg signed [31:0] round_mask,round_threshold;
    initial begin
        $readmemh(QPARAM_FILE,qparams);
        $readmemh(PIXEL_FILE,pixels);
    end
    // Defined for the compiler-proven range: nonpositive shift, no int32 overflow.
    function automatic signed [31:0] rescale(input signed [31:0] a, m, input signed [7:0] shift);
        reg signed [63:0] prod, adjusted;
        reg signed [31:0] high, mask, threshold;
        integer e;
        begin
            prod = a * m;
            e = 31-shift;
            adjusted = prod + (64'sd1 << (e-1));
            rescale = adjusted >>> e;
            if(kind==2) begin
                adjusted = prod + (prod>=0 ? 64'sd1073741824 : -64'sd1073741823);
                high = adjusted>=0 ? adjusted >>> 31 : -((-adjusted) >>> 31);
                e = -shift;
                mask = (32'd1 << e)-1;
                threshold = (mask >> 1) + (high<0);
                rescale = (high >>> e) + (((high & mask)>threshold) ? 32'sd1 : 32'sd0);
            end
        end
    endfunction
    reg signed [31:0] result;
    always @* begin
        sy = (stride==2 ? {y,1'b0} : y)+ky-pady;
        sx = (stride==2 ? {x,1'b0} : x)+kx-padx;
        padding = sy<0 || sx<0 || sy>=ih || sx>=iw;
        source_address = ((((sy << lgiw)+sx) << lgic)) + (kind==2 ? ch : ci);
        if(ic==3) source_address = ((sy << lgiw)+sx)*3+ci;
        if (kind==3) source_address = (ci << lgic)+ch;
        if (kind==4) source_address = ci;
        kernel_index = (kw==3 ? ky*3 : ky)+kx;
        if(kind==2) weight_offset=kernel_index << lgoc;
        else if(ic==3) weight_offset=kernel_index*3+ci;
        // For power-of-two channel counts, the channel field and kernel field
        // do not overlap; concatenation avoids another carry chain per MAC.
        else weight_offset=(kernel_index << lgic) | ci;
        weight_address=weight_base+weight_offset;
        output_address = (((y << lgow)+x) << lgoc)+ch;
        param_addr = state==BIAS || state==BWAIT || state==BGET ? bbase+ch*4 : weight_pointer;
        image_addr = layer==0 ? idx : source_pointer;
        result = final_byte;
    end
    assign image_we = busy && state==PSTORE;
    assign image_data = pixel;
    assign trace_valid = busy && (state==STORE || state==PSTORE);
    assign trace_layer = layer;
    assign trace_addr = layer==0 ? idx : output_pointer;
    assign trace_data = layer==0 ? pixel : result[7:0];
    always @(posedge clk) begin
        aq <= bank_a[source_pointer];
        bq <= bank_b[source_pointer];
        qp <= qparams[qbase+ch];
        if (busy && state==STORE) begin
            if (dst_bank==1) bank_a[output_pointer] <= result[7:0];
            if (dst_bank==2) bank_b[output_pointer] <= result[7:0];
        end
    end
    reg signed [8:0] centered;
    reg signed [7:0] weight;
    always @* begin
        centered = $signed(source_q)-iz;
        weight = $signed(param_q[weight_pointer[1:0]*8 +: 8]);
    end
    always @(posedge clk) begin
        done <= 0;
        if (reset) begin
            state<=IDLE; busy<=0; cycles<=0; layer<=0; idx<=0; ch<=0;
            qbase<=0; class_id<=0; output_pointer<=0;
            for (integer i=0;i<4;i=i+1) scores[i]<=0;
        end else if (abort) begin
            busy<=0; state<=IDLE;
        end else begin
            if (busy) cycles<=cycles+1;
            case(state)
                IDLE: if(start) begin busy<=1; cycles<=0; layer<=0; state<=LOAD; end
                LOAD: begin
                    `include "cnn_profile.vh"
                    state<=INIT;
                end
                INIT: begin
                    y<=0; x<=0; ch<=0; ky<=0; kx<=0; ci<=0; idx<=0; acc<=0;
                    state<=layer==0 ? PIXEL : BIAS;
                end
                PIXEL: state<=PWAIT;
                PWAIT: state<=PLUT;
                PLUT: begin pixel<=pixels[image_q]; state<=PSTORE; end
                PSTORE: if(idx==12287) begin layer<=1; state<=LOAD; end
                    else begin idx<=idx+1; state<=PIXEL; end
                BIAS: begin
                    output_pointer<=output_address;
                    channel_rows<=kh==3 ? ch*3 : ch;
                    state<=BWAIT;
                end
                BWAIT: begin
                    channel_kernel<=kw==3 ? channel_rows*3 : channel_rows;
                    state<=BGET;
                end
                BGET: begin
                    weight_base<=kind==2 ? wbase+ch : wbase+(ic==3 ? channel_kernel*3 : channel_kernel << lgic);
                    acc<=kind==3 ? 0 : $signed(param_q); ky<=0; kx<=0; ci<=0; state<=ISSUE;
                end
                ISSUE: begin
                    source_pointer<=padding && kind!=3 && kind!=4 ? 0 : source_address;
                    weight_pointer<=weight_address;
                    pad_q<=padding;
                    state<=WAIT_READ;
                end
                WAIT_READ: state<=MAC;
                MAC: begin
                    activation_reg <= kind==3 ? $signed(source_q)+9'sd128 : centered;
                    weight_reg <= kind==3 ? 8'sd1 : (!pad_q || kind==4) ? weight : 8'sd0;
                    state<=MULTIPLY;
                end
                MULTIPLY: begin
                    mac_product <= activation_reg*weight_reg;
                    state<=ACCUM;
                end
                ACCUM: begin
                    acc<=acc+mac_product;
                    if(ci+1 < (kind==2 ? 1 : kind==3 ? 4 : ic)) ci<=ci+1;
                    else begin
                        ci<=0;
                        if(kx+1<kw) kx<=kx+1;
                        else begin kx<=0; if(ky+1<kh) ky<=ky+1;
                            else begin ky<=0; state<=kind==3 ? RQBIAS : RQMUL; end
                        end
                    end
                    if (!(ci+1 >= (kind==2 ? 1 : kind==3 ? 4 : ic) && kx+1>=kw && ky+1>=kh)) state<=ISSUE;
                end
                RQMUL: begin
                    p00 <= $signed({1'b0,acc[15:0]}) * $signed({1'b0,qp[15:0]});
                    p01 <= $signed({1'b0,acc[15:0]}) * $signed(qp[31:16]);
                    p10 <= $signed(acc[31:16]) * $signed({1'b0,qp[15:0]});
                    p11 <= $signed(acc[31:16]) * $signed(qp[31:16]);
                    shift_count <= -$signed(qp[39:32]);
                    state<=RQCOMB;
                end
                RQCOMB: begin
                    product <= {{30{p00[33]}},p00} + ($signed({{30{p01[33]}},p01}) <<< 16) +
                        ($signed({{30{p10[33]}},p10}) <<< 16) + ($signed({{30{p11[33]}},p11}) <<< 32);
                    state<=RQADD;
                end
                RQADD: begin
                    if(kind==2) rounded<=product+(product>=0 ? 64'sd1073741824 : -64'sd1073741823);
                    else rounded<=product+(64'sd1 << (30+shift_count));
                    state<=RQHIGH;
                end
                RQHIGH: begin
                    if(kind==2) high<=rounded>=0 ? rounded >>> 31 : -((-rounded) >>> 31);
                    else high<=rounded >>> (31+shift_count);
                    round_mask<=(32'd1 << shift_count)-1;
                    state<=RQROUND;
                end
                RQROUND: begin
                    if(kind==2) scaled<=(high >>> shift_count) +
                        (((high & round_mask)>((round_mask >> 1)+(high<0 ? 32'sd1 : 32'sd0))) ? 32'sd1 : 32'sd0);
                    else scaled<=high;
                    state<=RQBIAS;
                end
                RQBIAS: begin
                    biased<=scaled+oz;
                    // MEAN uses double rounding on a proven 0..1020 sum.
                    pool_round<=acc[9:0]+10'd3;
                    state<=RQCLAMP;
                end
                RQCLAMP: begin
                    if(kind==3) final_byte<=pool_round[9:2]-8'd128;
                    else if(biased<clo) final_byte<=clo;
                    else if(biased>chi) final_byte<=chi;
                    else final_byte<=biased[7:0];
                    state<=STORE;
                end
                STORE: begin
                    if(layer==29) stage_scores[ch]<=result[7:0];
                    state<=NEXT;
                end
                NEXT: begin
                    if(ch+1<oc) begin ch<=ch+1; state<=BIAS; end
                    else begin
                        ch<=0;
                        if(x+1<ow) begin x<=x+1; state<=BIAS; end
                        else begin x<=0;
                            if(y+1<oh) begin y<=y+1; state<=BIAS; end
                            else if(layer==29) state<=FINISH;
                            else begin layer<=layer+1; state<=LOAD; end
                        end
                    end
                end
                FINISH: begin
                    for(integer i=0;i<4;i=i+1) scores[i]<=stage_scores[i];
                    if(stage_scores[0]>=stage_scores[1] && stage_scores[0]>=stage_scores[2] && stage_scores[0]>=stage_scores[3]) class_id<=0;
                    else if(stage_scores[1]>=stage_scores[2] && stage_scores[1]>=stage_scores[3]) class_id<=1;
                    else if(stage_scores[2]>=stage_scores[3]) class_id<=2;
                    else class_id<=3;
                    busy<=0; done<=1; state<=IDLE;
                end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
