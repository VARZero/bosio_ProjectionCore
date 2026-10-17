#include "projector.hpp"
#include <ap_int.h>

static uint64_t magnitude(int32_t v) {
    return v < 0 ? uint64_t(-int64_t(v)) : uint64_t(v);
}

static uint8_t channel(uint32_t p00, uint32_t p10, uint32_t p01,
                       uint32_t p11, unsigned shift, unsigned ax, unsigned ay) {
    uint32_t a = ((p00 >> shift) & 255) * (256-ax)
               + ((p10 >> shift) & 255) * ax;
    uint32_t b = ((p01 >> shift) & 255) * (256-ax)
               + ((p11 >> shift) & 255) * ax;
    return (a * (256-ay) + b * ay + 32768) >> 16;
}

extern "C" uint32_t bosio_projector(const uint32_t *texture,
    const uint32_t *descriptors, uint32_t *scene, uint32_t texture_words,
    uint32_t descriptor_words, uint32_t scene_words, uint32_t job_count,
    uint32_t &written_cells) {
#pragma HLS INTERFACE m_axi port=texture offset=slave bundle=gmem depth=65536 max_read_burst_length=16 max_write_burst_length=16 num_read_outstanding=2 num_write_outstanding=2
#pragma HLS INTERFACE m_axi port=descriptors offset=slave bundle=gmem depth=147456
#pragma HLS INTERFACE m_axi port=scene offset=slave bundle=gmem depth=1080320
#pragma HLS INTERFACE s_axilite port=texture bundle=control
#pragma HLS INTERFACE s_axilite port=descriptors bundle=control
#pragma HLS INTERFACE s_axilite port=scene bundle=control
#pragma HLS INTERFACE s_axilite port=texture_words bundle=control
#pragma HLS INTERFACE s_axilite port=descriptor_words bundle=control
#pragma HLS INTERFACE s_axilite port=scene_words bundle=control
#pragma HLS INTERFACE s_axilite port=job_count bundle=control
#pragma HLS INTERFACE s_axilite port=written_cells bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    // Correctness-first single processing lane. No claim of one cell/cycle.
    written_cells=0;
    if (job_count>MAX_JOBS || uint64_t(job_count)*DESC_WORDS>descriptor_words)
        return 0x10000000;
    uint32_t tile[TILE_WORDS];
#pragma HLS BIND_STORAGE variable=tile type=ram_2p impl=bram
jobs: for (uint32_t job=0; job<job_count; ++job) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4220
        uint32_t desc[DESC_WORDS];
load_descriptor: for (unsigned i=0;i<DESC_WORDS;++i) {
#pragma HLS PIPELINE II=1
            desc[i]=descriptors[job*DESC_WORDS+i];
        }
        uint32_t src=desc[0], dst=desc[1], width=desc[2], height=desc[3];
        uint32_t stride=desc[4], flags=desc[5];
        if (!width || !height || width>2048 || height>2048 || stride<width ||
            uint64_t(src)+uint64_t(height-1)*stride+width>texture_words ||
            uint64_t(dst)+TILE_WORDS>scene_words || (flags & ~0xff01U))
            return 0x20000000 | job;
        int32_t c[3][4];
#pragma HLS ARRAY_PARTITION variable=c complete
        for (unsigned k=0;k<3;++k) {
            uint64_t bound=0;
            for (unsigned i=0;i<4;++i) {
                c[k][i]=int32_t(desc[6+k*4+i]);
                bound+=magnitude(c[k][i])*((i==1 || i==2)?31:1);
            }
            // Conservative rectangular bound also covers every triangular cell.
            if (bound>0x3fffffffULL) return 0x30000000 | job;
        }
        uint32_t bg=(flags>>8)&255; bg*=0x01010101U;
load_tile: for (unsigned i=0;i<TILE_WORDS;++i) {
#pragma HLS PIPELINE II=1
            tile[i]=(flags&1)?bg:scene[dst+i];
        }
        int32_t ru=c[0][0],rv=c[1][0],rd=c[2][0];
        unsigned index=0;
rows: for (unsigned row=0;row<32;++row) {
            int32_t cu=ru,cv=rv,cd=rd;
columns: for (unsigned col=0;col<=row;++col) {
orientations: for (unsigned inv=0;inv<(col<row?2U:1U);++inv) {
                    int32_t u=cu+(inv?c[0][3]:0);
                    int32_t v=cv+(inv?c[1][3]:0);
                    int32_t d=cd+(inv?c[2][3]:0);
                    bool inside=d>0 && u>=0 && v>=0;
                    if (inside) inside=uint64_t(u)<=uint64_t(d)*(width-1)
                                     && uint64_t(v)<=uint64_t(d)*(height-1);
                    if (inside) {
                        ap_uint<40> un=ap_uint<40>(uint32_t(u))<<8;
                        ap_uint<40> vn=ap_uint<40>(uint32_t(v))<<8;
                        uint32_t xq=(un/ap_uint<32>(uint32_t(d))).to_uint();
                        uint32_t yq=(vn/ap_uint<32>(uint32_t(d))).to_uint();
                        unsigned x=xq>>8,y=yq>>8,ax=xq&255,ay=yq&255;
                        unsigned x1=x+1<width?x+1:x;
                        unsigned y1=y+1<height?y+1:y;
                        uint32_t p00=texture[src+y*stride+x];
                        uint32_t p10=texture[src+y*stride+x1];
                        uint32_t p01=texture[src+y1*stride+x];
                        uint32_t p11=texture[src+y1*stride+x1];
                        unsigned r=channel(p00,p10,p01,p11,16,ax,ay);
                        unsigned g=channel(p00,p10,p01,p11,8,ax,ay);
                        unsigned b=channel(p00,p10,p01,p11,0,ax,ay);
                        unsigned color=(r&224)|((g>>3)&28)|(b>>6);
                        unsigned shift=(index&3)*8;
                        tile[index>>2]=(tile[index>>2]&~(255U<<shift))|(color<<shift);
                        ++written_cells;
                    }
                    ++index;
                }
                cu+=c[0][2];cv+=c[1][2];cd+=c[2][2];
            }
            ru+=c[0][1];rv+=c[1][1];rd+=c[2][1];
        }
store_tile: for (unsigned i=0;i<TILE_WORDS;++i) {
#pragma HLS PIPELINE II=1
            scene[dst+i]=tile[i];
        }
    }
    return 0;
}
