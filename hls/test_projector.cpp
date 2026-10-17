#include "projector.hpp"
#include <vector>
#include <cstdio>
#include <cassert>
#include <cstring>

int main() {
    std::vector<uint32_t> texture(65536,0xdeadbeef), scene(1080320,0xa5a5a5a5);
    for(unsigned y=0;y<64;++y) for(unsigned x=0;x<64;++x)
        texture[4+y*67+x]=((x*4)<<16)|((y*4)<<8)|((x^y)*4);
    std::vector<uint32_t> storage(147456,0); uint32_t *d=storage.data();
    d[0]=4;d[1]=2;d[2]=64;d[3]=64;d[4]=67;d[5]=1|(0x39<<8);
    // x = column + inv/2; y = row + inv/2. Fractional coordinates exercise bilinear.
    int32_t coeff[12]={0,0,256,128,0,256,0,128,256,0,0,0};
    std::memcpy(d+6,coeff,sizeof(coeff));
    uint32_t written=0;
    assert(bosio_projector(texture.data(),d,scene.data(),texture.size(),18,260,1,written)==0);
    assert(written==1024);
    const uint8_t *out=reinterpret_cast<const uint8_t*>(scene.data()+2);
    unsigned n=0;
    for(unsigned row=0;row<32;++row) for(unsigned col=0;col<=row;++col)
        for(unsigned inv=0;inv<(col<row?2U:1U);++inv) {
            unsigned r=col*4+inv*2,g=row*4+inv*2;
            unsigned b=inv?(((col^row)+((col+1)^row)+(col^(row+1))+((col+1)^(row+1)))):(col^row)*4;
            unsigned expected=(r&224)|((g>>3)&28)|(b>>6);
            assert(out[n++]==expected);
        }
    assert(scene[0]==0xa5a5a5a5 && scene[1]==0xa5a5a5a5 && scene[258]==0xa5a5a5a5 && scene[259]==0xa5a5a5a5);
    // Zero denominator clears to background; no texture reads should be needed.
    d[14]=0;
    assert(bosio_projector(texture.data(),d,scene.data(),texture.size(),18,260,1,written)==0);
    assert(written==0 && scene[2]==0x39393939);
    // Preserve outside cells for ordered opaque overlay jobs.
    d[5]=0; scene[2]=0x12345678;
    assert(bosio_projector(texture.data(),d,scene.data(),texture.size(),18,260,1,written)==0);
    assert(scene[2]==0x12345678);
    d[1]=260-255;
    assert((bosio_projector(texture.data(),d,scene.data(),texture.size(),18,260,1,written)&0xf0000000)==0x20000000);
    assert(bosio_projector(texture.data(),d,scene.data(),texture.size(),17,260,1,written)==0x10000000);
    std::puts("PASS: bilinear, triangular order, clear/preserve, guards, invalid lengths");
}
