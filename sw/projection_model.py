"""CPU descriptor preparation and an integer reference for M=32 projection.

CPU: geometry, window transforms, conservative tile culling, ordered job lists.
FPGA: per-cell perspective divide, bilinear RGB sampling, packed RGB332 DDR writes.
No full-sphere direction-vector table is allocated.
"""
import math
from functools import lru_cache
import numpy as np
from geometry_reference import VERTICES, triangle_centers

M=32
DESC_WORDS=18
TILE_WORDS=256

def tile_barycentric_vertices(tile):
    if not 0<=tile<211:raise ValueError('invalid tile')
    region=tile//49 if tile<147 else 3
    n=8 if region==3 else 7
    local=tile-(147 if region==3 else region*49)
    row=math.isqrt(local);col=(local-row*row)//2;inv=(local-row*row)&1
    origin=np.array([n-1-row,row-col-inv,col],dtype=float)
    bary=(origin+(1-np.eye(3) if inv else np.eye(3)))/n
    if region==3:
        bary=(1-bary)/2
    else:
        bary/=2;bary[:,region]+=.5
    return bary

def tile_vertices(face, tile):
    if not 0<=face<20:raise ValueError('invalid face')
    return tile_barycentric_vertices(tile)@VERTICES[face]

@lru_cache(maxsize=1)
def all_tile_vertices():
    # Only 4,220 triangles x 3 vertices, about 297 KiB; no 4.3M-cell ray table.
    bary=np.array([tile_barycentric_vertices(t) for t in range(211)])
    return np.einsum('tvc,fcd->ftvd',bary,VERTICES).reshape(4220,3,3)

def window_axes(yaw=0,pitch=0,roll=0):
    az,el,rl=np.radians([yaw,pitch,roll])
    c=np.array([math.cos(el)*math.sin(az),math.sin(el),-math.cos(el)*math.cos(az)])
    r=np.array([math.cos(az),0,math.sin(az)]);u=np.cross(r,c)
    return c,r*math.cos(rl)+u*math.sin(rl),u*math.cos(rl)-r*math.sin(rl)

def tile_descriptor(face,tile,width,height,stride=None,src=0,dst=0,
                    yaw=0,pitch=0,roll=0,fov_h=48,fov_v=36,clear=True,background=0):
    if not (1<=width<=2048 and 1<=height<=2048 and 0<fov_h<160 and 0<fov_v<160):
        raise ValueError('invalid dimensions/FOV')
    q=tile_vertices(face,tile)
    c,r,u=window_axes(yaw,pitch,roll)
    den=q@c
    un=(q@r/(2*math.tan(math.radians(fov_h/2)))+den/2)*(width-1)
    vn=(-q@u/(2*math.tan(math.radians(fov_v/2)))+den/2)*(height-1)
    coeff=[]
    for a,b,c0 in (un,vn,den):
        coeff.append([94*a+b+c0,3*(b-a),3*(c0-b),a-2*b+c0])
    coeff=np.array(coeff)
    bound=np.max(np.abs(coeff)@np.array([1,31,31,1]))
    scale=min(2.**22,2.**math.floor(math.log2((2.**29)/max(bound,1e-30))))
    fixed=np.rint(coeff*scale).astype(np.int64)
    assert np.max(np.abs(fixed)@np.array([1,31,31,1]))<=0x3fffffff
    desc=np.zeros(DESC_WORDS,dtype=np.uint32)
    desc[:6]=[src,dst,width,height,stride or width,int(clear)|(int(background)<<8)]
    desc[6:]=fixed.astype(np.int32).view(np.uint32).reshape(-1)
    # Conservative culling uses the QUANTIZED coefficients consumed by hardware.
    # This avoids losing boundary cells due to float/fixed rounding differences.
    # r>=col>=0, r<=31: a linear function reaches its maximum at a triangle
    # corner. Allow either orientation at every corner (a safe overestimate).
    fu,fv,fd=fixed
    planes=[fd,fu,fv,fd*(width-1)-fu,fd*(height-1)-fv]
    upper=[max(p[0],p[0]+31*p[1],p[0]+31*(p[1]+p[2]))+max(0,p[3]) for p in planes]
    visible=upper[0]>0 and all(x>=0 for x in upper[1:])
    return desc,visible

def cell_coordinates(desc):
    coeff=np.asarray(desc[6:],dtype=np.uint32).view(np.int32).astype(np.int64).reshape(3,4)
    ids=np.arange(1024);row=np.floor(np.sqrt(ids)).astype(np.int64)
    col=(ids-row*row)//2;inv=(ids-row*row)&1
    return coeff@np.array([np.ones(1024,dtype=np.int64),row,col,inv])

def prepare_window(width,height,stride=None,src=0,dst=0,yaw=0,pitch=0,roll=0,
                   fov_h=48,fov_v=36,clear=True,background=0):
    """Batch CPU preparation: one descriptor per tile, and a conservative mask."""
    if not (1<=width<=2048 and 1<=height<=2048 and 0<fov_h<160 and 0<fov_v<160):
        raise ValueError('invalid dimensions/FOV')
    q=all_tile_vertices();c,r,u=window_axes(yaw,pitch,roll)
    den=q@c
    un=(q@r/(2*math.tan(math.radians(fov_h/2)))+den/2)*(width-1)
    vn=(-q@u/(2*math.tan(math.radians(fov_v/2)))+den/2)*(height-1)
    values=np.stack([un,vn,den],axis=1)
    a,b,c0=values[:,:,0],values[:,:,1],values[:,:,2]
    coeff=np.stack([94*a+b+c0,3*(b-a),3*(c0-b),a-2*b+c0],axis=-1)
    bound=np.max(np.sum(np.abs(coeff)*np.array([1,31,31,1]),axis=-1),axis=-1)
    scale=np.exp2(np.minimum(22,np.floor(np.log2(2.**29/np.maximum(bound,1e-30)))))
    fixed=np.rint(coeff*scale[:,None,None]).astype(np.int64)
    assert np.max(np.sum(np.abs(fixed)*np.array([1,31,31,1]),axis=-1))<=0x3fffffff
    desc=np.zeros((4220,DESC_WORDS),dtype=np.uint32)
    desc[:,:6]=[src,dst,width,height,stride or width,int(clear)|(int(background)<<8)]
    desc[:,1]+=np.arange(4220,dtype=np.uint32)*256
    desc[:,6:]=fixed.astype(np.int32).view(np.uint32).reshape(4220,12)
    fu,fv,fd=fixed[:,0],fixed[:,1],fixed[:,2]
    p=np.stack([fd,fu,fv,fd*(width-1)-fu,fd*(height-1)-fv],axis=1)
    upper=np.maximum.reduce([p[:,:,0],p[:,:,0]+31*p[:,:,1],p[:,:,0]+31*(p[:,:,1]+p[:,:,2])])+np.maximum(0,p[:,:,3])
    visible=(upper[:,0]>0)&np.all(upper[:,1:]>=0,axis=1)
    return desc,visible

def reference_job(texture,scene,desc):
    """Exact fixed-point model, vectorized independently of the HLS loops."""
    src,dst,w,h,stride,flags=map(int,desc[:6])
    uv=cell_coordinates(desc);u,v,d=uv
    mask=(d>0)&(u>=0)&(v>=0)&(u<=d*(w-1))&(v<=d*(h-1))
    out=scene[dst:dst+256].view(np.uint8).copy()
    if flags&1:out[:]=(flags>>8)&255
    valid=np.flatnonzero(mask)
    if valid.size:
        xq=(u[mask]*256)//d[mask];yq=(v[mask]*256)//d[mask]
        x=xq//256;y=yq//256;ax=xq%256;ay=yq%256
        x1=np.minimum(w-1,x+1);y1=np.minimum(h-1,y+1)
        pixels=[texture[src+y*stride+x],texture[src+y*stride+x1],
                texture[src+y1*stride+x],texture[src+y1*stride+x1]]
        channels=[]
        for shift in (16,8,0):
            a,b,c,e=[((p.astype(np.int64)>>shift)&255) for p in pixels]
            value=((a*(256-ax)+b*ax)*(256-ay)+(c*(256-ax)+e*ax)*ay+32768)>>16
            channels.append(value)
        r,g,b=channels
        out[mask]=((r&224)|((g>>3)&28)|(b>>6)).astype(np.uint8)
    scene[dst:dst+256]=out.view(np.uint32)
    return int(valid.size)

def floating_coordinates(face,tile,width,height,yaw=0,pitch=0,roll=0,fov_h=48,fov_v=36):
    # Independently form every centroid in 3D, then use the original dot/divide mapping.
    q=triangle_centers(32)@tile_vertices(face,tile)
    c,r,u=window_axes(yaw,pitch,roll);d=q@c
    with np.errstate(divide='ignore',invalid='ignore'):
        x=((q@r)/d/math.tan(math.radians(fov_h/2))+1)*.5*(width-1)
        y=(1-(q@u)/d/math.tan(math.radians(fov_v/2)))*.5*(height-1)
    return x,y,d
