"""Canonical icosahedron, 211 triangular tiles/face, M*M triangular cells/tile.

Cell storage is row-major TRIANGLE order: row**2 + 2*column + inverted.
Neither level uses a square UV texture. Camera coefficients are Q24, evaluated
at pixel centers by the FPGA DDA. No screen image is rendered by the driver.
"""
import math
import numpy as np

TILES = 211
Q = 1 << 24
MAX_CACHE_BYTES = 20 * TILES * 32 * 32  # BS25 full-sphere DDR capacity.

def vertices():
    y=1/math.sqrt(5);r=2/math.sqrt(5)
    def ring(e):
        return [np.array([r*math.sin(math.radians(72*i+e)), y if e==18 else -y,
                          -r*math.cos(math.radians(72*i+e))]) for i in range(5)]
    u,l=ring(18),ring(-18);n=np.array([0.,1.,0.]);s=-n
    fs={}
    for i in range(5):
        j=(i+1)%5
        fs[1+2*i]=[u[i],l[i],l[j]]
        fs[2+2*i]=[s,l[i],l[j]]
        fs[11+2*i]=[l[j],u[i],u[j]]
        fs[14+2*i if i<4 else 12]=[n,u[i],u[j]]
    return np.asarray([fs[i] for i in range(1,21)])

VERTICES=vertices()
INVERSES=np.linalg.inv(VERTICES.transpose(0,2,1))

def triangle_centers(n):
    """n*n centroid barycentrics, indexed row**2+2*column+orientation."""
    out=[]
    for row in range(n):
        a=n-1-row
        for col in range(row+1):
            out.append([(a+1/3)/n,(row-col+1/3)/n,(col+1/3)/n])
            if col<row:
                out.append([(a+2/3)/n,(row-col-1+2/3)/n,(col+2/3)/n])
    return np.asarray(out)

def cell_barycentrics(m=16):
    if m not in (8,16,32):raise ValueError('M must be 8,16,32')
    cells=triangle_centers(m)
    result=[]
    for region in range(4):
        n=8 if region==3 else 7
        for row in range(n):
            a=n-1-row
            for col in range(row+1):
                for inv in range(2 if col<row else 1):
                    origin=np.array([a,row-col-inv,col])
                    local=(origin+(1-cells if inv else cells))/n
                    if region==3:parent=(1-local)/2
                    else:
                        parent=local/2;parent[:,region]+=.5
                    result.append(parent)
    return np.asarray(result)

def cell_rays(m=16):
    p=np.einsum('tmv,fvc->ftmc',cell_barycentrics(m),VERTICES)
    return p/np.linalg.norm(p,axis=-1,keepdims=True)

def camera_coefficients(yaw=0,pitch=0,roll=0,fov_h=48,fov_v=36,width=1280,height=720):
    if not(0<fov_h<160 and 0<fov_v<160):raise ValueError('FOV must be between 0 and 160 degrees')
    az,el,rl=np.radians([yaw,pitch,roll])
    c=np.array([math.cos(el)*math.sin(az),math.sin(el),-math.cos(el)*math.cos(az)])
    r=np.array([math.cos(az),0,math.sin(az)]);u=np.cross(r,c)
    rr=r*math.cos(rl)+u*math.sin(rl);uu=u*math.cos(rl)-r*math.sin(rl)
    dx=rr*(2*math.tan(math.radians(fov_h/2))/width)
    dy=-uu*(2*math.tan(math.radians(fov_v/2))/height)
    start=c-dx*(width-1)/2-dy*(height-1)/2
    coeff=np.stack([INVERSES@start,INVERSES@dx,INVERSES@dy],axis=-1)
    return np.rint(coeff*Q).astype(np.int32).reshape(60,3)

def locate(bary,m=16):
    """Floating reference mapping. Input shape (...,3), nonnegative sum=1."""
    b=np.asarray(bary);reg=np.where(b[...,0]>=.5,0,np.where(b[...,1]>=.5,1,np.where(b[...,2]>=.5,2,3)))
    local=np.where((reg==3)[...,None],1-2*b,2*b-np.eye(3)[np.minimum(reg,2)])
    n=np.where(reg==3,8,7)
    scaled=np.clip(local*n[...,None],0,n[...,None]-1e-8)
    f=np.floor(scaled).astype(int);inv=f.sum(axis=-1)==n-2
    row=n-1-f[...,0];tile=np.where(reg==3,147,reg*49)+row*row+2*f[...,2]+inv
    frac=scaled-f;frac=np.where(inv[...,None],1-frac,frac)
    g=np.floor(np.clip(frac*m,0,m-1e-8)).astype(int)
    rin=m-1-g[...,0];iin=g.sum(axis=-1)==m-2
    cell=rin*rin+2*g[...,2]+iin
    return tile,cell

def pack_scene(rgb,m=16):
    """RGB[20,211,M*M,3] -> palette + directory + dense active triangle cells.

    Black cells index 0, invalid directory 0xffffffff. Palette RGB332 preserves
    primary colors; 256 entries can later be supplied by a custom quantizer.
    DMA words: 256 palette, 4220 directory, then packed 8-bit cell indices.
    """
    rgb=np.asarray(rgb,dtype=np.uint8)
    if rgb.shape!=(20,TILES,m*m,3):raise ValueError('Invalid triangular cell array shape')
    idx=(rgb[...,0]&224)|((rgb[...,1]>>3)&28)|(rgb[...,2]>>6)
    active=np.any(idx!=0,axis=-1).reshape(-1)
    data=idx.reshape(20*TILES,m*m)[active].reshape(-1)
    if len(data)>MAX_CACHE_BYTES:raise ValueError(f'Active tiles exceed cache: {len(data)} > {MAX_CACHE_BYTES} bytes')
    directory=np.full(20*TILES,0xffffffff,dtype=np.uint32)
    directory[active]=np.arange(active.sum(),dtype=np.uint32)*(m*m)
    k=np.arange(256,dtype=np.uint32)
    red=((k>>5)*255//7);green=(((k>>2)&7)*255//7);blue=((k&3)*255//3)
    level = k >> 5
    neutral = (((k >> 2) & 7) == level) & ((k & 3) == ((level * 255 // 7) >> 6))
    gray = level * 255 // 7
    red = np.where(neutral, gray, red)
    green = np.where(neutral, gray, green)
    blue = np.where(neutral, gray, blue)
    palette=(red<<16)|(blue<<8)|green
    data=np.pad(data,(0,(-len(data))%64))
    words=np.concatenate([palette,directory,data.view('<u4')])
    words=np.pad(words,(0,(-len(words))%16))
    return words.astype('<u4'),int(active.sum())
