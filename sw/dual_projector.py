"""PS scheduler for two independent DDR projection engines; no Overlay reload."""
import fcntl
import json
import time
from pathlib import Path
import numpy as np
from pynq import MMIO,PL,allocate
from pynq.pl_server import embedded_device
from partition_jobs import split_by_tile,DESC_WORDS

REG=json.loads(Path(__file__).with_name('registers.json').read_text())
CANARY=np.uint32(0xa5c35a3c)

class DualProjector:
    def __init__(self):
        self.lock=open('/tmp/bosio-projector.lock','w')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.ips=[];self.jobs=[];self.live=set()
        for index,address in enumerate((0x43c10000,0x43c20000)):
            assert PL.ip_dict['projector_'+str(index)]['phys_addr']==address
            ip=MMIO(address,0x10000)
            assert ip.read(0)&4, 'Projection engine is busy'
            self.ips.append(ip)
        try:
            for _ in self.ips:self.jobs.append(allocate(shape=(8192*DESC_WORDS,),dtype=np.uint32))
        except BaseException:
            self.close()
            raise

    def run(self,texture,descriptors,scene,workers=2,timeout=90,
            texture_words=None,scene_words=None,verify_descriptors=True):
        call_started=time.perf_counter()
        queues=split_by_tile(descriptors,workers)
        # Every group contains an entire tile, and cross-core tile ranges are
        # disjoint. No CPU accesses to these buffers are allowed until return.
        for buf in (texture,scene):
            if buf.dtype!=np.uint32 or not buf.flags.c_contiguous:
                raise ValueError('Buffers must be contiguous allocated uint32 arrays')
            if not 0<=int(buf.physical_address)<2**32:
                raise ValueError('Expected a 32-bit PYNQ allocation address')
        a=int(texture.physical_address);b=int(scene.physical_address)
        if a<b+scene.nbytes and b<a+texture.nbytes:
            raise ValueError('Texture and destination allocations overlap')
        texture_words=len(texture) if texture_words is None else int(texture_words)
        scene_words=len(scene) if scene_words is None else int(scene_words)
        if not 0<=texture_words<=len(texture) or not 0<=scene_words<=len(scene):
            raise ValueError('Bounds must stay within the owned allocations')
        texture[:texture_words].flush();scene[:scene_words].flush()
        for index,q in enumerate(queues):
            ip=self.ips[index];buf=self.jobs[index]
            assert ip.read(0)&4, 'Projection engine is busy'
            flat=q.reshape(-1)
            guard_end=len(buf) if verify_descriptors else min(len(buf),len(flat)+16)
            buf[len(flat):guard_end]=CANARY;buf[:len(flat)]=flat;buf[:guard_end].flush()
            for name,value in (('texture',texture),('descriptors',buf),('scene',scene)):
                ip.write(REG[name],int(value.physical_address));ip.write(REG[name]+4,0)
            for name,value in (('texture_words',texture_words),('descriptor_words',len(flat)),
                               ('scene_words',scene_words),('job_count',len(q))):
                ip.write(REG[name],value)
        starts=[];completed=[None]*workers
        started=time.perf_counter()
        prepare_ms=(started-call_started)*1000
        try:
            for index in range(workers):
                self.live.add(index)
                starts.append(time.perf_counter())
                self.ips[index].write(0,1)
            deadline=time.monotonic()+timeout
            while self.live:
                for index in tuple(self.live):
                    if self.ips[index].read(0)&2:
                        completed[index]=(time.perf_counter()-starts[index])*1000
                        self.live.remove(index)
                if time.monotonic()>deadline:
                    raise TimeoutError('Dual projection did not complete')
                if self.live:time.sleep(.0001)
        except BaseException:
            # Keep all caller-owned allocations alive until both masters stop.
            # A stuck hardware engine deliberately prevents unsafe buffer release.
            while self.live:
                for index in tuple(self.live):
                    if self.ips[index].read(0)&4:self.live.remove(index)
                if self.live:time.sleep(.1)
            raise
        elapsed=(time.perf_counter()-started)*1000
        scene[:scene_words].invalidate();texture[:texture_words].invalidate()
        status=[];written=[]
        for index,q in enumerate(queues):
            flat=q.reshape(-1)
            guard_end=len(self.jobs[index]) if verify_descriptors else min(len(self.jobs[index]),len(flat)+16)
            self.jobs[index][:guard_end].invalidate()
            assert np.array_equal(self.jobs[index][:len(flat)],flat)
            assert np.all(self.jobs[index][len(flat):guard_end]==CANARY)
            status.append(self.ips[index].read(REG['return']))
            written.append(self.ips[index].read(REG['written_cells']))
        return dict(milliseconds=elapsed,workers=workers,jobs_per_core=[len(q) for q in queues],
                    prepare_ms=prepare_ms,total_call_ms=(time.perf_counter()-call_started)*1000,
                    core_milliseconds=completed,start_skew_ms=(starts[-1]-starts[0])*1000,
                    status=status,written_cells=sum(written),written_cells_per_core=written,
                    descriptors_unchanged=True,descriptor_guards_intact=True,
                    full_descriptor_guard_scan=verify_descriptors)

    def close(self):
        if self.live:raise RuntimeError('Cannot release buffers owned by running hardware')
        for buf in self.jobs:buf.freebuffer()
        self.jobs=[]
        self.lock.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
