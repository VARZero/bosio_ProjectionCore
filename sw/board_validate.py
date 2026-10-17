"""Standalone PYNQ memory tests. Does not configure video or install services."""
import json
import time
from pathlib import Path
import numpy as np
from pynq import Overlay, allocate, Clocks
from projection_model import *

ROOT=Path(__file__).resolve().parent
REG=json.loads((ROOT/'registers.json').read_text())
GUARD=16
CANARY=np.uint32(0xa5c35a3c)

def main():
    overlay=Overlay(str(ROOT.parent/'bitstream/projection_memory.bit'))
    inherited_clock=float(Clocks.fclk0_mhz)
    Clocks.fclk0_mhz=100.0
    assert abs(float(Clocks.fclk0_mhz)-100.0)<.1
    ip=overlay.projector_0
    report=dict(clock_mhz=float(Clocks.fclk0_mhz),inherited_clock_mhz=inherited_clock,
                tests=[],scope='projection-only DDR; no Output Core')
    rng=np.random.default_rng(20261017)
    width,height,stride=257,193,263
    source=rng.integers(0,1<<24,size=GUARD+stride*height+GUARD,dtype=np.uint32)
    source[:GUARD]=CANARY;source[-GUARD:]=CANARY
    tex=allocate(shape=source.shape,dtype=np.uint32)
    dst=allocate(shape=(GUARD+4220*TILE_WORDS+GUARD,),dtype=np.uint32)
    jobs=allocate(shape=(8192*DESC_WORDS,),dtype=np.uint32)
    tex[:]=source;tex.flush()
    live=False

    def write(name,value):ip.write(REG[name],int(value))
    def pointer(name,buf):
        address=int(buf.physical_address)
        assert address<2**32
        write(name,address);ip.write(REG[name]+4,0)

    def run(name,descs,initial=None,expected_status=0,descriptor_words=None,scene_words=None):
        nonlocal live
        descs=np.asarray(descs,dtype=np.uint32).reshape(-1,DESC_WORDS)
        if initial is None:
            initial=np.zeros(len(dst),dtype=np.uint32)
            initial[:GUARD]=CANARY;initial[-GUARD:]=CANARY
        expected=initial.copy()
        expected_cells=0
        if expected_status==0:
            for desc in descs:expected_cells+=reference_job(source,expected,desc)
        dst[:]=initial;dst.flush()
        flat=descs.reshape(-1)
        jobs[:]=CANARY;jobs[:len(flat)]=flat;jobs.flush()
        pointer('texture',tex);pointer('descriptors',jobs);pointer('scene',dst)
        write('texture_words',len(source)-GUARD)
        write('descriptor_words',len(flat) if descriptor_words is None else descriptor_words)
        write('scene_words',len(dst)-GUARD if scene_words is None else scene_words)
        write('job_count',len(descs))
        start=time.perf_counter();write('control',1);live=True
        deadline=time.monotonic()+90
        while True:
            control=ip.read(REG['control'])
            if control&2:break
            if time.monotonic()>deadline:
                report['timeout']=name
                (ROOT/'board_results.json').write_text(json.dumps(report,indent=2))
                print('TIMEOUT: retaining DMA buffers until hardware becomes idle',flush=True)
                # Never free a buffer that may still be owned by the accelerator.
                while not(ip.read(REG['control'])&4):time.sleep(1)
                raise TimeoutError(name)
            time.sleep(.0001)
        elapsed=time.perf_counter()-start;live=False
        status=ip.read(REG['return']);written=ip.read(REG['written_cells'])
        dst.invalidate();tex.invalidate();jobs.invalidate()
        mismatches=np.flatnonzero(dst!=expected)
        item=dict(name=name,jobs=len(descs),milliseconds=elapsed*1000,status=status,
                  written_cells=written,expected_cells=expected_cells,
                  mismatched_words=int(len(mismatches)),
                  texture_unchanged=bool(np.array_equal(tex,source)),
                  descriptors_unchanged=bool(np.array_equal(jobs[:len(flat)],flat)),
                  descriptor_guard_intact=bool(np.all(jobs[len(flat):]==CANARY)))
        report['tests'].append(item)
        (ROOT/'board_results.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(item),flush=True)
        assert status==expected_status,(name,hex(status),hex(expected_status))
        assert written==expected_cells,(name,written,expected_cells)
        assert len(mismatches)==0,(name,mismatches[:12].tolist(),dst[mismatches[:12]].tolist(),expected[mismatches[:12]].tolist())
        assert item['texture_unchanged'] and item['descriptors_unchanged'] and item['descriptor_guard_intact']
        return np.array(dst)

    try:
        run('empty_job_list',[])
        desc=np.zeros(DESC_WORDS,dtype=np.uint32)
        desc[:6]=[GUARD,GUARD,width,height,stride,1|(0x35<<8)]
        coeff=np.array([0,0,256,128,0,256,0,128,256,0,0,0],dtype=np.int32)
        desc[6:]=coeff.view(np.uint32)
        run('fractional_bilinear_padded_stride',[desc])
        negative=desc.copy();negative[6]=np.uint32(0xfffff000)
        run('partial_window_clipping',[negative])
        zero=desc.copy();zero[14]=0
        run('zero_denominator_clear',[zero])
        back=desc.copy();back[14]=np.uint32(0xffffff00)
        run('behind_window_clear',[back])
        preserve=zero.copy();preserve[5]=0
        initial=rng.integers(0,2**32,size=len(dst),dtype=np.uint32)
        initial[:GUARD]=CANARY;initial[-GUARD:]=CANARY
        run('outside_preserves_existing_scene',[preserve],initial)
        edge=desc.copy();edge[6:]=np.array([(width-1)*256,0,0,0,(height-1)*256,0,0,0,256,0,0,0],dtype=np.uint32)
        run('exact_last_texel_clamped_bilinear',[edge])
        invalid=desc.copy();invalid[1]=len(dst)-GUARD-255
        run('reject_destination_overrun',[invalid],expected_status=0x20000000)
        invalid=desc.copy();invalid[0]=len(source)
        run('reject_texture_overrun',[invalid],expected_status=0x20000000)
        invalid=desc.copy();invalid[6]=0x7fffffff
        run('reject_coefficient_overflow',[invalid],expected_status=0x30000000)
        run('reject_short_descriptor_buffer',[desc],descriptor_words=17,expected_status=0x10000000)

        prepare_started=time.perf_counter()
        pose=dict(yaw=19,pitch=8,roll=11,fov_h=80,fov_v=62)
        all_jobs,mask=prepare_window(width,height,stride,src=GUARD,dst=GUARD,**pose)
        culled=all_jobs[mask]
        report['software_preparation']=dict(seconds=time.perf_counter()-prepare_started,
            examined_tiles=len(all_jobs),submitted_after_culling=len(culled),
            note='batched preparation including initial 297KiB geometry cache; no per-cell projection on CPU')
        full=run('full_sphere_4220_tiles',all_jobs)
        sparse=run('software_culled_same_scene',culled)
        assert np.array_equal(full,sparse)
        report['culled_matches_full_sphere']=True
        # Keep the actual hardware-produced payload for offline inspection.
        np.savez_compressed(ROOT/'m32_memory_scene.npz',
            cells=full[GUARD:-GUARD].view(np.uint8).reshape(20,211,1024),
            texture=source[GUARD:GUARD+height*stride].reshape(height,stride)[:,:width],
            pose=np.array([19,8,11,80,62],dtype=np.float64))
        # Multiple jobs targeting the same tile check ordered opaque overlay.
        first=all_jobs[0].copy();first[5]=1|(0x45<<8)
        candidates=[d.copy() for d in culled[:4]]
        for d in candidates:d[1]=GUARD;d[5]=0
        run('ordered_overlapping_window_jobs',[first]+candidates)
        report['passed']=True
        report['buffers_freed_after_completion']=True
        (ROOT/'board_results.json').write_text(json.dumps(report,indent=2))
        print('ALL BOARD MEMORY TESTS PASSED',flush=True)
    finally:
        if not live:
            tex.freebuffer();dst.freebuffer();jobs.freebuffer()

if __name__=='__main__':main()
