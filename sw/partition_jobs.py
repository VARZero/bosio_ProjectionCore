"""Partition complete destination tiles; preserve each tile's layer order."""
import numpy as np

DESC_WORDS=18
TILE_WORDS=256

def split_by_tile(descriptors, workers=2):
    if workers not in (1,2):
        raise ValueError('workers must be 1 or 2')
    jobs=np.asarray(descriptors,dtype=np.uint32).reshape(-1,DESC_WORDS)
    groups={}
    for job in jobs:
        groups.setdefault(int(job[1]),[]).append(job)
    offsets=sorted(groups)
    if any(b-a<TILE_WORDS for a,b in zip(offsets,offsets[1:])):
        raise ValueError('Destination tile ranges partially overlap')
    queues=[[] for _ in range(workers)]
    loads=[0]*workers
    # Stable longest-group-first scheduling balances multi-layer tiles while
    # leaving the order of writes to each individual tile unchanged.
    for group in sorted(groups.values(),key=len,reverse=True):
        owner=min(range(workers),key=lambda i:loads[i])
        queues[owner].extend(group)
        loads[owner]+=len(group)
    if max(loads,default=0)>8192:
        raise ValueError('A core cannot accept more than 8192 jobs')
    return [np.asarray(q,dtype=np.uint32).reshape(-1,DESC_WORDS) for q in queues]
