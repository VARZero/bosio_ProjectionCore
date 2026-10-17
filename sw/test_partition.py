"""Check disjoint ownership and layer-order preservation before board execution."""
import unittest
import numpy as np
from partition_jobs import split_by_tile

class PartitionTests(unittest.TestCase):
    def test_shuffled_layers_preserve_per_tile_order(self):
        rng=np.random.default_rng(32)
        jobs=np.zeros((120,18),dtype=np.uint32)
        jobs[:,1]=16+rng.integers(0,17,120,dtype=np.uint32)*256
        jobs[:,5]=np.arange(120,dtype=np.uint32)
        queues=split_by_tile(jobs)
        owners=[set(map(int,q[:,1])) for q in queues]
        self.assertFalse(owners[0]&owners[1])
        self.assertEqual(sum(map(len,queues)),len(jobs))
        for dst in set(map(int,jobs[:,1])):
            selected=[q[q[:,1]==dst] for q in queues if dst in q[:,1]][0]
            np.testing.assert_array_equal(selected,jobs[jobs[:,1]==dst])

    def test_adjacent_tiles_and_empty_queue(self):
        jobs=np.zeros((4220,18),dtype=np.uint32);jobs[:,1]=16+np.arange(4220)*256
        self.assertEqual([len(q) for q in split_by_tile(jobs)],[2110,2110])
        self.assertEqual([len(q) for q in split_by_tile(jobs[:1])],[1,0])
        self.assertEqual([len(q) for q in split_by_tile(jobs[:0])],[0,0])

    def test_partial_overlap_rejected(self):
        jobs=np.zeros((2,18),dtype=np.uint32);jobs[:,1]=[16,17]
        with self.assertRaises(ValueError):split_by_tile(jobs)

    def test_oversized_tile_group_rejected(self):
        with self.assertRaises(ValueError):split_by_tile(np.zeros((8193,18),dtype=np.uint32))

if __name__=='__main__':unittest.main()
