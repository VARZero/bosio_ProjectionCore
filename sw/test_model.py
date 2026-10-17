import json
import unittest
from pathlib import Path
import numpy as np
from projection_model import *
from geometry_reference import cell_barycentrics

class GeometryTests(unittest.TestCase):
    def test_batched_descriptors(self):
        pose=dict(yaw=19,pitch=8,roll=11,fov_h=80,fov_v=62)
        batch,mask=prepare_window(257,193,stride=263,src=16,dst=16,**pose)
        for face in range(20):
            for tile in range(211):
                i=face*211+tile
                desc,visible=tile_descriptor(face,tile,257,193,stride=263,src=16,dst=16+i*256,**pose)
                np.testing.assert_array_equal(batch[i],desc)
                self.assertEqual(bool(mask[i]),visible)
    def test_all_tile_centroids(self):
        bary=cell_barycentrics(32);local=triangle_centers(32)
        for face in (0,7,19):
            for tile in range(211):
                np.testing.assert_allclose(local@tile_vertices(face,tile),bary[tile]@VERTICES[face],atol=1e-14)

    def test_fixed_coordinates_against_direct_geometry(self):
        maximum=0.;checked=0
        for face in range(20):
            for tile in (0,24,48,49,90,146,147,180,210):
                pose=dict(yaw=19,pitch=8,roll=11,fov_h=80,fov_v=62)
                desc,visible=tile_descriptor(face,tile,257,193,stride=263,**pose)
                u,v,d=cell_coordinates(desc)
                x,y,z=floating_coordinates(face,tile,257,193,**pose)
                mask=(z>.01)&(x>=0)&(x<=256)&(y>=0)&(y<=192)&(d>0)
                if mask.any():
                    self.assertTrue(visible)
                    maximum=max(maximum,float(np.max(np.abs(u[mask]/d[mask]-x[mask]))),float(np.max(np.abs(v[mask]/d[mask]-y[mask]))))
                    checked+=int(mask.sum())
        self.assertLess(maximum,.01)
        print(json.dumps(dict(visible_cells=checked,maximum_coordinate_error_pixels=maximum)))

    def test_culling_never_discards_fixed_point_visible_cells(self):
        for face in range(20):
            for tile in range(211):
                desc,visible=tile_descriptor(face,tile,257,193,yaw=19,pitch=8,roll=11,fov_h=80,fov_v=62)
                u,v,d=cell_coordinates(desc)
                mask=(d>0)&(u>=0)&(v>=0)&(u<=d*256)&(v<=d*192)
                if mask.any():self.assertTrue(visible)

if __name__=='__main__':unittest.main()
