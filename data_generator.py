
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator, StreamlinePostProcessor
from tools import QuadMeshGenerator
from tools import MeshCheck
from tools.plotting_tools import *
from tools.save_load import *
from torch_geometric.data import Data
import numpy as np
import torch
import os
import multiprocessing as mp
import time


def get_mesh():
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())

    try:
        airfoil                     = NACA_airfoil()
        random_lc                   = 0.04 + 0.02 * np.random.rand()
        mesh_gen                    = MeshGenerator(airfoil, quadMesh=False, lc=random_lc)
        frameField                  = FrameField(mesh_gen.mesh)
        streamline                  = StreamlineGenerator(frameField.mesh)
        streamlines_post_processed  = StreamlinePostProcessor(streamline.mesh)
        blocked_mesh                = streamlines_post_processed.block_mesh
        transfiniteInterpolator     = QuadMeshGenerator(blocked_mesh)
        quad_mesh                   = transfiniteInterpolator.transfinite_mesh
        tri_mesh                    = streamlines_post_processed.mesh
        mesh_check                  = MeshCheck(tri_mesh, quad_mesh, tol=1e-3)
        success                     = mesh_check.is_valid
        print(f'!!! \n area difference: \n {mesh_check.quad_area - mesh_check.tri_area}\n !!!')
        if success == True:
            mesh = extract_mesh_data(tri_mesh, quad_mesh, blocked_mesh)
            print('succssess')
            return mesh
        else:
            print('failed')
            return None
    except Exception as e:
        print(f'\n domain partition failed: {e} \n')



def extract_mesh_data(tri_mesh, quad_mesh, block_mesh):

    # Extract features of the triangulated mesh incl. frame_field & streamline generation

    tri_coordinates     = tri_mesh.x
    tri_faces           = tri_mesh.faces
    tri_edges           = tri_mesh.edge_index
    tri_edges_attr      = tri_mesh.edge_attr
    tri_mesh_face_attr  = tri_mesh.face_attr
    streamlines         = tri_mesh.streamlines
    frame_field_angle   = tri_mesh.frame_field_iteration_number
    frame_field_u       = tri_mesh.u
    singularities_coords = tri_mesh.singularities_coords
    frame_field_iteration_number   = tri_mesh.frame_field_iteration_number

    frame_field_time = tri_mesh.time_frame_field_generator
    singularities = tri_mesh.singularities

    # Extract the block structure
    blocking_nodes = block_mesh.x
    blocking_faces = block_mesh.faces

    # Extract transfinite Interpolated mesh
    quad_coordinates    = quad_mesh.x
    quad_faces          = quad_mesh.faces
    quad_edges          = quad_mesh.edge_index

    final_mesh          = Data(blocking_nodes=blocking_nodes, blocking_faces=blocking_faces, quad_coordinates=quad_coordinates, quad_faces=quad_faces, quad_edges=quad_edges, singularities=singularities, singularities_coords=singularities_coords, frame_field_time=frame_field_time, frame_field_iteration_number=frame_field_iteration_number, frame_field_u=frame_field_u, frame_field_angle=frame_field_angle, streamlines=streamlines, tri_edges_attr=tri_edges_attr, tri_mesh_face_attr=tri_mesh_face_attr, tri_edges=tri_edges, tri_faces=tri_faces, tri_coordinates=tri_coordinates)

    return final_mesh


if __name__ == "__main__":
    main()
