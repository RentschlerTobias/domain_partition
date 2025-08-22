
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


def main():

    number_of_meshes = 100
    checkpoint_interval = 10  # Speichere alle x erfolgreiche Meshes
    checkpoint_dir = "./saved_meshes/checkpoints_mars"

    os.makedirs(checkpoint_dir, exist_ok=True)

    database = []
    successful_meshes = 0
    failed_meshes = 0
    counter = 0
    for n in range(number_of_meshes):
        is_valid = False

        while is_valid == False:
            try:
                mesh_data = mp.Pool(1).apply_async(get_mesh).get(timeout=300)
                counter += 1
                print(f"\n \n \n counter: {counter} \n \n \n")
                if mesh_data is not None:
                    is_valid = True
                    database.append(mesh_data)
                    successful_meshes += 1
                    print(f"successful meshes: {successful_meshes}")

                    if successful_meshes % checkpoint_interval == 0:
                        try:
                            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_mesh_{successful_meshes}.pt')
                            torch.save(database, checkpoint_path)
                            database = []  # resett database
                            print(f"Checkpoint gespeichert: {checkpoint_path}")
                        except Exception as checkpoint_error:
                            print(f"Warnung: Fehler beim Speichern des Checkpoints: {checkpoint_error}")
                else:
                    failed_meshes += 1
                    print(f"Warning: Transifinite Mesh is not valid")
            except Exception as checkpoint_error:
                print(f"Warnung: Fehler beim Speichern des Checkpoints: {checkpoint_error}")

    print(f'total failed meshes {failed_meshes}; total successful meshes {successful_meshes}')

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_mesh_{successful_meshes}.pt')
    torch.save(database, checkpoint_path)
    print(f"Final Checkpoint reached")


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
