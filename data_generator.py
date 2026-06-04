
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator_v2, StreamlinePostProcessor
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
import tempfile


def get_mesh():
    np.random.seed(int(time.time() * 1000) % 2**32 + os.getpid())

    try:
        airfoil = NACA_airfoil()
        random_lc = 0.04 + 0.02 * np.random.rand()
        mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=random_lc)
        frameField = FrameField(mesh_gen.mesh)
        streamline = StreamlineGenerator_v2(frameField.mesh)
        streamlines_post_processed = StreamlinePostProcessor(streamline.mesh)
        blocked_mesh = streamlines_post_processed.block_mesh
        transfiniteInterpolator = QuadMeshGenerator(blocked_mesh)
        quad_mesh = transfiniteInterpolator.transfinite_mesh
        tri_mesh = streamlines_post_processed.mesh
        mesh_check = MeshCheck(tri_mesh, quad_mesh, tol=1e-3)
        success = mesh_check.is_valid
        print(f'!!! \n area difference: \n {
              mesh_check.quad_area - mesh_check.tri_area}\n !!!')
        if success == True:
            mesh = extract_mesh_data(tri_mesh, quad_mesh, blocked_mesh)
            print('succssess')
            return mesh
        else:
            print('failed')
            return None
    except Exception as e:
        print(f'\n domain partition failed: {e} \n')


def _mesh_worker(queue, idx):
    try:
        mesh = get_mesh()  # → enthält Torch Tensoren
        if mesh is None:
            queue.put(None)
            return

        tmpfile = tempfile.NamedTemporaryFile(
            delete=False, suffix=f"_mesh_{idx}.pt"
        )
        torch.save(mesh, tmpfile.name)
        queue.put(tmpfile.name)

    except Exception as e:
        print(f"Fehler im Worker: {e}")
        queue.put(None)


# Hilfsfunktion: führt get_mesh in eigenem Prozess aus, killt bei Timeout
def run_with_timeout(idx, timeout=300):
    q = mp.Queue()
    p = mp.Process(target=_mesh_worker, args=(q, idx))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return None  # Timeout → kein Mesh

    return q.get() if not q.empty() else None


def main():
    number_of_meshes = 1000
    checkpoint_interval = 10  # alle 10 speichern
    checkpoint_dir = "./saved_meshes/checkpoints_snickers"
    os.makedirs(checkpoint_dir, exist_ok=True)

    successful_meshes = 0
    failed_meshes = 0
    counter = 0
    database = []  # sammelt Meshes bis 10

    while successful_meshes < number_of_meshes:
        tmp_path = run_with_timeout(counter, timeout=300)
        counter += 1
        print(f"\n--- counter: {counter} ---\n")

        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                # Mesh im Hauptprozess laden
                mesh_data = torch.load(tmp_path)
                os.remove(tmp_path)  # Temp-Datei wieder löschen

                successful_meshes += 1
                database.append(mesh_data)
                print(f"successful meshes: {successful_meshes}")

                # Alle 10 abspeichern
                if successful_meshes % checkpoint_interval == 0:
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f'checkpoint_mesh_{
                            successful_meshes}.pt'
                    )
                    torch.save(database, checkpoint_path)
                    database = []  # RAM freigeben
                    print(f"Checkpoint gespeichert: {checkpoint_path}")

            except Exception as e:
                failed_meshes += 1
                print(f"⚠️ Fehler beim Laden des Meshes: {e}")

        else:
            failed_meshes += 1
            print("⚠️ Mesh ist nicht valide oder Timeout erreicht")

    # letzten Rest speichern (<10)
    if database:
        checkpoint_path = os.path.join(
            checkpoint_dir, f'checkpoint_mesh_{successful_meshes}.pt'
        )
        torch.save(database, checkpoint_path)
        print(f"Final checkpoint gespeichert: {checkpoint_path}")

    print(f"Total failed meshes {
          failed_meshes}; total successful meshes {successful_meshes}")


def extract_mesh_data(tri_mesh, quad_mesh, block_mesh):

    # Extract features of the triangulated mesh incl. frame_field & streamline generation

    tri_coordinates = tri_mesh.x
    tri_faces = tri_mesh.faces
    tri_edges = tri_mesh.edge_index
    tri_edges_attr = tri_mesh.edge_attr
    tri_mesh_face_attr = tri_mesh.face_attr
    streamlines = tri_mesh.streamlines
    frame_field_angle = tri_mesh.frame_field_iteration_number
    frame_field_u = tri_mesh.u
    singularities_coords = tri_mesh.singularities_coords
    frame_field_iteration_number = tri_mesh.frame_field_iteration_number

    frame_field_time = tri_mesh.time_frame_field_generator
    singularities = tri_mesh.singularities

    # Extract the block structure
    blocking_nodes = block_mesh.x
    blocking_faces = block_mesh.faces
    edge_to_streamline = block_mesh.edge_to_streamline
    # Extract transfinite Interpolated mesh
    quad_coordinates = quad_mesh.x
    quad_faces = quad_mesh.faces
    quad_edges = quad_mesh.edge_index

    final_mesh = Data(blocking_nodes=blocking_nodes,
                      blocking_faces=blocking_faces,
                      quad_coordinates=quad_coordinates,
                      quad_faces=quad_faces, quad_edges=quad_edges,
                      singularities=singularities,
                      singularities_coords=singularities_coords,
                      frame_field_time=frame_field_time,
                      frame_field_iteration_number=frame_field_iteration_number,
                      frame_field_u=frame_field_u,
                      frame_field_angle=frame_field_angle,
                      streamlines=streamlines,
                      edge_to_streamline=edge_to_streamline,
                      tri_edges_attr=tri_edges_attr,
                      tri_mesh_face_attr=tri_mesh_face_attr,
                      tri_edges=tri_edges, tri_faces=tri_faces,
                      tri_coordinates=tri_coordinates)

    return final_mesh


if __name__ == "__main__":
    main()
