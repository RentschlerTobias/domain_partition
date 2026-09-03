
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator_v2, StreamlinePostProcessor
from tools import QuadMeshGenerator
from tools import MeshCheck, QuadPartitionValidator
from tools.plotting_tools import *
from tools.save_load import *
from torch_geometric.data import Data
import numpy as np
import torch
import os
import sys
import multiprocessing as mp
import time
import tempfile
import argparse
from tqdm import tqdm

# Node-local scratch. Mesh temp-handoff + checkpoints land hier statt auf dem
# (langsamen / netzgebundenen) Repo-Filesystem. Per env MESH_SCRATCH ueberschreibbar.
SCRATCH_DIR = os.environ.get("MESH_SCRATCH", "/sandbox/data_trent")


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
        tri_mesh = streamlines_post_processed.mesh

        validator = QuadPartitionValidator(
            blocked_mesh, tri_mesh, strict=False)

        if not validator.is_valid():
            print('failed: blocked mesh invalid (pre-filter)')
            print('\n'.join(validator.diagnostics()))
            return None
        qs = validator.quality_score()
        print(f"Pre-filter quality: SJ_min={qs.get('scaled_jacobian_min', -1):.3f}, "
              f"angle=[{qs.get('min_interior_angle', -1)                        :.1f}, {qs.get('max_interior_angle', -1):.1f}], "
              f"aspect={qs.get('edge_length_ratio_max', -1):.2f}")
        # --------------------------------------------------------------------

        transfiniteInterpolator = QuadMeshGenerator(blocked_mesh)
        quad_mesh = transfiniteInterpolator.transfinite_mesh
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
        import traceback
        print(f'\n domain partition failed: {type(e).__name__}: {e} \n')
        traceback.print_exc()


def _mesh_worker(queue, idx, quiet=False):
    # Silence ALL child output (Python prints + gmsh/numba C-level stdout/stderr)
    # at the file-descriptor level so the tqdm bar in the parent stays clean.
    if quiet:
        devnull = os.open(os.devnull, os.O_WRONLY)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    try:
        mesh = get_mesh()  # → enthält Torch Tensoren
        if mesh is None:
            queue.put(None)
            return

        os.makedirs(SCRATCH_DIR, exist_ok=True)
        tmpfile = tempfile.NamedTemporaryFile(
            delete=False, dir=SCRATCH_DIR, suffix=f"_mesh_{idx}.pt"
        )
        torch.save(mesh, tmpfile.name)
        queue.put(tmpfile.name)

    except Exception as e:
        print(f"Fehler im Worker: {e}")
        queue.put(None)


def main(quiet=True, number_of_meshes=10000, checkpoint_dir=None,
         num_workers=8, timeout=300):
    checkpoint_interval = 100
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(SCRATCH_DIR, "saved_meshes")
    os.makedirs(checkpoint_dir, exist_ok=True)

    successful_meshes = 0
    failed_meshes = 0
    counter = 0  # launched attempts
    database = []

    pbar = tqdm(total=number_of_meshes, desc="meshes",
                unit="mesh", disable=not quiet)

    active = {}  # slot -> {proc, q, t0}

    def launch(slot):
        nonlocal counter
        q = mp.Queue()
        p = mp.Process(target=_mesh_worker, args=(q, counter, quiet))
        p.start()
        active[slot] = {"proc": p, "q": q, "t0": time.time()}
        counter += 1

    # keep all worker slots busy while we still need successes
    for slot in range(num_workers):
        if successful_meshes < number_of_meshes:
            launch(slot)

    while active:
        time.sleep(0.05)
        for slot in list(active.keys()):
            info = active[slot]
            p = info["proc"]
            timed_out = (time.time() - info["t0"]) > timeout
            if p.is_alive() and not timed_out:
                continue

            tmp_path = None
            if p.is_alive():  # timed out
                p.terminate()
                p.join()
            else:
                p.join()
                try:
                    tmp_path = info["q"].get_nowait()
                except Exception:
                    tmp_path = None
            del active[slot]

            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    mesh_data = torch.load(tmp_path, weights_only=False)
                    os.remove(tmp_path)
                    successful_meshes += 1
                    database.append(mesh_data)
                    if successful_meshes <= number_of_meshes:
                        pbar.update(1)
                    if successful_meshes % checkpoint_interval == 0:
                        checkpoint_path = os.path.join(
                            checkpoint_dir,
                            f'checkpoint_mesh_{successful_meshes}.pt')
                        torch.save(database, checkpoint_path)
                        database = []
                except Exception:
                    failed_meshes += 1
            else:
                failed_meshes += 1

            pbar.set_postfix(trials=counter, fails=failed_meshes,
                             rate=f"{successful_meshes / max(counter, 1):.1%}")

            # refill the freed slot until target reached
            if successful_meshes < number_of_meshes:
                launch(slot)

    pbar.close()

    # flush remaining meshes (< checkpoint_interval)
    if database:
        checkpoint_path = os.path.join(
            checkpoint_dir, f'checkpoint_mesh_{successful_meshes}.pt')
        torch.save(database, checkpoint_path)
        print(f"Final checkpoint saved: {checkpoint_path}")

    print(f"Total failed meshes {failed_meshes}; "
          f"total successful meshes {successful_meshes}")


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
    parser = argparse.ArgumentParser(
        description="Quad-domain-partition data generator")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Full output (incl. gmsh), no progress bar. Default: silent + tqdm bar.")
    parser.add_argument("-n", "--number", type=int, default=1000,
                        help="Number of successful meshes to generate (default: 1000)")
    parser.add_argument("-o", "--out", type=str, default=None,
                        help=f"Checkpoint output dir (default: {SCRATCH_DIR}/saved_meshes, node-local)")
    parser.add_argument("-j", "--workers", type=int, default=8,
                        help="Concurrent worker processes (default: 8)")
    args = parser.parse_args()
    main(quiet=not args.verbose, number_of_meshes=args.number,
         checkpoint_dir=args.out, num_workers=args.workers)
