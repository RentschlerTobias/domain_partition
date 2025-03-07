import signal
from tools import MeshGenerator, FrameField, NACA_airfoil
import gmsh
import torch

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    """Raises TimeoutException when the function runs too long."""
    raise TimeoutException()

def generate_mesh_with_timeout(timeout=60):
    """Runs generate_mesh with a timeout using signals (Unix-based systems only)."""
    signal.signal(signal.SIGALRM, timeout_handler)  # Set signal handler
    signal.alarm(timeout)  # Set an alarm for `timeout` seconds

    try:
        mesh = generate_mesh()
        signal.alarm(0)  
        return mesh
    except TimeoutException:
        print("Mesh generation timed out, skipping...")
        return None
    

def generate_mesh():
    """Mesh generation function."""
    airfoil = NACA_airfoil()
    mesh_gen = MeshGenerator(airfoil, quadMesh=True, lc=0.1)
    return mesh_gen.mesh

def main():
    num_datapoints = 22 
    num_needed_iteration = num_datapoints
    meshes = []

    for i in range(num_datapoints):
        print(f'mesh number {i-(num_needed_iteration-num_datapoints)}')
        try:
            mesh = generate_mesh_with_timeout(timeout=60)  # Set a 5-second timeout
            if mesh is not None:
                meshes.append(mesh)
            else:
                print(f"Skipping mesh {i+1} due to timeout.")
                num_needed_iteration =num_needed_iteration +1
                gmsh.clear()
                gmsh.finalize()
        except Exception as e:
            print(f"Mesh generation failed: {e}")
            num_needed_iteration =num_needed_iteration +1
            gmsh.clear()
            gmsh.finalize()  
        if (i+1) % 100 == 0:
            torch.save(meshes, f"./datasets_test/quad_mesh_gmsh_dataset_{i+1}_meshes.pt")
            print(f"Saved intermediate checkpoint: quad_mesh_gmsh_dataset_{i+1}_meshes.pt")

    torch.save(meshes, f"./datasets/quad_mesh_gmsh_dataset.pt")
    print("Final dataset saved: ./datasets/quad_mesh_gmsh_dataset.pt")
    print(f'failed number of meshes {num_needed_iteration-num_datapoints}')

if __name__ == "__main__":
    main() 
