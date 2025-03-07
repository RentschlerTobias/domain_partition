import multiprocessing
import torch
import gmsh
import os
from tools import MeshGenerator, FrameField, NACA_airfoil

SAVE_DIR = "./datasets/"
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_mesh(result_list):
    
    try:
        airfoil = NACA_airfoil()
        mesh_gen = MeshGenerator(airfoil, quadMesh=True, lc=0.1)
        result_list.append(mesh_gen.mesh)  
    except Exception as e:
        gmsh.clear()
        gmsh.finalize()
        print(f'Mesh generation failed: {e}')

def generate_mesh_with_timeout(timeout_sec=60):
    with multiprocessing.Manager() as manager:
        result = manager.list()  
        process = multiprocessing.Process(target=generate_mesh, args=(result,))
        
        process.start()
        process.join(timeout=timeout_sec)

        if process.is_alive():
            print("Mesh generation timed out. Terminating process...")
            process.terminate()
            process.join()
            process.close()
            return None  

        process.close()
        return result if result else None  

def main():
    num_datapoints = 30
    num_needed_iteration = num_datapoints
    meshes = []
    timer_to_stop_gmsh = 60
    checkpoint_size = 100  

    for i in range(num_datapoints):
        mesh = generate_mesh_with_timeout(timeout_sec=timer_to_stop_gmsh)

        if mesh is not None:
            meshes.append(mesh)
        else:
            print(f"Skipping mesh {i+1} due to timeout or error.")
            num_needed_iteration += 1

        if (i + 1) % checkpoint_size == 0:
            torch.save(meshes, os.path.join(SAVE_DIR, f"quad_mesh_gmsh_dataset_{i+1}_meshes.pt"))
            print(f"Saved checkpoint: quad_mesh_gmsh_dataset_{i+1}_meshes.pt")

            meshes.clear()

    print("Merging all saved meshes into one dataset...")
    
    all_meshes = []
    for i in range(checkpoint_size, num_datapoints + 1, checkpoint_size):
        file_path = os.path.join(SAVE_DIR, f"quad_mesh_gmsh_dataset_{i}_meshes.pt")
        if os.path.exists(file_path):
            all_meshes.extend(torch.load(file_path))  # ✅ Load and merge all checkpoints

    torch.save(all_meshes, os.path.join(SAVE_DIR, "quad_mesh_gmsh_dataset.pt"))
    print("Final dataset saved: ./datasets/quad_mesh_gmsh_dataset.pt")
    print(f"Failed number of meshes: {num_needed_iteration - num_datapoints}")

if __name__ == "__main__":
    main()

