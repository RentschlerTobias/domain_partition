import torch
from tools import MeshGenerator, NACA_airfoil
from tools.plotting_tools import *

def main():
    path_dataset = f"./datasets/quad_mesh_gmsh_dataset.pt"
    dataset = torch.load(path_dataset)    
    for i in range(len(dataset)):
        image_name = f'./figures/quad_mesh/quad_mesh_dataset_{i+1}.png'
        plot_egdes(dataset[i],output_file=image_name)

if __name__ == "__main__":
    main()
