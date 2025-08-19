
import os
import torch
import numpy as np
from torch_geometric.data import Data
from tools.plotting_tools import *
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator, StreamlineGenerator_v2, Transfinite_Interpolation


for i in range(9):
    pth = i + 1
    meshes = torch.load(f'saved_meshes/checkpoints/checkpoint_mesh_{pth}.pt', weights_only=False)
    for j, mesh in enumerate(meshes):
        plot_final_mesh(mesh, output_file=f"./figures/streamlines/quad_mesh_{pth}_{j}.png")
