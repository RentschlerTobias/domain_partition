
from torch_geometric.data import Data
import torch
from tools.plotting_tools import *
import multiprocessing as mp
from data_generator import get_mesh    
import matplotlib.pyplot as plt


is_valid = False
        
while is_valid == False:
    block_mesh = mp.Pool(1).apply_async(get_mesh).get(timeout=600)
                   
    if block_mesh is not None:
        is_valid = True 


# plt.close('all')
#
# #
# # ### Load data
extension= 420
#
# extension += 1 
# meshes = torch.load(f'./saved_meshes/checkpoints/checkpoint_mesh_{extension}.pt')
# len(meshes)
# mesh = meshes[0]

file = f"./figures/streamlines/quad_mesh_{extension}.png"
plot_final_mesh(block_mesh,output_file=file)
#
#
file_streamlines = f"./figures/streamlines/streamlines_{extension}.png"
plot_streamlines(block_mesh,output_file=file_streamlines)
#
# # # Get amount of meshes inside the list
# number_data_point = len(meshes)
#
# mesh.streamline_intersections_points
# mesh.streamlines[6].shape
# #
# # ### Save list
# # path = './saved_meshes/checkpoints_test/checkpoint_mesh_0.pt'
# # torch.save(database, checkpoint_path)
# #
# #
# mesh
# def cut_streamlines(mesh):
#     streamlines = mesh.streamlines
#     singularities 
#     for streamline in streamlines:
#


## Start the domain partition pipeline
# mesh_data = mp.Pool(1).apply_async(get_mesh).get(timeout=600) #Single call 

## Analyzing the mesh

# plot_final_mesh(block_mesh,output_file=file)
# plot_final_mesh(block_mesh,output_file=file)
#

# i = 0
# mesh = meshes[i]
# plot_streamlines(mesh, output_file=f"./figures/streamlines{i}.png")
#
