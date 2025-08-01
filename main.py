
from tools import StreamlinePostProcessor
from tools import StreamlineSimplificator
from tools import MeshCheck
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator_v2, Transfinite_Interpolation
from tools.plotting_tools import *
from torch_geometric.data import Data
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from data_generator import extract_mesh_data


#
# meshes = torch.load('./saved_meshes/checkpoints/checkpoint_mesh_2.pt')
# mesh = meshes[h]
#
plt.close

airfoil                     = NACA_airfoil()
random_lc                   = 0.04 + 0.02 * np.random.rand()
mesh_gen                    = MeshGenerator(airfoil, quadMesh=False, lc=random_lc)
frameField                  = FrameField(mesh_gen.mesh)
streamline                  = StreamlineGenerator_v2(frameField.mesh)

streamlines_post_processed  = StreamlineSimplificator(streamline.mesh)
blocked_mesh                = streamlines_post_processed.quad_mesh
transfiniteInterpolation    = Transfinite_Interpolation(blocked_mesh)
quad_mesh                   = transfiniteInterpolation.quad_mesh
tri_mesh                    = streamlines_post_processed.mesh
mesh_check                  = MeshCheck(tri_mesh, quad_mesh, tol=0.01)
success                     = mesh_check.is_valid
print(f'!!! \n area difference: \n {mesh_check.quad_area - mesh_check.tri_area}\n !!!')

mesh = extract_mesh_data(tri_mesh, quad_mesh, blocked_mesh)

plot_faces(blocked_mesh, colored=True, output_file="./figures/streamlines/faces_colored.png")
plot_intersections(mesh, output_file="./figures/streamlines/intersections.png")
plot_streamlines(mesh, output_file="./figures/streamlines/streamlines_post_processed.png")
plot_final_mesh(mesh, output_file="./figures/streamlines/quad_mesh.png")


torch.inf
PostProcessor = StreamlinePostProcessor(streamline.mesh)
PostProcessor.get_matching_streamlines()


Streamlines = PostProcessor.Streamlines
Singularities = PostProcessor.Singularities

for key in Singularities.keys():
    streamlines_in = Singularities[key]["s_in"]
    print(len(streamlines_in))


for key in Singularities.keys():
    streamlines_in = Singularities[key]["s_in"]
    if len(streamlines_in) > 0:
        for streamline_in in streamlines_in:
            print(streamline_in)


plot_streamlines(streamline.mesh, output_file="./figures/streamlines/streamlines.png")
