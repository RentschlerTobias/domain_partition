
from tools import StreamlinePostProcessor
from tools import StreamlineSimplificator
from tools import MeshCheck
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator,StreamlineGenerator_v2, Transfinite_Interpolation
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

frame_field_mesh = torch.load('mesh_frame_field.pt')

streamline                  = StreamlineGenerator(frame_field_mesh)

streamlines_post_processed  = StreamlineSimplificator(streamline.mesh)

airfoil                     = NACA_airfoil()
random_lc                   = 0.04 + 0.02 * np.random.rand()
mesh_gen                    = MeshGenerator(airfoil, quadMesh=False, lc=random_lc)
frameField                  = FrameField(mesh_gen.mesh)
streamline                  = StreamlineGenerator(frameField.mesh)

plot_streamlines_independet(streamline.mesh)
len(streamline.mesh.streamlines)

streamlines_post_processed.Singularity
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
plot_intersections(streamlines_post_processed.mesh, output_file="./figures/streamlines/intersections.png")
plot_streamlines(mesh, output_file="./figures/streamlines/streamlines_post_processed.png")
plot_final_mesh(mesh, output_file="./figures/streamlines/quad_mesh.png")
plot_faces_independet(mesh)
plot_streamlines(streamline.mesh, output_file="./figures/streamlines/streamlines.png")

streamlines_post_processed.get_streamlines_as_splines()

sls =   streamlines_post_processed.Streamlines

for key in sls.keys():
    print(sls[key]["coords"].shape)

for i in range(len(sls)):
    sl = sls[i]
    print(sl.shape)

x = sls[0][:,0]
x.size

sls =   streamline.mesh.streamlines
for i in range(len(sls)):
    sl = sls[i]
    print(sl.shape)

len(streamline.mesh.streamlines)
len(sls)
for  sl in streamlines_post_processed.mesh.streamlines:
    print(sl.size())

