
from data_generator import extract_mesh_data
import os
import torch
import numpy as np
from torch_geometric.data import Data
from tools.plotting_tools import *
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator, StreamlineGenerator_v2, Transfinite_Interpolation
from tools import MeshCheck
from tools import StreamlineSimplificator_v2
from tools import StreamlineSimplificator
from tools import StreamlinePostProcessor
import matplotlib.pyplot as plt
plt.close()


torch.save(streamline.mesh, 'simple_mesh.pt')

mesh = torch.load('simple_mesh.pt', weights_only=False)

streamlines_post_processed  = StreamlinePostProcessor(mesh)
streamlines_post_processed.mesh.streamline_intersections['spline_intersections']
streamlines_post_processed.mesh
airfoil                     = NACA_airfoil()
random_lc                   = 0.04 + 0.02 * np.random.rand()
mesh_gen                    = MeshGenerator(airfoil, quadMesh=False, lc=random_lc)
frameField                  = FrameField(mesh_gen.mesh)
streamline                  = StreamlineGenerator(frameField.mesh)
streamlines_post_processed  = StreamlinePostProcessor(streamline.mesh)
blocked_mesh                = streamlines_post_processed.quad_mesh
transfiniteInterpolation    = Transfinite_Interpolation(blocked_mesh)
quad_mesh                   = transfiniteInterpolation.quad_mesh
tri_mesh                    = streamlines_post_processed.mesh
mesh_check                  = MeshCheck(tri_mesh, quad_mesh, tol=1e-3)
success                     = mesh_check.is_valid
print(f'!!! \n area difference: \n {mesh_check.quad_area - mesh_check.tri_area}\n !!!')

mesh = extract_mesh_data(tri_mesh, quad_mesh, blocked_mesh)

plot_faces(blocked_mesh, colored=True, output_file="./figures/streamlines/faces_colored.png")
plot_intersections(streamlines_post_processed.mesh, output_file="./figures/streamlines/intersections.png")
plot_streamlines(mesh, output_file="./figures/streamlines/streamlines_post_processed.png")
plot_final_mesh(mesh, output_file="./figures/streamlines/quad_mesh.png")
plot_faces_independet(mesh)
plot_streamlines(streamline.mesh, output_file="./figures/streamlines/streamlines.png")
print('f**')


streamlines_post_processed
