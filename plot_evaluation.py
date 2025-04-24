
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator
from tools.plotting_tools import *
import torch

mesh = torch.load("./saved_meshes/good_mesh.pt")

post_processing = StreamlineSimplificator(mesh)
mesh_pp = post_processing.mesh
# mesh_post_processed.mesh.streamline_intersections
file = 'dfg_25f'

plot_boundary_egdes(mesh, output_file=f"./figures/mesh_boundary_edges_{file}.png")
plot_egdes(mesh, output_file=f"./figures/mesh_init_triangulation_{file}.png")


plot_vector_field(mesh, init = True, output_file=f"./figures/vector_field_init_{file}.png")
plot_vector_field(mesh, init = False, output_file=f"./figures/vector_field_propagated_{file}.png")

plot_cross_field(mesh, init=True, output_file=f"./figures/cross_field_init_{file}.png")
plot_cross_field(mesh, init=False, output_file=f"./figures/cross_field_propagated_{file}.png")

plot_singularities(mesh,output_file = f"./figures/mesh_singularities_{file}.png")


plot_streamlines(mesh,output_file=f"./figures/streamlines_colored_{file}.png",colored=True)

file = 'test'

plot_intersections(mesh_pp, output_file=f"./figures/intersections_{file}.png")

