
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator, Transfinite_Interpolation
from tools.plotting_tools import *
import torch


file = 'dfg_25f'
mesh = torch.load("./saved_meshes/good_mesh.pt", weights_only=False)

post_processing = StreamlineSimplificator(mesh)
mesh_pp = post_processing.mesh
blocked_mesh = post_processing.quad_mesh
transfinite_interpolation = Transfinite_Interpolation(blocked_mesh)
quad_mesh = transfinite_interpolation

plot_egdes(mesh, output_file=f"./figures/mesh_init_triangulation_{file}.png")


plot_boundary_egdes(mesh, output_file=f"./figures/mesh_boundary_edges_{file}.png")
plot_egdes(mesh, output_file=f"./figures/mesh_init_triangulation_{file}.png")
plot_vector_field(mesh, init=True, output_file=f"./figures/vector_field_init_{file}.png")
plot_vector_field(mesh, init=False, output_file=f"./figures/vector_field_propagated_{file}.png")
plot_cross_field(mesh, init=True, output_file=f"./figures/cross_field_init_{file}.png")
plot_cross_field(mesh, init=False, output_file=f"./figures/cross_field_propagated_{file}.png")
plot_singularities(mesh, output_file=f"./figures/mesh_singularities_{file}.png")
plot_streamlines(mesh, output_file=f"./figures/streamlines_colored_{file}.png", colored=True)
plot_intersections(mesh_pp, output_file=f"./figures/intersections_{file}.png")
plot_faces(blocked_mesh, output_file=f"./figures/faces_{file}.png")

len(blocked_mesh.edge_subdomain_points)

blocked_mesh.faces.size()
blocked_mesh.edge_index.size()
blocked_mesh.edge_subdomain_index


def main():

    file = 'dfg_25f'
    mesh = torch.load("./saved_meshes/good_mesh.pt", weights_only=False)

    post_processing = StreamlineSimplificator(mesh)
    mesh_pp = post_processing.mesh
    blocked_mesh = post_processing.quad_mesh

    plot_boundary_egdes(mesh, output_file=f"./figures/mesh_boundary_edges_{file}.png")
    plot_egdes(mesh, output_file=f"./figures/mesh_init_triangulation_{file}.png")
    plot_vector_field(mesh, init=True, output_file=f"./figures/vector_field_init_{file}.png")
    plot_vector_field(mesh, init=False, output_file=f"./figures/vector_field_propagated_{file}.png")
    plot_cross_field(mesh, init=True, output_file=f"./figures/cross_field_init_{file}.png")
    plot_cross_field(mesh, init=False, output_file=f"./figures/cross_field_propagated_{file}.png")
    plot_singularities(mesh, output_file=f"./figures/mesh_singularities_{file}.png")
    plot_streamlines(mesh, output_file=f"./figures/streamlines_colored_{file}.png", colored=True)
    plot_intersections(mesh_pp, output_file=f"./figures/intersections_{file}.png")
    plot_faces(blocked_mesh, output_file=f"./figures/faces_{file}.png")


if __name__ == "__main__":
    main()
