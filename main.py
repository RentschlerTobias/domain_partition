
from tools import StreamlineGenerator_v2
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator
from tools import Transfinite_Interpolation
from tools.plotting_tools import *
from tools.save_load import *
from torch_geometric.data import Data
import numpy as np
import torch


# mesh = torch.load("./saved_meshes/good_mesh.pt")


airfoil = NACA_airfoil()
mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.05)
frameField = FrameField(mesh_gen.mesh)
streamline = StreamlineGenerator(frameField.mesh)
streamlines_post_processed = StreamlineSimplificator(streamline.mesh)
blocked_mesh = streamlines_post_processed.quad_mesh

file = 'test'

plot_streamlines(streamline.mesh, output_file=f"./figures/streamlines_{file}.png")
plot_intersections(streamlines_post_processed.mesh, output_file=f"./figures/intersections_{file}.png")
plot_faces(blocked_mesh, output_file=f"./figures/faces_{file}.png")

quad_mesh = Transfinite_Interpolation(blocked_mesh)

plot_mesh(quad_mesh.quad_mesh, output_file=f"./figures/transfinite_{file}.png")

tri_faces = streamline.mesh.faces
tri_face = tri_faces[:, 0]
tri_nodes = streamline.mesh.x[tri_faces, 0:2]

AB = tri_nodes[0, :] - tri_nodes[1, :]
AC = tri_nodes[0, :] - tri_nodes[0, :]
cross_product = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]

AB.size()


def main():

    airfoil = NACA_airfoil()
    mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.05)
    frameField = FrameField(mesh_gen.mesh)
    streamline = StreamlineGenerator(frameField.mesh)
    streamlines_post_processed = StreamlineSimplificator(streamline.mesh)
    blocked_mesh = streamlines_post_processed.quad_mesh
    plot_streamlines(streamline.mesh)
    plot_intersections(streamlines_post_processed.mesh)
    plot_faces(blocked_mesh)


def is_mesh_valid(tri_mesh, quad_mesh, tol=1e-6):

    tri_vertices = tri_mesh.x[:, 0:2]
    tri_faces = tri_mesh.faces.T
    tri_nodes = tri_vertices[tri_faces]
    A = tri_nodes[:, 0, :]
    B = tri_nodes[:, 1, :]
    C = tri_nodes[:, 2, :]
    AB = B - A
    AC = C - A
    cross_product = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
    tri_area = torch.sum(torch.abs(cross_product) / 2.0)

    quad_vertices = quad_mesh.x[:, 0:2]
    quad_faces = quad_mesh.faces.T
    quad_nodes = quad_vertices[quad_faces]
    A = quad_nodes[:, 0, :]
    B = quad_nodes[:, 1, :]
    C = quad_nodes[:, 2, :]
    D = quad_nodes[:, 3, :]
    AB = B - A
    AD = D - A
    BC = C - B
    cross1 = AB[:, 0] * AD[:, 1] - AB[:, 1] * AD[:, 0]
    cross2 = BC[:, 0] * AD[:, 1] - BC[:, 1] * AD[:, 0]
    quad_area = torch.sum(torch.abs(cross1 + cross2) / 2.0)

    return torch.abs(tri_area - quad_area) <= tol


if __name__ == "__main__":
    main()
