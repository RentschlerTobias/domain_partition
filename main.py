from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator
from tools.plotting_tools import *
import torch
from torch_geometric.data import Data
from scipy.interpolate import make_interp_spline, splprep, splev
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric
mesh = torch.load('good_mesh.pt')
simp = StreamlineSimplificator(mesh)
from torch_geometric.transforms import BaseTransform

len(simp.faces)
simp.faces[0]
faces = torch.tensor(simp.faces).T
edge_index = torch.cat([faces[:2],faces[1:3],faces[2:4],faces[::2],faces[1::2],faces[::3],], dim=1)
nodes = torch.tensor(simp.nodes_subdomain)
graph = Data(x = nodes, edge_index = edge_index,face = faces)

plot_faces(graph)

len(mesh.streamline_intersections)

mesh.streamline_intersections = simp.intersection_data[0]['points']
plot_intersections(mesh)
simp.intersection_data[0]['connectivity']






## MeshGeneration
#
# import gmsh
# gmsh.clear()
# gmsh.finalize()
#
# airfoil = NACA_airfoil()
# mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.05)
# mesh = mesh_gen.mesh
# frameField = FrameField(mesh_gen.mesh)
# streamline = StreamlineGenerator(frameField.mesh)
# mesh = streamline.mesh
# plot_streamlines(mesh)
# torch.save(mesh,'good_mesh.pt')


def main():

    airfoil = NACA_airfoil()
    mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.025)
    frameField = FrameField(mesh_gen.mesh)
    streamline = StreamlineGenerator(frameField.mesh)

    streamline.mesh.streamlines
if __name__ == "__main__":
    main()
