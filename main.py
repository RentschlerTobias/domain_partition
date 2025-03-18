from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator
from tools.plotting_tools import *
import torch
from torch_geometric.data import Data
from scipy.interpolate import make_interp_spline, splprep, splev
import numpy as np
import matplotlib.pyplot as plt


mesh = torch.load('good_mesh.pt')
simp = StreamlineSimplificator(mesh)
mesh.streamline_intersections = simp.intersection_data[0]['points']
plot_intersections(mesh)

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
