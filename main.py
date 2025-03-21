
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
#
# # mesh = torch.load('good_mesh.pt')
# mesh_failed_init = torch.load('failed_mesh.pt')
# mesh_good_init = torch.load('good_mesh.pt')
#
#
# simp_mesh_failed = StreamlineSimplificator(mesh_failed_init)
# mesh_failed = simp_mesh_failed.mesh
# quad_mesh = simp_mesh_failed.quad_mesh
# quad_mesh.streamlines = mesh_failed.streamlines
# plot_faces(quad_mesh)
#
# simp_mesh_good = StreamlineSimplificator(mesh_good_init)
# mesh_good = simp_mesh_good.mesh
#
# quad_mesh = simp_mesh_good.quad_mesh
# plot_faces(quad_mesh)
#
#
# quad_mesh = simp_mesh_failed.quad_mesh
# plot_faces(quad_mesh)
#
# # torch.save(streamline.mesh,'failed_mesh.pt')


def main():

    airfoil                    = NACA_airfoil()
    mesh_gen                   = MeshGenerator(airfoil, quadMesh=False, lc=0.05)
    frameField                 = FrameField(mesh_gen.mesh)
    streamline                 = StreamlineGenerator(frameField.mesh)
    streamlines_post_processed = StreamlineSimplificator(streamline.mesh)
    blocked_mesh               =  streamlines_post_processed.quad_mesh

    plot_streamlines(streamline.mesh)
    plot_intersections(streamlines_post_processed.mesh)
    plot_faces(blocked_mesh)

if __name__ == "__main__":
    main()
