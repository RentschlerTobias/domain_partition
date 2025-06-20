
from tools import StreamlineGenerator_v2
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator
from tools.plotting_tools import *
from tools.save_load import *
from torch_geometric.data import Data
import numpy as np
import torch


# mesh = torch.load("./saved_meshes/good_mesh.pt")

airfoil = NACA_airfoil()
mesh_gen = MeshGenerator(airfoil, quadMesh=True, lc=0.05)
mesh = mesh_gen.mesh

file = 'csme_motivation_06'

plot_mesh(mesh, output_file=f"./figures/{file}.png")




frameField = FrameField(mesh_gen.mesh)
streamline = StreamlineGenerator(frameField.mesh)
streamlines_post_processed = StreamlineSimplificator(streamline.mesh)
blocked_mesh = streamlines_post_processed.quad_mesh
file = 'test'

plot_streamlines(streamline.mesh,output_file=f"./figures/streamlines_{file}.png")
plot_intersections(streamlines_post_processed.mesh, output_file=f"./figures/intersections_{file}.png")
plot_faces(blocked_mesh,output_file=f"./figures/faces_{file}.png")





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


if __name__ == "__main__":
    main()
