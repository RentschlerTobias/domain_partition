
from tools import StreamlineGenerator_v2
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator
from tools.plotting_tools import *
from tools.save_load import *
from torch_geometric.data import Data
import numpy as np
import torch

# from csme_plots import Evaluation
# eval = Evaluation()
#


mesh = torch.load("./saved_meshes/good_mesh.pt")
len(mesh.streamlines)
streamline = StreamlineGenerator(mesh)

len(streamline.mesh.streamlines)
streamlines_post_processed = StreamlineSimplificator(streamline.mesh)

len(streamlines_post_processed.mesh.streamlines)
# blocked_mesh = streamlines_post_processed.quad_mesh
streamlines_post_processed.mesh.intersections
file = 'success'

plot_streamlines(streamline.mesh,output_file=f"./figures/streamlines_{file}.png",colored=True)
plot_intersections(streamlines_post_processed.mesh, output_file=f"./figures/intersections_{file}.png")
# plot_faces(blocked_mesh,output_file=f"./figures/faces_{file}.png")



airfoil = NACA_airfoil()
mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.05)
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
