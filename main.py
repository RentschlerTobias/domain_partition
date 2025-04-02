
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator
from tools.plotting_tools import *
from tools.save_load import *
from torch_geometric.data import Data
import numpy as np


# airfoil                    = NACA_airfoil()

# save_object(airfoil, 'airfoil_failed_streamline.pkl')
airfoil = load_object('airfoil_failed_streamline.pkl')

mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.04)
frameField = FrameField(mesh_gen.mesh)
streamline = StreamlineGenerator(frameField.mesh)
streamlines_post_processed = StreamlineSimplificator(streamline.mesh)
blocked_mesh = streamlines_post_processed.quad_mesh
plot_streamlines(streamline.mesh,output_file="./figures/streamlines_merged.png")
plot_intersections(streamlines_post_processed.mesh,output_file="./figures/intersections_merged.png")
plot_faces(blocked_mesh,output_file="./figures/faces_merged.png")

streamline.mesh.streamlines[10][-1, :]
streamlines_post_processed.edges_subdomain
streamlines_post_processed.nodes_subdomain
streamlines_post_processed.edge_points


mesh = torch.load('good_mesh.pt')
plot_intersections(mesh, output_file='./figures/streamlines.png', colored=True)

plot_intersections(
    streamline.mesh, output_file='./figures/domain_partition_failed.png', colored=False)
plot_streamlines(
    test_mesh, output_file='./figures/domain_partition_failed.png')


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
