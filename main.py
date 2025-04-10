
from tools import StreamlineGenerator_v2
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator
from tools.plotting_tools import *
from tools.save_load import *
from torch_geometric.data import Data
import numpy as np

import torch
#
# mesh_init = torch.load('mesh_wrong_separatrix.pt')
# streamline = StreamlineGenerator_v2(mesh_init)
# streamlines_post_processed = StreamlineSimplificator(streamline.mesh)
# blocked_mesh = streamlines_post_processed.quad_mesh
# plot_streamlines(streamline.mesh,output_file='./figures/streamlines_v2',colored = True)
# plot_intersections(streamlines_post_processed.mesh,output_file="./figures/intersections_merged_v3.png")
# plot_faces(blocked_mesh,output_file="./figures/faces_merged.png")
#

plot_faces(frameField.mesh,face_color ='black')

meshes = torch.load('./saved_meshes/frame_field_time_measured/all_meshes.pt')
mesh = frameField.mesh
mesh.time_frame_field_generator 
# Saveing
# save_object(airfoil, 'airfoil_failed_streamline.pkl')
# torch.save(mesh_gen.mesh, 'mesh_wrong_separatrix.pt')

# Loading 
# mesh = torch.load('good_mesh.pt')
# mesh = torch.load('mesh_wrong_separatrix.pt')

# mesh_init = torch.load('mesh_wrong_separatrix.pt')
# airfoil = load_objet('airfoil_failed_streamline.pkl')# Mesh where one streamline/separatrix is missing, further errors

airfoil = NACA_airfoil()
mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.1)
frameField = FrameField(mesh_gen.mesh)
streamline = StreamlineGenerator_v2(frameField.mesh)
streamlines_post_processed = StreamlineSimplificator(streamline.mesh)
blocked_mesh = streamlines_post_processed.quad_mesh

plot_streamlines(streamline.mesh)
plot_intersections(streamlines_post_processed.mesh)
plot_faces(blocked_mesh)

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
