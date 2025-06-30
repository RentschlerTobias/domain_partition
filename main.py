
from tools import StreamlineGenerator_v2
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator
from tools import Transfinite_Interpolation
from tools import MeshCheck
from tools.plotting_tools import *
from tools.save_load import *
from torch_geometric.data import Data
import numpy as np
import torch
 

# mesh = torch.load("./saved_meshes/good_mesh.pt")
#
# tri_mesh = streamline.mesh
# torch.save(tri_mesh, 'tri_mesh.pt')
#
# torch.save(quad_mesh, 'quad_mesh.pt')
#

tri_mesh  = torch.load('tri_mesh.pt')
quad_mesh  = torch.load('quad_mesh.pt')


mesh_check=MeshCheck(tri_mesh,quad_mesh)
print(mesh_check.is_vaild)
print(mesh_check.tri_area,mesh_check.quad_area)
airfoil = NACA_airfoil()
mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.05)
frameField = FrameField(mesh_gen.mesh)
streamline = StreamlineGenerator(frameField.mesh)
streamlines_post_processed = StreamlineSimplificator(streamline.mesh)
blocked_mesh = streamlines_post_processed.quad_mesh
transfiniteInterpolation = Transfinite_Interpolation(blocked_mesh)
quad_mesh =transfiniteInterpolation.quad_mesh

file = 'test'

plot_streamlines(streamline.mesh, output_file=f"./figures/streamlines_{file}.png")
plot_intersections(streamlines_post_processed.mesh, output_file=f"./figures/intersections_{file}.png")
plot_faces(blocked_mesh, output_file=f"./figures/faces_{file}.png")
plot_mesh(quad_mesh, output_file=f"./figures/transfinite_{file}.png")

plot_mesh(tri_mesh, output_file=f"./figures/triangulated_{file}.png")

is_valid=is_mesh_valid(streamline.mesh,quad_mesh)

for face in quad_mesh.faces

def main():

    
        airfoil = NACA_airfoil()
        mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.05)
        frameField = FrameField(mesh_gen.mesh)
        streamline = StreamlineGenerator(frameField.mesh)
        streamlines_post_processed = StreamlineSimplificator(streamline.mesh)
        streamlines_post_processed = StreamlineSimplificator(mesh)
        blocked_mesh = streamlines_post_processed.quad_mesh
        transfiniteInterpolation = Transfinite_Interpolation(blocked_mesh)
        quad_mesh =transfiniteInterpolation.quad_mesh
   
        file = 'test'
        
        plot_streamlines(streamline.mesh, output_file=f"./figures/streamlines_{file}.png")
        plot_intersections(streamlines_post_processed.mesh, output_file=f"./figures/intersections_{file}.png")
        plot_faces(blocked_mesh, output_file=f"./figures/faces_{file}.png")
        plot_mesh(quad_mesh, output_file=f"./figures/transfinite_{file}.png")
    
        is_valid=is_mesh_valid(streamline.mesh,quad_mesh)
        print(f'valid mesh{is_valid})

if __name__ == "__main__":
    main()
