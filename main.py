
from tools import SeparatrixGenerator_v2
from tools import StreamlineGenerator_v2
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator
from tools.plotting_tools import *
from tools.save_load import *
from torch_geometric.data import Data
import numpy as np

import torch

# Saveing
# save_object(airfoil, 'airfoil_failed_streamline.pkl')
# torch.save(mesh_gen.mesh, 'mesh_wrong_separatrix.pt')

# Loading 
# airfoil                    = NACA_airfoil() 
# mesh = torch.load('good_mesh.pt')
# mesh = torch.load('mesh_wrong_separatrix.pt')

# airfoil = load_objet('airfoil_failed_streamline.pkl')# Mesh where one streamline/separatrix is missing, further errors


mesh_init = torch.load('mesh_wrong_separatrix.pt')
separatrices = SeparatrixGenerator_v2(mesh_init)

mesh =separatrices.mesh
torch.sum(torch.abs(mesh.singularities))
len(mesh.separatrices)
streamline = StreamlineGenerator_v2(mesh)

nodes = [sepa['singularity_coords'] for sepa in mesh.separatrices]
ids = [sepa['face_id'] for sepa in mesh.separatrices]

print(ids)
print(torch.stack(nodes))


streamline = StreamlineGenerator(mesh_init)
streamlines_post_processed = StreamlineSimplificator(streamline.mesh)
blocked_mesh = streamlines_post_processed.quad_mesh
plot_singularities(streamline.mesh)
plot_intersections(streamlines_post_processed.mesh,output_file="./figures/intersections_merged_v2.png")
plot_faces(blocked_mesh,output_file="./figures/faces_merged.png")

print([stream['face_id'] for stream in streamline.mesh.separatrices])


        self.separatrices = SeparatrixGenerator(mesh)
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
