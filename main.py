from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator
from tools.plotting_tools import *
import torch
from torch_geometric.data import Data
from scipy.interpolate import make_interp_spline, splprep, splev
import numpy as np
import matplotlib.pyplot as plt

# import gmsh
# gmsh.clear()
# gmsh.finalize()
#

## MeshGeneration

airfoil = NACA_airfoil()
mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.05)
mesh = mesh_gen.mesh
frameField = FrameField(mesh_gen.mesh)
streamline = StreamlineGenerator(frameField.mesh)
mesh = streamline.mesh
plot_streamlines(mesh)
# torch.save(mesh,'good_mesh.pt')
mesh.streamlines
mesh = torch.load('good_mesh.pt')

simp = StreamlineSimplificator(mesh)
len(simp.streamline_splines[0][0])
len(mesh.streamlines)
plt.figure(figsize=(5, 5))

streamline = mesh.streamlines[i]
streamline = np.array(streamline)  
x = streamline[:, 0]
y =streamline[:, 1]
tck, u = splprep([x, y], s=0)
x.size
u_fine = np.linspace(0, 1, 100)  # More points for a smoother curve
x_smooth, y_smooth = splev(u_fine, tck)
plt.plot(x, y, 'ro', label='Original Points')
plt.plot(x_smooth, y_smooth, 'b-', label='Parametric B-Spline')
plt.plot(streamline[:, 0], streamline[:, 1],'r')
output_file = './figures/spline_test.png'
plt.savefig(output_file, dpi=300)



plot_vector_field(streamline.mesh)
plot_cross_field(streamline.mesh, init=False)

plot_cross_field(streamline.mesh, init=True, output_file="cross_field.png")
def main():

    airfoil = NACA_airfoil()
    mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.025)
    frameField = FrameField(mesh_gen.mesh)
    streamline = StreamlineGenerator(frameField.mesh)

    streamline.mesh.streamlines
if __name__ == "__main__":
    main()
