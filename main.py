from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools.plotting_tools import *
from tools.singularity_detector import detect_singularities
from tools.separatrix_generator import SeparatrixGenerator
import gmsh
#gmsh.clear()
#gmsh.finalize()
#

airfoil = NACA_airfoil()
mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.05)
frameField = FrameField(mesh_gen.mesh)
streamline = StreamlineGenerator(frameField.mesh)
plot_streamlines(streamline.mesh)
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
