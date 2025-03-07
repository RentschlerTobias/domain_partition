from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools.plotting_tools import *
import gmsh
#gmsh.clear()
#gmsh.finalize()
# airfoil = NACA_airfoil()
# mesh_gen = MeshGenerator(airfoil, quadMesh=True, lc=0.1)
#plot_egdes(mesh_gen.mesh)
def main():

    airfoil = NACA_airfoil()
    mesh_gen = MeshGenerator(airfoil, quadMesh=True, lc=0.05)
    plot_egdes(mesh_gen.mesh)

if __name__ == "__main__":
    main()
