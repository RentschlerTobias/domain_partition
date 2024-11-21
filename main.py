from tools import MeshGenerator_v2, FrameField, NACA_airfoil
from tools.plotting_tools import plot_mesh
from tools.streamline_tools.detect_singularities import *

# airfoil = NACA_airfoil()
# mesh_gen = MeshGenerator_v2(airfoil, quadMesh=True, lc=0.05)
# mesh = mesh_gen.mesh
# plot_mesh(mesh)

airfoil = NACA_airfoil()
mesh_gen = MeshGenerator_v2(airfoil, quadMesh=True, lc=0.05)
# mesh_gen = MeshGenerator_v2(airfoil, quadMesh=False, lc=0.05)
mesh = mesh_gen.mesh
plot_mesh(mesh)
frameField = FrameField(mesh_gen.mesh)
sing = detect_singularities(frameField.mesh)
plot_vector_field(mesh)


def main():

    airfoil = NACA_airfoil()
    mesh_gen = MeshGenerator_v2(airfoil, quadMesh=True, lc=0.05)
    # mesh_gen = MeshGenerator_v2(airfoil, quadMesh=False, lc=0.05)
    mesh = mesh_gen.mesh
    plot_mesh(mesh)
    frameField = FrameField(mesh_gen.mesh)
    sing = detect_singularities(frameField.mesh)
    plot_vector_field(mesh)


if __name__ == "__main__":
    main()
