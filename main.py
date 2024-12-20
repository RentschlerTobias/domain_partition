from tools import MeshGenerator, FrameField, NACA_airfoil
from tools import MeshFromFieldgen  
from tools.plotting_tools import *
from tools.streamline_tools.detect_singularities import *


airfoil = NACA_airfoil()
mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.025)
mesh_gen.export_to_obj_field(filename = "mesh.obj")
#in console the command: ../fieldgen/fieldgen mesh.obj mesh_out.obj --degree=4 --alignToBoundary --s=0 
mesh = MeshFromFieldgen("./mesh_out.obj")
test_mesh = mesh.mesh
test_mesh.frame_field = test_mesh.cross_field
test_mesh.faces = test_mesh.face
torch.sum(torch.abs(mesh.mesh.singularities))

plot_mesh(test_mesh)
plot_vector_field(test_mesh)
# airfoil = NACA_airfoil()
# mesh_gen = MeshGenerator_v2(airfoil, quadMesh=True, lc=0.05)
# mesh = mesh_gen.mesh
# plot_mesh(mesh)

#mesh_gen = MeshGenerator_v2(airfoil, quadMesh=True, lc=0.05)
mesh_gen = MeshGenerator_v2(airfoil, quadMesh=False, lc=0.05)
mesh_gen.export_to_obj_fiel()
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
