from tools import MeshGenerator, NACA_airfoil

airfoil = NACA_airfoil()
mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.025)
mesh_gen.export_to_obj_field(filename = "mesh.obj")
#in console the command: ../fieldgen/fieldgen mesh.obj mesh_out.obj --degree=4 --alignToBoundary --s=0 
def main():

    airfoil = NACA_airfoil()
    mesh_gen = MeshGenerator(airfoil, quadMesh=False, lc=0.025)
    mesh_gen.export_to_obj_field(filename = "mesh.obj")

if __name__ == "__main__":
    main()
