from tools import MeshFromFieldgen  
from tools.plotting_tools import *

fied_generator= MeshFromFieldgen("./mesh_out.obj")
fied_generator.mesh
def main():

    fied_generator= MeshFromFieldgen("./mesh_out.obj")
    mesh = fied_generator.mesh

    plot_mesh(mesh)
    plot_vector_field(mesh)
    plot_cross_field(mesh)

if __name__ == "__main__":
    main()
