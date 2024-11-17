from tools.mesh_generator import MeshGenerator
from tools.frame_field.frame_field_generator import FrameField

mesh_gen = MeshGenerator(lc=0.5, minBoundary=-5, maxBoundary=5, seed=42)
frameField = FrameField(mesh_gen.mesh)
mesh = frameField.mesh
print(mesh.edge_attr.size())
print(mesh.face_attr.size())
num_nodes = mesh.x.size(0)
print(num_nodes)
mask_boundaryNodes = mesh.x[:, 2] != 2
print(mask_boundaryNodes.size())
mask_boundaryEdges = mesh.edge_attr == 1
mesh.edge_attr
print(mask_boundaryEdges)
boundary_edges = mesh.edge_index[:, mask_boundaryEdges[0]]


print(frameField)
frameField.foo()


frameField.add_cross_at_boundaries()


def main():
    # Create an instance of MeshGenerator with specified parameters
    mesh_gen = MeshGenerator(lc=0.5, minBoundary=-5, maxBoundary=5, seed=42)

    # Access the generated mesh
    mesh = mesh_gen.mesh
    # print(mesh.x)


if __name__ == "__main__":
    main()
