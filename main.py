from tools import MeshGenerator, FrameField

mesh_gen = MeshGenerator(lc=0.5, minBoundary=-5, maxBoundary=5, seed=42)

print(mesh_gen.mesh.edge_attr)

print(mesh_gen.mesh.face_attr)
frameField = FrameField(mesh_gen.mesh)
print(frameField)

frameField.add_cross_at_boundaries()
print(frameField.mesh)


def main():
    # Create an instance of MeshGenerator with specified parameters
    mesh_gen = MeshGenerator(lc=0.5, minBoundary=-5, maxBoundary=5, seed=42)

    # Access the generated mesh
    mesh = mesh_gen.mesh
    # print(mesh.x)


if __name__ == "__main__":
    main()
