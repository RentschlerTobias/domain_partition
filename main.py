from tools import MeshGenerator


def main():
    # Create an instance of MeshGenerator with specified parameters
    mesh_gen = MeshGenerator(lc=0.5, minBoundary=-5, maxBoundary=5, seed=42)

    # Access the generated mesh
    mesh = mesh_gen.mesh
    # print(mesh.x)


if __name__ == "__main__":
    main()
