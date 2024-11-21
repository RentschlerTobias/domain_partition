

def plot_mesh(mesh, output_file="mesh_airfoil.png"):
    import matplotlib.pyplot as plt
    coords = mesh.x[:, 0:2]

    for i in range(mesh.faces.size(1)):
        face = mesh.faces[:, i]
        nodes_of_face = mesh.x[face, :]
        plt.fill(nodes_of_face[:, 0], nodes_of_face[:, 1],
                 edgecolor='grey', fill=None)

    plt.axis('equal')  # Ensure equal scaling for x and y
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_vector_field(mesh, output_file="vector_field.png"):
    """
    Plot the 2D vector field stored in mesh and save the plot.

    Args:
    - mesh: The mesh object containing:
        - mesh.x[:, 0:2]: 2D node coordinates (Nx2 array)
        - mesh.frame_field: Corresponding vectors at each node (Nx2 array)
    - output_file: The name of the file to save the plot (default: "vector_field.png").
    """
    # Extract 2D node coordinates and vectors
    coords = mesh.x[:, 0:2]
    vectors = mesh.frame_field

    # Ensure numpy format for plotting
    coords = coords.cpu().numpy() if hasattr(coords, "cpu") else coords
    vectors = vectors.cpu().numpy() if hasattr(vectors, "cpu") else vectors

    # Unpack coordinates and vectors
    x, y = coords[:, 0], coords[:, 1]
    u, v = vectors[:, 0], vectors[:, 1]

    # Plot the vector field
    plt.figure(figsize=(10, 10))
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy',
               scale=20, color='blue', alpha=0.8)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Vector Field")
    plt.axis('equal')  # Ensure equal scaling for x and y
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Vector field plot saved as '{output_file}'.")
