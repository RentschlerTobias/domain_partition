from tools import QuadMeshGenerator
import torch
from tqdm import tqdm
#
# path_save = '/mnt/redacted'
# meshes = torch.load(path_save)
# len(meshes)
#


def main():

    path = f'/mnt/redacted'
    path_save = '/mnt/redacted'
    meshes = torch.load(path, weights_only=False)

    transfinite_divisions = [3, 4, 5]

    data = []
    for i in tqdm(range(len(meshes)), desc='mesh process'):

        mesh = meshes[i]
        mesh.faces = mesh.blocking_faces
        mesh.x = mesh.blocking_nodes

        for division in transfinite_divisions:
            generator = QuadMeshGenerator(mesh, transfinite_divisions=division)

            mesh_transfinite = generator.transfinite_mesh
            mesh_transfinite.tri_coordinates = mesh.tri_coordinates
            data.append(mesh_transfinite)

    torch.save(data, path_save)


if __name__ == "__main__":
    main()
