import torch

path= "./saved_meshes/frame_field_time_measured/num_meshes_200.pt"

meshes = torch.load(path)
list_of_meshes = meshes
len(meshes)


num_nodes = []
for i in range(len(list_of_meshes)):
    num_nodes.append(list_of_meshes[i].x.size(0))

len(num_nodes)
min(num_nodes)
max(num_nodes)

def group_meshes(list_of_meshes):
    num_nodes = []
    for in range(len(list_of_meshes)):
        num_nodes.append(list_of_meshes[i].x.size(0))

    return num_nodes

