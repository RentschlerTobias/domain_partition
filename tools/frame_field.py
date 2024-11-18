import torch
import numpy as np
from tools.mesh_generator import MeshGenerator
import math


class FrameField:
    def __init__(self, meshOfMeshGenerator):

        self.mesh = meshOfMeshGenerator
        self.add_cross_at_boundaries()

    def map_cross_vectors_to_reference_vector(self, angle_rad):
        pi = torch.tensor(math.pi)

        if angle_rad < 0:
            angle_rad = angle_rad + 2*pi
        angle = angle_rad % (pi/2)
        angles = torch.tensor(
            [angle, angle + (pi/2), angle + (pi), angle + (3/2*pi)])
        ref_vec_of_cross = 4*torch.min(angles)
        return ref_vec_of_cross

    def add_cross_at_boundaries(self):

        num_nodes = self.mesh.x.size(0)
        mask_boundaryNodes = self.mesh.x[:, 2] != 2

        mask_boundaryEdges = self.mesh.edge_attr == 1
        boundary_edges = self.mesh.edge_index[:, mask_boundaryEdges[0]]
        midpoint = torch.mean(self.mesh.x[:, :2], dim=0)

        frame_field_angle = torch.zeros((num_nodes), dtype=torch.float)
        frame_field_coords = torch.zeros((num_nodes, 2), dtype=torch.float)
        normals = torch.zeros((num_nodes, 2), dtype=torch.float)

        for i in range(num_nodes):
            if mask_boundaryNodes[i] == True:
                boundary_edges_of_node = (
                    torch.where(boundary_edges[0, :] == i))[0]
                neighbours_idx = boundary_edges[1, boundary_edges_of_node]

                source_node = self.mesh.x[i, 0:2]
                destination_node0 = self.mesh.x[neighbours_idx[0], 0:2]
                destination_node1 = self.mesh.x[neighbours_idx[1], 0:2]
                edge0 = destination_node0 - source_node
                edge1 = destination_node1 - source_node

                #                length0           = torch.sqrt(torch.pow(destination_node0[0]-source_node[0],2)+torch.pow(destination_node0[1]-source_node[1],2))
                #                length1           = torch.sqrt(torch.pow(destination_node1[0]-source_node[0],2)+torch.pow(destination_node1[1]-source_node[1],2))
                #                length            = (length0+length1)/2
                #                length            = length0
                edge_midPoint = midpoint - source_node            # Calculate the angle
                # Convert radians to degrees and adjust to the range 0 to 360
                angle0 = (torch.atan2(edge0[1], edge0[0]))
                # Convert radians to degrees and adjust to the range 0 to 360
                angle1 = (torch.atan2(edge1[1], edge1[0]))

                angle_midpoint = (torch.atan2(
                    edge_midPoint[1], edge_midPoint[0]))

                angle = (angle0)  # +angle1) /2

                diff_angle = torch.absolute(angle - angle_midpoint)
                if diff_angle < torch.tensor(np.pi/2):
                    if angle > torch.tensor(np.pi):

                        angle -= torch.tensor(np.pi)
                    else:
                        angle += torch.tensor(np.pi)
                if angle0 < 0:
                    angle0 += torch.tensor(2*np.pi)

                if angle1 < 0:
                    angle1 += torch.tensor(2*np.pi)

                angle_ref_cross_vec = self.map_cross_vectors_to_reference_vector(
                    angle)

                frame_field_angle[i] = angle_ref_cross_vec
                frame_field_coords[i, 0] = torch.cos(angle_ref_cross_vec)
                frame_field_coords[i, 1] = torch.sin(angle_ref_cross_vec)
            else:
                frame_field_angle[i] = 0
                frame_field_coords[i, 0] = 0
                frame_field_coords[i, 1] = 0

        self.mesh.frame_field_angle = frame_field_angle
        self.mesh.frame_field_coords = frame_field_coords
