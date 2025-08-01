import torch
import numpy as np
from collections import defaultdict


class StreamlinePostProcessor:
    def __init__(self, mesh):

        self.mesh = mesh

        self.Singularities, self.Streamlines = self.preProcessingStreamlines()

    def preProcessingStreamlines(self):

        streamlines = self.mesh.streamlines

        Singularities   = {}
        Streamlines     = {}

        tol = 10e-3

        mask_c0_nodes       = self.mesh.x[:, 2] == 0
        c0_nodes            = self.mesh.x[mask_c0_nodes, 0:2]
        singularity_coords  = torch.tensor([self.mesh.singularities_coords[sing] for sing in self.mesh.singularities_coords])

        streamline_termination_nodes = torch.cat((c0_nodes, singularity_coords), 0)

        num_c0_nodes = c0_nodes.size(0)

        for j in range(streamline_termination_nodes.size(0)):
            Singularities[j] = {"s_in": [], "s_out": [], "coords": streamline_termination_nodes[j, 0:2], "is_boundary": j < num_c0_nodes}

        for i in range(len(streamlines)):
            Streamlines[i] = {"s_in": None, "s_out": None, "angle_in": None, "angle_out": None, "coords": None, "cut_at_sing": None, "starts_at_boundary": False, "ends_at_boundary": False}

        for i in range(len(streamlines)):

            streamline = torch.from_numpy(streamlines[i])

            start   = streamline[0]
            end     = streamline[-1]

            for j in range(streamline_termination_nodes.size(0)):
                termination_node  = streamline_termination_nodes[j, :]
                distance_start = torch.norm(start - termination_node)
                distance_end = torch.norm(end - termination_node)

                if distance_start < tol:
                    dx = streamline[1, 0] - streamline[0, 0]
                    dy = streamline[1, 1] - streamline[0, 1]

                    Singularities[j]["s_out"].append(i)
                    Singularities[j]["coords"] = termination_node

                    Streamlines[i]["s_out"] = j
                    Streamlines[i]["angle_out"] = torch.atan2(dy, dx)
                    Streamlines[i]["starts_at_boundary"] = Singularities[j]["is_boundary"]
                elif distance_end < tol:

                    dx = streamline[-2, 0] - streamline[-1, 0]
                    dy = streamline[-2, 1] - streamline[-1, 1]

                    Singularities[j]["s_in"].append(i)

                    Streamlines[i]["s_in"] = j
                    Streamlines[i]["angle_in"] = torch.atan2(dy, dx)

                    Streamlines[i]["ends_at_boundary"] = Singularities[j]["is_boundary"]

            return Singularities, Streamlines

    def get_matching_streamlines(self):

        for key in self.Singularities.keys():

            # singularity_out = self.Streamlines[key]["s_out"]
            streamlines_in = self.Singularities[key]["s_in"]

            if len(streamlines_in) > 0:

                best_angle = torch.inf
                coords_sing_in = self.Singularities[key]["coords"]
                for streamline_in in streamlines_in:

                    id_singularity              = self.Streamlines[streamline_in]["s_out"]
                    coords_sing_out             = self.Singularities[id_singularity]["coords"]
                    dx      = coords_sing_in[0] - coords_sing_out[0]
                    dy      = coords_sing_in[1] - coords_sing_out[1]

                    angel   = torch.atan2(dy, dx)

                    print(streamline_in)
