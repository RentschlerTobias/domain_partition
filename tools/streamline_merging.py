from gmsh import merge
import numpy as np
import torch
from torch_geometric.data import Data
from scipy.interpolate import splprep, splev


class StreamlineMerging:

    def __init__(self, mesh: Data, verbose: bool = True):

        self.verbose = verbose
        self.Streamlines = self.prepare_streamlines(mesh)
        self.new_streamlines = self.merge_streamlines(self.Streamlines)

    def prepare_streamlines(self, mesh: Data):

        if self.verbose == True:
            print("\n function pre_processing \n")

        tol = 10e-3
        Singularities = {}
        Streamlines = {}

        mask_c0_nodes               = mesh.x[:, 2] == 0
        c0_nodes                    = mesh.x[mask_c0_nodes, 0:2]
        singularity_coords          = torch.tensor([mesh.singularities_coords[sing] for sing in mesh.singularities_coords])
        streamline_termination_nodes = torch.cat((c0_nodes, singularity_coords), 0)

        streamlines = mesh.streamlines

        for j in range(streamline_termination_nodes.size(0)):
            Singularities[j] = {"streamline_in": [], "streamline_out": [], "coords": streamline_termination_nodes[j], "is_boundary": j >= len(singularity_coords)}

        for i in range(len(streamlines)):
            streamline = torch.from_numpy(streamlines[i])
            Streamlines[i] = {"singularity_in": None, "singularity_out": None, "coords": streamline, "is_boundary": False}

        for i in range(len(streamlines)):
            streamline = torch.from_numpy(streamlines[i])
            start = streamline[0]
            end = streamline[-1]

            for j in range(streamline_termination_nodes.size(0)):

                termination_node    = streamline_termination_nodes[j, :]
                distance_start      = torch.linalg.norm(start - termination_node)
                distance_end        = torch.linalg.norm(end - termination_node)

                start_is_boundary = False
                end_is_boundary = False

                if distance_start < tol:

                    start_is_boundary = Singularities[j]["is_boundary"]
                    Singularities[j]["streamline_out"].append(i)
                    Streamlines[i]["singularity_out"] = j

                elif distance_end < tol:

                    end_is_boundary = Singularities[j]["is_boundary"]
                    Singularities[j]["streamline_in"].append(i)
                    Streamlines[i]["singularity_in"] = j

                else:

                    distance = torch.linalg.norm(streamline - termination_node, axis=1)
                    distance_min_idx = torch.argmin(distance)
                    distance_min = distance[distance_min_idx]

                    if distance_min < tol:

                        if self.verbose == True:
                            print(f"\n streamline cutted at index {distance_min_idx} \n")

                        end_is_boundary = Singularities[j]["is_boundary"]

                        cutted_streamline = streamline[0:distance_min_idx, :]

                        Singularities[j]["streamline_in"].append(i)
                        Streamlines[i]["coords"] = cutted_streamline
                        Streamlines[i]["singularity_in"] = j
                if start_is_boundary and end_is_boundary:
                    Streamlines[i]["is_boundary"] = True

        return Streamlines

    def merge_streamlines(self, Streamlines: dict):

        if self.verbose:
            print(f"\n started function merge_streamlines\n")

        merge_pairs = self.search_duplicated_streamlines(Streamlines)

        if self.verbose:
            print(f"\n found {len(merge_pairs)} streamlines to merge\n")

        merged_keys = set()
        for pair in merge_pairs:
            merged_keys.add(pair[0])
            merged_keys.add(pair[1])

        new_streamlines = []

        for key in Streamlines.keys():
            if key not in merged_keys:
                new_streamlines.append(np.array(Streamlines[key]["coords"]))

        for pair in merge_pairs:
            streamline_1 = Streamlines[pair[0]]["coords"]
            streamline_2 = Streamlines[pair[1]]["coords"]

            merged_streamline = self.interpolate_streamlines(streamline_1, streamline_2)
            new_streamlines.append(merged_streamline)

        return new_streamlines

    def search_duplicated_streamlines(self, Streamlines: dict):

        if self.verbose == True:
            print(f"\n started function search_duplicated_streamlines\n")

        merge_pairs = []

        for key_i in Streamlines.keys():
            start_i = Streamlines[key_i]["singularity_out"]
            end_i   = Streamlines[key_i]["singularity_in"]

            if end_i is None:
                continue

            for key_j in Streamlines.keys():
                start_j = Streamlines[key_j]["singularity_out"]
                end_j   = Streamlines[key_j]["singularity_in"]

                if end_j is None:
                    continue

                if Streamlines[key_i]["is_boundary"] and Streamlines[key_j]["is_boundary"]:
                    continue

                if start_i == end_j and start_j == end_i:

                    if [key_i, key_j] not in merge_pairs and [key_j, key_i] not in merge_pairs:
                        merge_pairs.append([key_i, key_j])

        return merge_pairs

    def get_streamlines_as_splines(self, streamlines):

        splines = []
        for i in range(len(streamlines)):
            streamline = np.array(streamlines[i])
            x = streamline[:, 0]
            y = streamline[:, 1]
            if x.size == 2:
                tck, u = splprep([x, y], s=0, k=1)  # k=1 linear splines
                splines.append([tck, u])
            else:
                tck, u = splprep([x, y], s=0)  # Cubic Splines (Default)
                splines.append([tck, u])

        return splines

    def interpolate_streamlines(self, streamline_ij, streamline_ji, num_points=100):
        # Convert streamlines to splines
        splines_ij = self.get_streamlines_as_splines([streamline_ij])
        splines_ji = self.get_streamlines_as_splines([streamline_ji])

        # Extract splines (assuming get_streamlines_as_splines returns [tck, u] for each streamline)
        tck_ij, u_ij = splines_ij[0]
        tck_ji, u_ji = splines_ji[0]

        # Generate uniform parameter values
        u_new = np.linspace(0, 1, num_points)

        # Evaluate splines at new parameter values
        x_ij, y_ij = splev(u_new, tck_ij)
        x_ji, y_ji = splev(u_new, tck_ji)

        x_merged = (1 - u_new) * x_ij + u_new * x_ji
        y_merged = (1 - u_new) * y_ij + u_new * y_ji
        # Stack and return merged streamline
        merged_streamline = np.vstack((x_merged, y_merged)).T

        return merged_streamline
