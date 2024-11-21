import gmsh
import numpy as np
import torch
from torch_geometric.data import torch_geometric
from torch_geometric.utils import to_undirected, remove_isolated_nodes
from torch_geometric.data import Data


class MeshGenerator_v2:
    def __init__(self, airfoil, lc=0.5, quadMesh=False):
        self.is_quad_mesh = quadMesh
        self.mesh = self.get_mesh_of_airfoil(lc, airfoil)

    def get_mesh_of_airfoil(self, lc, airfoil):
        gmsh.initialize()
        gmsh.model.add("Airfoil Mesh")
        occ = gmsh.model.occ

        boundary_points = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ]
        boundary_tags = []
        for i, point in enumerate(boundary_points):
            boundary_tags.append(occ.addPoint(*point, lc, tag=i + 1))
        boundary_lines = [
            occ.addLine(boundary_tags[0], boundary_tags[1]),
            occ.addLine(boundary_tags[1], boundary_tags[2]),
            occ.addLine(boundary_tags[2], boundary_tags[3]),
            occ.addLine(boundary_tags[3], boundary_tags[0])
        ]
        outer_loop = occ.addCurveLoop(boundary_lines)

        # Add airfoil geometry as a spline
        suction_points = []
        pressure_points = []
        for i, point in enumerate(airfoil.suction_side_rotated):
            suction_points.append(occ.addPoint(point[0], point[1], 0, lc))
        for i, point in enumerate(airfoil.pressure_side_rotated):
            pressure_points.append(occ.addPoint(point[0], point[1], 0, lc))

        suction_spline = occ.addSpline(suction_points)
        pressure_spline = occ.addSpline(pressure_points)
        airfoil_loop = occ.addCurveLoop([suction_spline, pressure_spline])

        # Add the plane surface
        plane_surface = occ.addPlaneSurface([outer_loop, airfoil_loop])
        occ.synchronize()
        if self.is_quad_mesh == True:
            # Mesh settings
            gmsh.option.setNumber("Mesh.Algorithm", 11)  # MeshAdapt algorithm
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Smoothing", 10)  # Smoothing steps

        # Generate the mesh
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords).reshape(-1, 3)
        node_coords_tensor = torch.from_numpy(node_coords).float()
        node_tags = node_tags - 1  # Convert to 0-based indexing

        element_types, element_tags, node_tags_per_element = gmsh.model.mesh.getElements()

        faces = None

        # Check for triangles and quads
        for etype, etags, ntags in zip(element_types, element_tags, node_tags_per_element):
            if etype == 2:  # Triangles
                faces = np.array(ntags).reshape(-1, 3) - \
                    1  # Convert to 0-based indexing
                break
            elif etype == 3:  # Quadrilaterals
                faces = np.array(ntags).reshape(-1, 4) - \
                    1  # Convert to 0-based indexing

        if faces is None:
            raise ValueError(
                "No triangular or quadrilateral elements found in the mesh.")

        # Convert faces to PyTorch format
        faces_tensor = torch.tensor(faces.T.astype(
            np.int64), dtype=torch.long)  # Transpose and cast to int64

        # Remove isolated nodes and adjust new node indices in faces_tensor
        edge_index = self.face_to_edges(faces_tensor)
        new_edge_index, _, mask = remove_isolated_nodes(edge_index)

        # Create a mapping for old to new indices
        index_mapping = torch.full(
            (node_coords_tensor.size(0),), -1, dtype=torch.long)
        index_mapping[mask] = torch.arange(mask.sum(), dtype=torch.long)

        # Adjust faces to remove invalid faces and remap indices
        valid_faces_mask = (index_mapping[faces_tensor] >= 0).all(
            dim=0)  # Check if all nodes in a face are valid
        # Keep only valid faces
        filtered_faces = faces_tensor[:, valid_faces_mask]
        # Remap old indices to new indices
        updated_faces = index_mapping[filtered_faces]

        # Update node coordinates based on mask
        new_node_coords = node_coords_tensor[mask, :]

        # Process node dimensions if needed
        for i in range(new_node_coords.size(0)):
            nodeTag = i + 1
            coord, _, dim, tag = gmsh.model.mesh.getNode(nodeTag)
            new_node_coords[i, 2] = dim

        gmsh.finalize()

        mesh = Data(x=new_node_coords, edge_index=new_edge_index,
                    faces=updated_faces)

        return mesh

    def face_to_edges(self, faces):
        if faces.size()[0] == 3:
            edges = torch.cat(
                [faces[[0, 1], :], faces[[1, 2], :], faces[[2, 0], :]], dim=1)
        if faces.size()[0] == 4:
            edges = torch.cat([faces[[0, 1], :], faces[[1, 2], :], faces[[
                              2, 3], :], faces[[3, 0], :]], dim=1)

        edges = torch_geometric.utils.to_undirected(edges)
        edges = edges.to(torch.long)
        return edges
