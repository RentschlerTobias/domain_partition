from os import write
import gmsh
import numpy as np
import torch
from torch_geometric.utils import to_undirected, remove_isolated_nodes
from torch_geometric.data import Data


class QuadMeshGenerator:
    def __init__(self, blocked_geometry, lc=0.5):

        self.geometry = blocked_geometry 
        self.mesh = self.get_mesh(lc)

    def get_spline_of_edge(self,edge):

        source_node = self.geometry.x[edge[0],0,:2]
        destination_node = self.geometry.x[edge[1],0,:2]
        streamlines = self.geometry.streamlines    
        spline = torch.where
    def get_mesh(self, lc):

        gmsh.initialize()
        gmsh.model.add("Blockstructured Mesh")
        occ = gmsh.model.occ

        for face in self.geometry.faces.T:

            quad_nodes = self.geometry.x[face, :]

            edges = []
            edges = [[face[0],face[1]],[[face[1],face[2]],[[face[2],face[3]],[[face[3],face[0]]]

            for edge in egdes:
                spline = self.get_spline_of_edge(edge)



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
        streamlines =[]
        num_points = len(boundary_points)
        for i in range(num_points):
            start_point = boundary_points[i][:2]  # (x, y) of the first point
            end_point   = boundary_points[(i + 1) % num_points][:2]  # (x, y) of the next point
            boundary_streamline = []
            boundary_streamline.append(np.array(start_point))  # First point
            boundary_streamline.append(np.array(end_point))    # Second point
            
            streamlines.append(np.array(boundary_streamline))
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

        streamlines.append(np.array(airfoil.suction_side_rotated))
        streamlines.append(np.array(airfoil.pressure_side_rotated))
        print(f"suction_side_rotated shape: {np.array(airfoil.suction_side_rotated).shape}")
        print(f"pressure_side_rotated shape: {np.array(airfoil.pressure_side_rotated).shape}")
        # Add the plane surface
        plane_surface = occ.addPlaneSurface([outer_loop, airfoil_loop])
        occ.synchronize()

