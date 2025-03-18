from scipy.interpolate import make_interp_spline, splprep, splev
import numpy as np
from collections import defaultdict
import networkx as nx
import torch


class StreamlineSimplificator:
    def __init__(self,mesh):
        self.mesh = mesh
        self.streamline_splines = self.get_streamlines_as_splines()
        self.intersection_data  = self.find_all_intersections()
        self.quad_edges = self.split_splines_at_intersections(self.intersection_data)

        self.edges_subdomain, self.nodes_subdomain, self.edge_points = self.extract_subdomain_arrays(self.intersection_data)
        self.faces =  self.get_faces()
        
        # self.quad_mesh = self.get_quad_mesh()

    # def get_quad_mesh(self):

    def get_faces(self):
        num_edges = self.edges_subdomain[0].size
        edges = []
        for i in range(num_edges):
            edge = (self.edges_subdomain[0][i],self.edges_subdomain[1][i])
            edges.append(edge)
        graph = nx.Graph()
        
        graph.add_edges_from(edges)
        
        faces = list(nx.simple_cycles(graph, length_bound=4))
        return faces 

    def get_streamlines_as_splines(self):
        splines = []
        streamlines = self.mesh.streamlines
        for i in range(len(streamlines)):
            streamline = np.array(streamlines[i])  
            x = streamline[:, 0]
            y = streamline[:, 1]
            if x.size == 2:
                tck, u = splprep([x, y], s=0, k=1)  # k=1 linear splines
                splines.append([tck, u])
            else:
                tck, u = splprep([x, y], s=0) # Cubic Splines (Default)
                splines.append([tck, u])

        return splines
 
    def split_splines_at_intersections(self, intersection_data):

        """
        Returns:
        Dictionary with:
            - 'edges': List of edges where each edge is a list of points
            - 'edge_to_parent': Dict mapping edge_idx to original spline_idx
            - 'nodes': List of intersection nodes
        """

        splines = self.streamline_splines
        
        # Extract the dictionary from the tuple
        intersection_dict       = intersection_data[0]
        points                  = intersection_dict['points']
        spline_intersections    = intersection_dict['spline_intersections']
        
        edges = []
        edge_to_parent = {}
        
        # Process each spline
        for spline_idx, spline in enumerate(splines):
            if spline is None:
                continue
                
            tck, u = spline
            
            # Get all intersection parameters for this spline, sorted by t
            intersections = []
            if spline_idx in spline_intersections:
                # Each intersection is (point_idx, t)
                intersections = sorted(spline_intersections[spline_idx], key=lambda x: x[1])
            
            # If no intersections, add the entire spline as one edge
            if not intersections:
                t_values = np.linspace(0, 1, 50)  # Sample points along the spline
                edge_points = np.array(splev(t_values, tck)).T.tolist()
                edge_idx = len(edges)
                edges.append(edge_points)
                edge_to_parent[edge_idx] = spline_idx
                continue
            
            # Add endpoints as special cases (t=0 and t=1)
            # Check if the first intersection is not at t=0
            if intersections[0][1] > 0.001:  # Small tolerance
                t_values = np.linspace(0, intersections[0][1], 20)
                edge_points = np.array(splev(t_values, tck)).T.tolist()
                edge_idx = len(edges)
                edges.append(edge_points)
                edge_to_parent[edge_idx] = spline_idx
            
            # Create edges between consecutive intersections
            for i in range(len(intersections) - 1):
                point_idx1, t1 = intersections[i]
                point_idx2, t2 = intersections[i + 1]
                
                # Skip if they're too close
                if abs(t2 - t1) < 0.001:
                    continue
                    
                t_values = np.linspace(t1, t2, max(2, int((t2 - t1) * 50)))
                edge_points = np.array(splev(t_values, tck)).T.tolist()
                edge_idx = len(edges)
                edges.append(edge_points)
                edge_to_parent[edge_idx] = spline_idx
            
            # Check if the last intersection is not at t=1
            if intersections[-1][1] < 0.999:  # Small tolerance
                t_values = np.linspace(intersections[-1][1], 1, 20)
                edge_points = np.array(splev(t_values, tck)).T.tolist()
                edge_idx = len(edges)
                edges.append(edge_points)
                edge_to_parent[edge_idx] = spline_idx
        
        # Create a convenient list of nodes for reference
        nodes = [tuple(point) for point in points]
        
        return {
            'edges': edges,
            'edge_to_parent': edge_to_parent,
            'nodes': nodes
        }
    


    def get_intersections(self):

        intersections = [] 
        dicc_id = 0  
        for i in range(len(self.streamline_splines)):
            streamline_I = self.streamline_splines[i]  
            for j in range(i + 1, len(self.streamline_splines)): 

                streamline_J = self.streamline_splines[j]

                confirmed_intersection = self.find_spline_intersections_with_params(streamline_I, streamline_J)
    
                if confirmed_intersection:

                    intersections.append(confirmed_intersection)
                
        return intersections


    def find_all_intersections(self, tolerance=1e-6):
            
            splines = self.streamline_splines 
            all_intersections = [] 
    
            for i in range(len(splines)):
                for j in range(i+1, len(splines)):
                    if splines[i] is None or splines[j] is None:
                        continue
    
                    intersections = self.find_spline_intersections_with_params(splines[i], splines[j], tolerance)
    
                    for point, t1, t2 in intersections:
                        all_intersections.append((point, i, t1, j, t2))
    
            unique_points = []
            point_indices = {}  
    
            spline_intersections = defaultdict(list)  # Maps spline_idx to [(point_idx, t), ...]
            connectivity = defaultdict(list)  # Maps point_idx to [spline_idx, ...]
    
            for point, spline1_idx, t1, spline2_idx, t2 in all_intersections:
                # Check if this point is already in unique_points
                is_new_point = True
                point_idx = None
    
                for idx, existing_point in enumerate(unique_points):
                    if np.linalg.norm(np.array(point) - np.array(existing_point)) < tolerance:
                        is_new_point = False
                        point_idx = idx
                        break

                if is_new_point:
                    point_idx = len(unique_points)
                    unique_points.append(point)
    
                # Update spline_intersections
                spline_intersections[spline1_idx].append((point_idx, t1))
                spline_intersections[spline2_idx].append((point_idx, t2))
    
                # Update connectivity
                if spline1_idx not in connectivity[point_idx]:
                    connectivity[point_idx].append(spline1_idx)
                if spline2_idx not in connectivity[point_idx]:
                    connectivity[point_idx].append(spline2_idx)
    
            # Sort spline_intersections by parameter value t
            for spline_idx in spline_intersections:
                spline_intersections[spline_idx].sort(key=lambda x: x[1])
    
            return {
                'points': unique_points,
                'spline_intersections': dict(spline_intersections),
                'connectivity': dict(connectivity)
            }, intersections
    

    def find_spline_intersections_with_params(self,spline1, spline2, tolerance=1e-6, num_samples=10):
        
        offset_boundingBox = 0.25 
        tck1, u1 = spline1
        tck2, u2 = spline2

        u1_fine = np.linspace(0, 1, num_samples)
        u2_fine = np.linspace(0, 1, num_samples)

        points1 = np.array(splev(u1_fine, tck1)).T
        points2 = np.array(splev(u2_fine, tck2)).T

        # Find potential intersection regions
        potential_intersections = []

        # For each segment in spline1, check for potential intersections with segments in spline2
        for i in range(len(points1) - 1):
            for j in range(len(points2) - 1):
                # Check if the bounding boxes of the segments overlap
                min_x1, max_x1 = min(points1[i][0], points1[i+1][0]), max(points1[i][0], points1[i+1][0])
                min_y1, max_y1 = min(points1[i][1], points1[i+1][1]), max(points1[i][1], points1[i+1][1])

                min_x2, max_x2 = min(points2[j][0], points2[j+1][0]), max(points2[j][0], points2[j+1][0])
                min_y2, max_y2 = min(points2[j][1], points2[j+1][1]), max(points2[j][1], points2[j+1][1])
                
                # extent Bounding Box to get intersections where start/endpoint are the same 
                # example horizonal and vertical line that are starting at the same point will not be detected.
                
                min_x11 = min_x1 - max(offset_boundingBox, offset_boundingBox * min_x1)
                max_x11 = max_x1 + max(offset_boundingBox, offset_boundingBox * max_x1)

                min_y11 = min_y1 - max(offset_boundingBox, offset_boundingBox * min_y1)
                max_y11 = max_y1 + max(offset_boundingBox, offset_boundingBox * max_y1)
   
                min_x22 = min_x2 - max(offset_boundingBox, offset_boundingBox * min_x2)
                max_x22 = max_x2 + max(offset_boundingBox, offset_boundingBox * max_x2)

                min_y22 = min_y2 - max(offset_boundingBox, offset_boundingBox * min_y2)
                max_y22 = max_y2 + max(offset_boundingBox, offset_boundingBox * max_y2)


                # If bounding boxes overlap, add to potential intersections
                if (min_x11 <= max_x22 and max_x11 >= min_x22 and
                    min_y11 <= max_y22 and max_y11 >= min_y22):
                    u1_val = u1_fine[i]
                    u2_val = u2_fine[j]
                    potential_intersections.append((u1_val, u2_val))

        # Refine potential intersections
        confirmed_intersections = []

        for u1_val, u2_val in potential_intersections:
            # Define a function to find the root of (distance between points on the splines)
            def distance_func(params):
                t1, t2 = params
                # Ensure t1 and t2 are within [0, 1]
                t1 = max(0, min(1, t1))
                t2 = max(0, min(1, t2))

                point1 = np.array(splev(t1, tck1)).reshape(2)
                point2 = np.array(splev(t2, tck2)).reshape(2)

                return [point1[0] - point2[0], point1[1] - point2[1]]

            # Initial guess
            initial_guess = [u1_val, u2_val]

            # Solve for intersection
            try:
                from scipy.optimize import fsolve
                t1_intersect, t2_intersect = fsolve(distance_func, initial_guess)

                # Check if t1_intersect and t2_intersect are within [0, 1]
                if 0 <= t1_intersect <= 1 and 0 <= t2_intersect <= 1:
                    point1 = np.array(splev(t1_intersect, tck1)).reshape(2)
                    point2 = np.array(splev(t2_intersect, tck2)).reshape(2)

                    # Check if points are close enough
                    if np.linalg.norm(point1 - point2) < tolerance:
                        # Average the two points to get the intersection point
                        intersection_point = ((point1[0] + point2[0])/2, (point1[1] + point2[1])/2)

                        # Check if this intersection is already in the list
                        is_duplicate = False
                        for existing_point, _, _ in confirmed_intersections:
                            if np.linalg.norm(np.array(existing_point) - np.array(intersection_point)) < tolerance:
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            confirmed_intersections.append((intersection_point, t1_intersect, t2_intersect))
            except:
                # If fsolve fails, skip this potential intersection
                continue

        return confirmed_intersections


    def extract_subdomain_arrays(self, intersection_data):
        """
        Extracts intersection data into NumPy arrays for subdomain creation.
        
        Parameters:
        intersection_data: Tuple containing intersection information
        
        Returns:
        Dictionary with:
            - 'edges_subdomain': [2 x n_edges] array with indices of connected intersection points
            - 'nodes_subdomain': [n_intersection_nodes x 2] array with x,y coordinates of intersection points
            - 'edge_points': List where each entry is an array of points along the corresponding edge
        """
        # Extract intersection dictionary from tuple
        intersection_dict = intersection_data[0]
        points = intersection_dict['points']
        spline_intersections = intersection_dict['spline_intersections']
        
        # Create nodes_subdomain: array of node coordinates
        nodes_subdomain = np.array(points)
        
        # Prepare to build edges_subdomain and edge_points
        edges_list = []
        edge_points = []
        
        # Process each spline
        for spline_idx in spline_intersections:
            # Get the spline data
            tck, u = self.streamline_splines[spline_idx]
            
            # Get all intersection points for this spline, sorted by parameter t
            intersections = sorted(spline_intersections[spline_idx], key=lambda x: x[1])
            
            # Create edges between consecutive intersection points on the same spline
            for i in range(len(intersections) - 1):
                point_idx1, t1 = intersections[i]
                point_idx2, t2 = intersections[i + 1]
                
                # Skip if they're too close
                if abs(t2 - t1) < 0.001:
                    continue
                    
                # Add the edge indices
                edges_list.append([point_idx1, point_idx2])
                
                # Generate points along this edge segment
                num_points = max(10, int((t2 - t1) * 50))  # Adjust number of points based on parameter length
                t_values = np.linspace(t1, t2, num_points)
                edge_segment_points = np.array(splev(t_values, tck)).T
                edge_points.append(edge_segment_points)
        
        # Convert to NumPy array and transpose to get [2 x n_edges]
        if edges_list:
            edges_subdomain = np.array(edges_list).T
        else:
            edges_subdomain = np.zeros((2, 0), dtype=int)
        
        return edges_subdomain, nodes_subdomain, edge_points
        
