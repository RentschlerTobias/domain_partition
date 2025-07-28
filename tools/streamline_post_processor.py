import torch
import numpy as np
from collections import defaultdict

class StreamlinePostProcessor:
    def __init__(self, simplificator):
        """Takes the StreamlineSimplificator object and processes its streamlines"""
        self.simplificator = simplificator
        self.mesh = simplificator.mesh
        self.original_streamlines = simplificator.mesh.streamlines
        self.singularities = self.extract_singularities()
        
    def extract_singularities(self):
        """Extract singularity information from the mesh"""
        singularities = {}
        for i, singularity_type in enumerate(self.mesh.singularities):
            if singularity_type != 0:
                singularities[i] = {
                    'coords': torch.tensor(self.mesh.singularities_coords[i], dtype=torch.float32),
                    'type': singularity_type,
                    'face_id': i
                }
        return singularities
    
    def run_post_processing(self):
        """
        Main post-processing pipeline
        Returns: corrected_streamlines (list) or None if no corrections applied
        """
        # Analyze connectivity
        streamline_connectivity = self.analyze_streamline_connectivity()
        
        # Apply corrections
        corrected_streamlines, corrections_applied = self.apply_streamline_corrections(streamline_connectivity)
        
        if corrections_applied > 0:
            print(f'    Applied {corrections_applied} corrections')
            return corrected_streamlines
        else:
            return None  # No corrections needed
    
    def analyze_streamline_connectivity(self):
        """Analyze how streamlines connect singularities"""
        streamline_connectivity = defaultdict(list)
        tolerance = 0.05
        
        for i, streamline in enumerate(self.original_streamlines):
            if len(streamline) < 2:
                continue
                
            start_point = torch.tensor(streamline[0], dtype=torch.float32)
            end_point = torch.tensor(streamline[-1], dtype=torch.float32)
            
            start_singularity = self.find_closest_singularity(start_point, tolerance)
            end_singularity = self.find_closest_singularity(end_point, tolerance)
            
            # Store connectivity
            if start_singularity is not None:
                streamline_connectivity[start_singularity].append({
                    'streamline_id': i,
                    'direction': 'out',
                    'other_end': end_singularity,
                    'streamline': streamline
                })
            
            if end_singularity is not None and end_singularity != start_singularity:
                streamline_connectivity[end_singularity].append({
                    'streamline_id': i,
                    'direction': 'in',
                    'other_end': start_singularity,
                    'streamline': streamline
                })
        
        return streamline_connectivity
    
    def apply_streamline_corrections(self, streamline_connectivity):
        """Apply corrections based on the three cases from Xiao et al."""
        corrected_streamlines = list(self.original_streamlines)  # Start with copy
        corrections_applied = 0
        processed_ids = set()
        
        # Process pairs of singularities
        for s1_id in self.singularities.keys():
            for s2_id in self.singularities.keys():
                if s1_id >= s2_id:
                    continue
                
                # Find connections between these singularities
                s1_to_s2 = self.find_connecting_streamlines(streamline_connectivity, s1_id, s2_id, 'out')
                s2_to_s1 = self.find_connecting_streamlines(streamline_connectivity, s2_id, s1_id, 'out')
                
                # Case 1: Bidirectional connection - merge them
                if len(s1_to_s2) > 0 and len(s2_to_s1) > 0:
                    try:
                        merged_streamline = self.simplificator.interpolate_streamlines(
                            s1_to_s2[0]['streamline'], s2_to_s1[0]['streamline']
                        )
                        
                        # Replace both streamlines with merged one
                        id1 = s1_to_s2[0]['streamline_id']
                        id2 = s2_to_s1[0]['streamline_id']
                        
                        if id1 not in processed_ids and id2 not in processed_ids:
                            corrected_streamlines[id1] = merged_streamline
                            # Mark second streamline for removal (set to None)
                            corrected_streamlines[id2] = None
                            processed_ids.add(id1)
                            processed_ids.add(id2)
                            corrections_applied += 1
                            print(f'      Case 1: Merged streamlines {id1} and {id2}')
                            
                    except Exception as e:
                        print(f'      Case 1 failed: {e}')
        
        # Remove None streamlines
        corrected_streamlines = [s for s in corrected_streamlines if s is not None]
        
        return corrected_streamlines, corrections_applied
    
    def find_connecting_streamlines(self, streamline_connectivity, start_sing, end_sing, direction):
        """Find streamlines connecting two specific singularities"""
        connections = streamline_connectivity[start_sing]
        return [conn for conn in connections 
                if conn['direction'] == direction and conn['other_end'] == end_sing]
    
    def find_closest_singularity(self, point, tolerance=0.05):
        """Find closest singularity within tolerance"""
        min_distance = float('inf')
        closest_singularity = None
        
        for sing_id, sing_info in self.singularities.items():
            distance = torch.norm(point - sing_info['coords'])
            if distance < min_distance and distance < tolerance:
                min_distance = distance
                closest_singularity = sing_id
        
        return closest_singularity
