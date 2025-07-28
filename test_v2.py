
from torch_geometric.data import Data
import torch
from tools.plotting_tools import *
import multiprocessing as mp
from data_generator import get_mesh    
import matplotlib.pyplot as plt
import numpy as np
### Generate a new mesh


def get_new_mesh():
    is_valid = False
            
    while is_valid == False:
        block_mesh = mp.Pool(1).apply_async(get_mesh).get(timeout=300)
                       
        if block_mesh is not None:
            is_valid = True 

    file = f"./figures/streamlines/quad_mesh_new.png"
    plot_final_mesh(block_mesh,output_file=file)
    
    file_streamlines = f"./figures/streamlines/streamlines_new_mesh.png"
    plot_streamlines(block_mesh,output_file=file_streamlines)

    return block_mesh


def load_mesh():
    extension = 'post_processing' 
    path = f'./saved_meshes/checkpoints_test/mesh_{extension}.pt'

    block_mesh = torch.load(path)
    return block_mesh


def streamline_post_processing(mesh):

    tol = 10e-3
    tol_2 = 10e-3
    streamlines = mesh.streamlines
    singularities_coords = mesh.singularities_coords
    
    streamlines_post_processed = []

    for streamline in streamlines:
        streamline_post_processed = Streamline()
        start = streamline[0]
        is_streamline_cutted = False
        for key, coords_singularity in singularities_coords.items(): 
            
            distance_singularity = np.linalg.norm(start-coords_singularity)
            # Check if singularity is start point of streamline
            if distance_singularity < tol: 
                streamline_post_processed.singularity_out = key
                dx = streamline[1,0]-streamline[0,0]
                dy = streamline[1,1]-streamline[0,1]
                streamline_post_processed.angle_out = np.arctan2(dy,dx)

                print('streamline starts at singularity')
                break
            else:
                distance = np.linalg.norm(streamline-coords_singularity,axis=1)
                distance_min_idx = np.argmin(distance)
                distance_min = distance[distance_min_idx]
                
                if distance_min < tol_2:
                    streamline_post_processed.singularity_in = key
                    dx = streamline[distance_min_idx,0]-streamline[distance_min_idx-1,0]
                    dy = streamline[distance_min_idx,1]-streamline[distance_min_idx-1,1]
                    streamline_post_processed.angle_in = np.arctan2(dy,dx)
                    new_streamline =streamline[:distance_min_idx,:] 
                    new_streamline = np.append(new_streamline,coords_singularity)
                    streamline_post_processed.nodes = new_streamline
                    is_streamline_cutted = True
                    print('streamline cutted')
                    break
        if is_streamline_cutted == False:
                    streamline_post_processed.nodes = streamline
        streamlines_post_processed.append(streamline_post_processed)

    return streamlines_post_processed

class Streamline:
    def __init__(self):

        self.nodes = None
        self.singularity_out = None
        self.singularity_in  = None
        self.angle_out = None
        self.angle_in = None
        

