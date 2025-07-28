from tools.plotting_tools import *
from torch_geometric.data import Data
import numpy as np
import torch
import os
import multiprocessing as mp


from test_v2 import load_mesh
from test_v2 import streamline_post_processing 

block_mesh= load_mesh()
new_streamlines = streamline_post_processing(block_mesh)
plot_post_processed_streamline(new_streamlines)

new_streamlines[-1].nodes

new_streamlines[-1].singularity_out
new_streamlines[-1].singularity_in

for streamline in block_mesh.streamlines:
    print(streamline.shape[1])

for streamline in new_streamlines:
    print(streamline.nodes.shape)
