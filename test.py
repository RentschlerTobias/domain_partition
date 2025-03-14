from torch_geometric.data import Data
import torch
from tools.plotting_tools import *

mesh = torch.load('good_mesh.pt')
plot_streamlines(mesh)
