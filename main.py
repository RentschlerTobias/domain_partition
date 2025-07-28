
from tools import StreamlineGenerator_v2
from tools import MeshGenerator, FrameField, NACA_airfoil, StreamlineGenerator
from tools import StreamlineSimplificator
from tools import Transfinite_Interpolation
from tools import MeshCheck
from tools.plotting_tools import *
from tools.save_load import *
from torch_geometric.data import Data
import numpy as np
import torch
import os
import multiprocessing as mp
from test import get_mesh    
from test import extract_mesh_data    



def main():

    number_of_meshes =10
    checkpoint_interval = 2  # Speichere alle x erfolgreiche Meshes
    checkpoint_dir = "./saved_meshes/checkpoints"

    os.makedirs(checkpoint_dir, exist_ok=True)

    database = []
    successful_meshes = 0  
    failed_meshes = 0  

    for n in range(number_of_meshes):
        is_valid = False
        
        while is_valid == False:
            mesh_data = mp.Pool(1).apply_async(get_mesh).get(timeout=600)
                           
            if mesh_data is not None:
                is_valid = True    
                database.append(mesh_data)
                successful_meshes += 1
                print(f"successful meshes: {successful_meshes}")
                
               # if n+1 % checkpoint_interval == 0:
                try:
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_mesh_{successful_meshes}.pt')
                    torch.save(database, checkpoint_path)
                    database = [] #resett database
                    print(f"Checkpoint gespeichert: {checkpoint_path}")
                except Exception as checkpoint_error:
                    print(f"Warnung: Fehler beim Speichern des Checkpoints: {checkpoint_error}")
            else:
                failed_meshes += 1
                print(f"Warning: Transifinite Mesh is not valid")

    print(f'total failed meshes {failed_meshes}; total successful meshes {successful_meshes }') 

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_mesh_{successful_meshes}.pt')
    torch.save(database, checkpoint_path)
    print(f"Final Checkpoint reached")

if __name__ == "__main__":                                                                         
    main()
