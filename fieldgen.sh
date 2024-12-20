!#/bin/bash

source ~/strent/environments/python/ai_env/bin/activate

python fieldgen_pre.py
../fieldgen/fieldgen mesh.obj mesh_out.obj --degree=4 --alignToBoundary --s=0 
python fieldgen_post.py

