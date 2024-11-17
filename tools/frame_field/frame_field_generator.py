from tools_frame_field_generator.pre_processing import pre_processing
import sys
from pathlib import Path


current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / "tools_frame_field_generator"))


class FrameField:
    def __init__(self, meshOfMeshGenerator):

        self.activate_Debug_Comments = True
        self.mesh = meshOfMeshGenerator
        self.pre_processing()

    def pre_precessing(self):
        pre_processing(self)
