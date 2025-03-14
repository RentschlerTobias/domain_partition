from scipy.interpolate import make_interp_spline, splprep, splev
import numpy as np

class StreamlineSimplificator:
    def __init__(self,mesh):
        self.mesh = mesh
        self.streamlines_point_array = mesh.streamlines
        self.streamline_splines = self.get_streamlines_as_splines(self.streamlines_point_array)

    def get_streamlines_as_splines(self,streamline_points):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        splines = []
        streamlines = self.mesh.streamlines
        for i in range(len(streamlines)):
           streamline = np.array(streamlines[i])  
           x = streamline[:, 0]
           y =streamline[:, 1]
           if x.size > 5:
               tck, u = splprep([x, y], s=0)
               splines.append([tck,u])
        return splines
