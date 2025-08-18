from torch_geometric.data import Data
from tools.streamline_merging import StreamlineMerging
from tools.streamline_intersection_splitter import StreamlineIntersectionSplitter
from tools.streamlines_to_quad_faces import QuadFaceGenerator

class StreamlinePostProcessor:

    def __init__(self, mesh: Data, verbose: bool = True):

        self.mesh                       = mesh
        streamlineMerging               = StreamlineMerging(self.mesh, verbose=verbose)
        new_streamlines                 = streamlineMerging.new_streamlines

        splitter                        = StreamlineIntersectionSplitter(offset_boundingBox=0.05, num_samples=5)
        updated_streamlines             = splitter.process_streamlines(new_streamlines)
        faceGenerator                   = QuadFaceGenerator(updated_streamlines)

        self.faces, self.edge_to_streamline,self.edge_index, self.nodes = faceGenerator.get_data()
        
