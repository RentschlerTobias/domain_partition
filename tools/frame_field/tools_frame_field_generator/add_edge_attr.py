
def add_edge_attr(self):
    edges         = self.mesh.edge_index
    numberOfEdges = edges.size()[1]
    edge_attr     = torch.zeros([1,numberOfEdges],dtype=torch.long)

    minBoundary = torch.min(self.mesh.x[:,0])
    maxBoundary = torch.max(self.mesh.x[:,0])

    for i in range(numberOfEdges):
        node0    = self.mesh.x[edges[0,i],:]
        node1    = self.mesh.x[edges[1,i],:]
        deltaX   = node0[0]-node1[0]
        deltaY   = node0[1]-node1[1]
        dimNode0 = node0[2] 
        dimNode1 = node1[2] 

        if dimNode0 == 2 or dimNode1 == 2:
            edge_attr[0,i] = 0
        else:
            if node0[0]==minBoundary or node0[0]==maxBoundary or node0[1]==minBoundary or node0[1]==maxBoundary or node1[0]==minBoundary or node1[0]==maxBoundary or node1[1]==minBoundary or node1[1]==maxBoundary:
                if deltaX!= 0 and deltaY !=0:
                    edge_attr[0,i] = 0
                else:
                    edge_attr[0,i] = 1

            else:
                edge_attr[0,i] = 1


    self.mesh.edge_attr=edge_attr


