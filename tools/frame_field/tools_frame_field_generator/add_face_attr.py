def add_face_attr(self):

    nodes      = self.mesh.x
    num_nodes  = nodes.size(0)
    num_faces  = self.mesh.faces.size(1)
    faces_attr = torch.zeros(num_faces)
    for i in range(num_nodes):
        node = nodes[i,:]
        if node[2]!=2:
            faces_of_node_i = self.mesh.nodes_faces_ids[i]

            for j in range(len(faces_of_node_i)):
                face_id             = faces_of_node_i[j]
                faces_attr[face_id] = 1

    self.mesh.face_attr = faces_attr

       self.mesh.frame_field_coords = frame_field_coords
