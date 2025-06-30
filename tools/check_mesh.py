import torch

class MeshCheck:
    def __init__(self,tri_mesh,quad_mesh,tol=1e-6):



        tri_vertices = tri_mesh.x[:, 0:2]
        tri_faces = tri_mesh.faces.T
        tri_nodes = tri_vertices[tri_faces]
        A = tri_nodes[:, 0, :]
        B = tri_nodes[:, 1, :]
        C = tri_nodes[:, 2, :]
        AB = B - A
        AC = C - A
        cross_product = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
        self.tri_area = torch.sum(torch.abs(cross_product) / 2.0)

        quad_vertices = quad_mesh.x[:, 0:2]
        quad_faces = quad_mesh.faces.T
        quad_nodes = quad_vertices[quad_faces]
        A = quad_nodes[:, 0, :]
        B = quad_nodes[:, 1, :]
        C = quad_nodes[:, 2, :]
        D = quad_nodes[:, 3, :]
        AB = B - A
        AD = D - A
        BC = C - B
        cross1 = AB[:, 0] * AD[:, 1] - AB[:, 1] * AD[:, 0]
        cross2 = BC[:, 0] * AD[:, 1] - BC[:, 1] * AD[:, 0]
        self.quad_area = torch.sum(torch.abs(cross1 + cross2) / 2.0)
        
        if torch.abs(self.tri_area - self.quad_area) <= tol:
            self.is_vaild = True
        else:

            self.is_vaild = False 


