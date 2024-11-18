from tools import MeshGenerator, FrameField
from manim import *
import numpy as np
import torch


class FrameFieldAnimation(Scene):
    def construct(self):
        self.scale = 0.5
        mesh_gen = MeshGenerator(
            lc=0.5, minBoundary=-5, maxBoundary=5, seed=42)
        frame_field = FrameField(mesh_gen.mesh)
        self.mesh = frame_field.mesh

        self.nodes = self.mesh.x
        self.nodes[:, 2] = 0
        self.u_all_iter = frame_field.u_all
        edge_lines = []

        for i in range(self.mesh.edge_index.size(1)):
            start_idx = self.mesh.edge_index[0, i]
            end_idx = self.mesh.edge_index[1, i]

            start_node = self.nodes[start_idx, :].numpy()
            end_node = self.nodes[end_idx, :].numpy()

            if self.mesh.edge_attr[0, i] == 1:
                line = Line(start_node, end_node, color=BLUE)
            else:
                line = Line(start_node, end_node, color=WHITE)
            edge_lines.append(line)

        mesh_group = VGroup(*edge_lines)

        vectors_init = self.create_vector_field(self.u_all_iter[0])
        vector_group_init = VGroup(*vectors_init)

        vectors_final = self.create_vector_field(self.u_all_iter[-1])
        vector_group_final = VGroup(*vectors_final)

        self.add(mesh_group, vector_group_init)

        self.play(Transform(vector_group_init, vector_group_final), run_time=2)
        self.wait(1)
    # for frame in range(len(self.u_all_iter)-1):

#        for frame in range(10):
#            vectors_new = self.create_vector_field(self.u_all_iter[frame+1])
#            vector_new_group = VGroup(*vectors_new)
#            self.play(Transform(vector_group, vector_new_group))
#            self.wait(1)
#            vector_group = vector_new_group
#
    def create_vector_field(self, vectors):
        arrow_list = []

        for n in range(self.nodes.size(0)):

            vec = torch.tensor([vectors[n, 0], vectors[n, 1], 0])
            start = self.nodes[n, :]

            end = start + self.scale*vec
            arrow = Arrow(start.numpy(), end.numpy(), buff=0, color=RED)
            arrow_list.append(arrow)
        return arrow_list
