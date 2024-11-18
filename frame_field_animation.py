from tools import MeshGenerator, FrameField
from manim import *
import numpy as np
import torch


class FrameFieldAnimation(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.camera.background_color = WHITE  # Set background color to white

    def construct(self):
        self.scale = 0.5
        mesh_gen = MeshGenerator(
            lc=0.5, minBoundary=-5, maxBoundary=5, seed=42)
        frame_field = FrameField(mesh_gen.mesh)
        self.mesh = frame_field.mesh

        self.nodes = self.mesh.x
        self.nodes[:, 2] = 0  # Convert 2D to 3D
        self.u_all_iter = frame_field.u_all

        # Step 1: Animate the nodes
        node_dots = []
        for i in range(self.nodes.size(0)):
            node = self.nodes[i, :].numpy()
            dot = Dot(node, color=BLUE)
            node_dots.append(dot)

        node_group = VGroup(*node_dots)
        self.play(FadeIn(node_group), run_time=2)

        # Step 2: Animate the edges
        edge_lines = []
        edge_animations = []
        for i in range(self.mesh.edge_index.size(1)):
            start_idx = self.mesh.edge_index[0, i]
            end_idx = self.mesh.edge_index[1, i]

            start_node = self.nodes[start_idx, :].numpy()
            end_node = self.nodes[end_idx, :].numpy()

            color = BLUE if self.mesh.edge_attr[0, i] == 1 else WHITE
            line = Line(start_node, end_node, color=color,
                        stroke_opacity=0.3)  # Transparent edges
            edge_lines.append(line)
            edge_animations.append(Create(line))

        edge_group = VGroup(*edge_lines)
        # Animate edges with delay
        self.play(FadeIn(edge_group), run_time=2)

        # Step 3: Animate the vector field
        vectors_init = self.create_vector_field(self.u_all_iter[0])
        vector_group_init = VGroup(*vectors_init)

        vectors_final = self.create_vector_field(self.u_all_iter[-1])
        vector_group_final = VGroup(*vectors_final)

        self.add(node_group, edge_group, vector_group_init)

        self.play(Transform(vector_group_init,
                  vector_group_final), run_time=2)
        self.wait(1)

    def create_vector_field(self, vectors):
        """Create a list of Arrow objects representing the vector field."""
        arrow_list = []

        for n in range(self.nodes.size(0)):
            vec = torch.tensor([vectors[n, 0], vectors[n, 1], 0])  # 2D to 3D
            start = self.nodes[n, :]

            end = start + self.scale * vec
            arrow = Arrow(
                start.numpy(),
                end.numpy(),
                buff=0,
                color=RED,
                stroke_opacity=0.7,  # Slight transparency for a modern look
                max_stroke_width_to_length_ratio=5,
            )
            arrow_list.append(arrow)

        return arrow_list
