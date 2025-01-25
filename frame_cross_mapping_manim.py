from manim import *
import numpy as np
import torch

from tools import MeshGenerator_v2, FrameField, NACA_airfoil


class CrossFieldAnimation(Scene):
    def construct(self):
        # Replace with actual mesh data
        init = True
        airfoil = NACA_airfoil()
        # mesh_gen = MeshGenerator_v2(airfoil, quadMesh=True, lc=0.05)
        mesh_gen = MeshGenerator_v2(airfoil, quadMesh=False, lc=0.05)

        frameField = FrameField(mesh_gen.mesh)
        mesh = frameField.mesh

        if init:
            boundary_mask = mesh['x'][:, 2] != 2
            coords = mesh['x'][boundary_mask, 0:2]
            frames = mesh['frame_field'][boundary_mask, :]
        else:
            coords = mesh['x'][:, 0:2]
            frames = mesh['frame_field']

        # Calculate cross field
        num_nodes = coords.shape[0]
        frame_field = np.arctan2(frames[:, 1], frames[:, 0]) % (2 * np.pi)
        cross_field = np.zeros((num_nodes, 4))
        base_angle = frame_field / 4
        cross_field[:, 0] = base_angle
        cross_field[:, 1] = base_angle + np.pi / 2
        cross_field[:, 2] = base_angle + np.pi
        cross_field[:, 3] = base_angle + 3 * np.pi / 2

        # Display mesh
        dots = self.display_nodes(coords)
        self.play(Create(dots))

        # Display initial cross field
        cross_vectors = self.get_cross_vectors(coords, cross_field)
        self.play(Create(cross_vectors))

        # Transform to frame field
        frame_vectors = self.get_frame_vectors(coords, frames)
        self.play(Transform(cross_vectors, frame_vectors))

        self.wait()

    def display_nodes(self, coords):
        # Display mesh nodes as dots
        dots = VGroup(*[Dot(point=[x, y, 0]) for x, y in coords])
        return dots

    def get_cross_vectors(self, coords, cross_field):
        # Create arrows for the cross field
        arrows = VGroup()
        for (x, y), angles in zip(coords, cross_field):
            for angle in angles:
                dx, dy = 0.3 * np.cos(angle), 0.3 * np.sin(angle)
                arrows.add(Arrow(start=[x, y, 0], end=[
                           x + dx, y + dy, 0], buff=0, color=BLUE))
        return arrows

    def get_frame_vectors(self, coords, frames):
        # Create arrows for the frame field
        arrows = VGroup()
        for (x, y), (dx, dy) in zip(coords, frames):
            arrows.add(Arrow(start=[x, y, 0], end=[
                       x + 0.3 * dx, y + 0.3 * dy, 0], buff=0, color=RED))
        return arrows
