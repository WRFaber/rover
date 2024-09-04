import torch
import numpy as np

class Terrain:
    def __init__(self, n, m, exit_pos, figure_pos):
        self.n = n
        self.m = m
        self.exit_pos = exit_pos
        self.figure_pos = figure_pos

    def move(self, direction):
        x, y = self.figure_pos
        if direction == "up":
            if y < self.n-1:
                self.figure_pos = (x, y+1)
        elif direction == "down":
            if y > 0:
                self.figure_pos = (x, y-1)
        elif direction == "left":
            if x > 0:
                self.figure_pos = (x-1, y)
        elif direction == "right":
            if x < self.m-1:
                self.figure_pos = (x+1, y)

    def is_at_exit(self):
        return self.figure_pos == self.exit_pos

    def get_state(self, device):
        return torch.FloatTensor(self.figure_pos).unsqueeze(0).to(device)
    
