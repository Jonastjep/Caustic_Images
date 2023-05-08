# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:29:39 2023

@author: jonas
"""
import numpy as np
import CausticLib as CL

w = 4; h = 4 # real dimentions, like meters
sub_w = 5; sub_h = 5 # how many elements

#creation of the grid points to be given to the object Grid
x, y = np.meshgrid(np.linspace(0., w, sub_w), np.linspace(0., h, sub_h))
grid = np.stack((x,y), axis=2)

g = CL.Grid(grid)

print(f'Cell areas of the grid 1:\n {g.cell_areas}')
g.plot_grid("Grid 1: Straight Grid")

g.pts = g.pts + np.random.rand(g.w,g.h,2)/5
print(f'Cell areas of the grid 2:\n {g.cell_areas}')
g.plot_grid("Grid 2: Added Randomness")