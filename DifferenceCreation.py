# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:09:27 2023

@author: jonas
"""

import numpy as np
from PIL import Image
import CausticLib as CL

im = Image.open('lena.png')
im_grey = im.convert('L') # convert the image to *greyscale*
im_arr = np.array(im_grey)

w = 0.1; h = 0.1 # real dimentions, like meters
sub_w, sub_h = np.shape(im_arr)

#creation of the grid points to be given to the object Grid
x, y = np.meshgrid(np.linspace(0., w, sub_w), np.linspace(0., h, sub_h))
grid_pts = np.stack((x,y), axis=2)

img_grid = CL.Grid(grid_pts)

cell_areas = img_grid.cell_areas
total_area = np.sum(cell_areas)

norm_img = C = im_arr/np.sum(im_arr)
D = cell_areas/total_area - C

CL.plot_hmap(D,"Difference", save = "Difference/D_plot.png")

np.save("Difference/D_testing.npy", D)