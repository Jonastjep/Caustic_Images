# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:42:36 2023

@author: jonas
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import CausticLib as CL

im = Image.open('lena.png')
im_grey = im.convert('L') # convert the image to *greyscale*
im_arr = np.array(im_grey)

plt.imshow(im_arr, cmap="gray")

w = 0.1; h = 0.1 # real dimentions, like meters
sub_w, sub_h = np.shape(im_arr)

#creation of the grid points to be given to the object Grid
x, y = np.meshgrid(np.linspace(0., w, sub_w), np.linspace(0., h, sub_h)) 
grid_pts = np.stack((x,y), axis=2)

img_grid = CL.Grid(grid_pts)

cell_areas = img_grid.cell_areas
total_area = np.sum(cell_areas)
print(total_area)

#CREATE D MATRIX
norm_img = C = im_arr/np.sum(im_arr)
D = cell_areas/total_area - C
CL.plot_hmap(D,"Difference",(-0.000003,0.000003))

#CREATE PHI MATRIX
h = w/sub_w
phi = CL.poisson_iteration_solver(D, h, CL.Jacobi_method, 200)

CL.plot_hmap(phi,"Phi")

