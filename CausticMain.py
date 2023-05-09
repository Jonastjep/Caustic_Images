# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:42:36 2023

@author: jonas
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import CausticLib as CL
from DifferenceCreation import compute_D
from PhiCreation import compute_Phi

run_computation = False

im = Image.open('lena.png')
im_grey = im.convert('L') # convert the image to *greyscale*
im_arr = np.array(im_grey)

fig,ax = plt.subplots(figsize=(10,10),dpi=500)
plt.imshow(im_arr, cmap="gray")
plt.show()

w = 0.1; h = 0.1 # real dimentions, like meters
sub_w, sub_h = np.shape(im_arr)

#creation of the grid points to be given to the object Grid
x, y = np.meshgrid(np.linspace(0., w, sub_w), np.linspace(0., h, sub_h)) 
grid_pts = np.stack((x,y), axis=2)

img_grid = CL.Grid(grid_pts)

# ############ CREATING D AND PHI ###################################
# if run_computation:

#     D = compute_D(im_arr, w, h, sub_w, sub_h, img_grid)
#     CL.plot_hmap(D,"Difference",(-0.000003,0.000003))
    
#     #CREATE PHI MATRIX
#     h = w/sub_w
#     nb_it = 200
    
#     phi = compute_Phi(D, nb_it)
#     CL.plot_hmap(phi,"Phi")
    
# else:
#     D = np.load("Difference/D_testing.npy")
#     phi = np.load("Phi/Phi_testing_2000.npy")
    
#     CL.plot_hmap(D,"Difference",(-0.000003,0.000003))
#     CL.plot_hmap(phi,"Phi")
    
# #####################################################################


for i in range(25):
    D = compute_D(im_arr, w, h, sub_w, sub_h, img_grid, i=str(i))
    
    h = w/sub_w
    h=1
    nb_it = 500
    
    phi = compute_Phi(-D, nb_it, h, str(i))
    
    
    v = CL.gradient(phi)
    np.save(f"VectorField{i}.npy", v)
    CL.plot_vectField(x,y,v,title="V VectField",save=f"VectorField/VectorField{i}.png")
    
    dt = (np.nanmin(img_grid.min_dist_dt(v)))/2
    
    img_grid.pts += v*dt
    
    img_grid.plot_grid(lineW=0.25,markerS=0, save=f"Grid/Grid{i}.png")
    
