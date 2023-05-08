# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:14:43 2023

@author: jonas
"""
import numpy as np
import CausticLib as CL

D = np.load("Difference/D_testing.npy")

w = 0.1; h = 0.1 # real dimentions, like meters
sub_w, sub_h = np.shape(D)

h = w/sub_w
nb_it = 2000
phi = CL.poisson_iteration_solver(D, h, CL.Jacobi_method, nb_it)

CL.plot_hmap(phi,"Phi",save = "Phi/Phi_plot.png")

np.save("Phi/Phi_testing.npy", phi)