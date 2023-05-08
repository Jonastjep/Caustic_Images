# -*- coding: utf-8 -*-
"""
Created on Mon May  8 15:14:43 2023

@author: jonas
"""
import numpy as np
import CausticLib as CL

def compute_Phi(D, nb_it, h, i=""):

    phi = CL.poisson_iteration_solver(D, h, CL.Jacobi_method, nb_it)
    
    CL.plot_hmap(phi,f"Phi {i}",save = f"Phi/Phi_plot{i}.png")
    
    np.save(f"Phi/Phi_testing{i}.npy", phi)
    
    return phi
    
def main():
    
    D = np.load("Difference/D_testing.npy")
    nb_it = 200
    
    w = 0.1; h = 0.1 # real dimentions, like meters
    sub_w, sub_h = np.shape(D)
    
    h = w/sub_w
    
    compute_Phi(D, nb_it, h)