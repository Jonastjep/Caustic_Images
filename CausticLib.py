# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:25:44 2023

@author: jonas
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection


#In this grid system, we create an area for each point of the grid, with the exception of the last column and row.
class Grid:
    def __init__(self, grid_pts):        
        self.pts = grid_pts
        self.w, self.h, d = np.shape(self.pts)
        
        
    def plot_grid(self, title = "", save = False):
        fig,ax = plt.subplots(figsize=(10,10))
        grid_lat = self.pts.transpose(1,0,2)

        plotsArr_pts = self.pts.reshape((-1,2)).T

        plt.scatter(plotsArr_pts[0], plotsArr_pts[1])
        plt.gca().add_collection(LineCollection(self.pts))
        plt.gca().add_collection(LineCollection(grid_lat))
        
        ax.set_title(title)
        
        if save:
            plt.savefig(save, dpi=1200)
        
        
        plt.show()
        
    @property
    def cells(self):
        cell_arr = []
        for i in range(self.w-1):
            cell_arr.append([self.Cell([i,j],self.pts) for j in range(self.h-1)])
        return cell_arr
    
    @property
    def cell_areas(self):
        cs = self.cells
        cell_a = np.zeros((self.w,self.h))
        for i in range(self.w):
            for j in range(self.h):
                cell_a[i][j] = cs[i-1][j-1].area
        return cell_a
    
#     @property
#     def total_area(self):
#         return sum([sum([c.area for c in subArr]) for subArr in self.cells])
        
        
    class Cell:
        def __init__(self, anchor_index, pts):
            self.i, self.j = anchor_index
            
            #clockwise direction
            self.vertex_inds = [
                (self.i,self.j),
                (self.i+1,self.j),
                (self.i+1,self.j+1),
                (self.i,self.j+1)
            ]
            
            self.vertex_pos = np.array([
                pts[self.i][self.j],
                pts[self.i+1][self.j],
                pts[self.i+1][self.j+1],
                pts[self.i][self.j+1]
            ])
            
            self.area = self._PolyAreas(self.vertex_pos)
        
        def _PolyAreas(self, poly):
            x = poly[:,0]
            y = poly[:,1]
            return (0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1))))
        
        
###################### PDE SOLVING ############################
def poisson_iteration_solver(D, h, method, maxIt = 200):
    phi = np.zeros_like(D)
    
    for i in range(maxIt):
        phi = method(phi,-D,h)
        
    return -phi

    
def Jacobi_method(phi_matrix, D, h):
    width,height = np.shape(phi_matrix)
    phi = _phi = np.copy(phi_matrix)
    
    for i in range(width):
        for j in range(height):

            if( i == 0 or j == 0 or i == width-1 or j == height-1):
                continue

            phi[i][j] = (_phi[i-1][j] + _phi[i+1][j] + _phi[i][j-1] + _phi[i][j+1] - (h**2)*D[i][j])/4
    
    
    return phi


        
##################### PLOTTING FUNCTIONS ##################################
lowest = 0.25
cdict = {'red':   [[0.0,  1.0, 1.0],
                   [0.5,  lowest, lowest],
                   [1.0,  lowest, lowest]],
         'green': [[0.0,  lowest, lowest],
                   [1.0,  lowest, lowest]],
         'blue':  [[0.0,  lowest, lowest],
                   [0.5,  lowest, lowest],
                   [1.0,  1.0, 1.0]]}

cmap_lsc = LinearSegmentedColormap("",cdict)

def plot_linearmap(cdict):
    newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
    rgba = newcmp(np.linspace(0, 1, 256))
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    col = ['r', 'g', 'b']
    for xx in [0.25, 0.5, 0.75]:
        ax.axvline(xx, color='0.7', linestyle='--')
    for i in range(3):
        ax.plot(np.arange(256)/256, rgba[:, i], color=col[i])
    ax.set_xlabel('index')
    ax.set_ylabel('RGB')
    ax.set_title("Linear Map RGB vs. Index")
    plt.show()

# plot_linearmap(cdict)

def plot_hmap(input_img, title, minmax = False, save = False):
    if minmax:
        vmin, vmax = minmax
        fig, ax = plt.subplots(figsize=(10,10))
        shw = ax.imshow(input_img, cmap=cmap_lsc, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(shw)
        if save:
            plt.savefig(save, dpi=1200)
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(10,10))
        shw = ax.imshow(input_img, cmap=cmap_lsc)
        ax.set_title(title)
        plt.colorbar(shw)
        if save:
            plt.savefig(save, dpi=1200)
        plt.show()
    
    