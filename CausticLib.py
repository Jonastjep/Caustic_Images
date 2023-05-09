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
        
        
    def plot_grid(self, title = "", save = False, lineW=2., markerS=4., dpi=500):
        fig,ax = plt.subplots(figsize=(10,10),dpi=dpi)
        grid_lat = self.pts.transpose(1,0,2)

        plotsArr_pts = self.pts.reshape((-1,2)).T

        plt.scatter(plotsArr_pts[0], plotsArr_pts[1], s=markerS)
        plt.gca().add_collection(LineCollection(self.pts,linewidth=lineW))
        plt.gca().add_collection(LineCollection(grid_lat,linewidth=lineW))
        
        plt.gca().invert_yaxis()
        
        ax.set_title(title)
        
        if save:
            plt.savefig(save, dpi=dpi)
        else:
            plt.show()
        
    def min_dist_dt(self, v):
        
        def single_dt(p1,p2,p3,v1,v2,v3):
            u = vect(v2 - v1)
            v = vect(v3 - v1)
            p12 = vect(p2 - p1)
            p13 = vect(p3 - p1)
            
            a = u.x*v.y - u.y*v.x
            b = u.x*p13.y + v.y*p12.x - u.y*p13.x - v.x*p12.y
            c = p12.x*p13.y - p12.y*p13.x
            
            delta = b**2-4*a*c
            
            if delta >= 0:
                ts = np.array([(-b+np.sqrt(delta))/(2*a),(-b-np.sqrt(delta))/(2*a)])
                ts[ts<=0] = np.nan
                return np.nanmin(ts)
            
            elif delta == 0:
                t = -b/a
                if t <= 0:
                    return np.Nan
                else:
                    return -b/a
            else:
                return np.NaN
        
        dt = np.zeros((self.w-1, self.h-1, 2))
        
        for i in range(self.w-1):
            for j in range(self.h-1):
                dt[i,j,0] = single_dt(self.pts[i,j],self.pts[i+1,j],self.pts[i+1,j+1],v[i,j],v[i+1,j],v[i+1,j+1])
                dt[i,j,1] = single_dt(self.pts[i,j],self.pts[i,j+1],self.pts[i+1,j+1],v[i,j],v[i,j+1],v[i+1,j+1])
                
        return dt
        
        
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
        phi = method(phi,D,h)
        
    return phi

    
def Jacobi_method(phi_matrix, D, h):
    width,height = np.shape(phi_matrix)
    phi = _phi = np.copy(phi_matrix)
    
    for i in range(width):
        for j in range(height):

            if( i == 0 or j == 0 or i == width-1 or j == height-1):
                continue

            phi[i][j] = (_phi[i-1][j] + _phi[i+1][j] + _phi[i][j-1] + _phi[i][j+1] - (h**2)*D[i][j])/4
    
    
    return phi

def mag(a, b):
    return np.linalg.norm(a-b)

class vect:
    def __init__(self, v):
        self.x = v[0]
        self.y = v[1]
        
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

def plot_hmap(input_img, title='', minmax = False, save = False, dpi = 500):
    if minmax:
        vmin, vmax = minmax
        fig, ax = plt.subplots(figsize=(10,10),dpi = dpi)
        shw = ax.imshow(input_img, cmap=cmap_lsc, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(shw)
        if save:
            plt.savefig(save, dpi=dpi)
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(10,10), dpi=dpi)
        shw = ax.imshow(input_img, cmap=cmap_lsc)
        ax.set_title(title)
        plt.colorbar(shw)
        if save:
            plt.savefig(save, dpi=dpi)
        plt.show()
    
def plot_vectField(x,y,v, arr_nb=5, dpi=500, save=False, title=""):
    fig, ax = plt.subplots(figsize=(10,10),dpi = dpi)
    ax.set_title(title)
    
    plt.quiver(x[::arr_nb,::arr_nb],y[::arr_nb,::arr_nb],v[:,:,0][::arr_nb,::arr_nb],v[:,:,1][::arr_nb,::arr_nb])
    
    plt.gca().invert_yaxis()
    
    if save:
        plt.savefig(save, dpi=dpi)
        
    
    plt.show()
    
    
def gradient(phi):
    N, M = np.shape(phi)
    grad_x, grad_y = np.zeros((N,N)), np.zeros((N,N))
    
    # #less efficient
    # for i in range(N):
    #     for j in range(M):
            
    #         if i<N-1:
    #             grad_x[i,j] = phi[i+1,j] - phi[i,j]
    #         if j<M-1:
    #             grad_y[i,j] = phi[i,j+1] - phi[i,j]
                
    #more efficient
    grad_x[:,1:N-1] = phi[:,1:N-1] - phi[:,:N-2]
    grad_y[1:N-1,:] = phi[1:N-1,:] - phi[:N-2,:]
    grad = np.zeros((N,N,2))
    grad[:,:,0] = grad_x; grad[:,:,1] = grad_y
            
    grad = np.zeros((N,N,2))
    grad[:,:,0] = grad_x; grad[:,:,1] = grad_y
    return grad

def divergence(N_xy):
    N, N, d = np.shape(N_xy)
    div = np.zeros((N,N))
    div[1:,1:] = N_xy[1:,1:,0]-N_xy[:N-1,1:,0] + N_xy[1:,1:,1]-N_xy[1:,:N-1,1]
    div[N-1,:] = np.zeros_like(div[N-1,:])
    div[:,N-1] = np.zeros_like(div[:,N-1])
    return div