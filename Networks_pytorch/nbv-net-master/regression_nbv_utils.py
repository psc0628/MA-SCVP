# Juan Irving Vasquez-Gomez
# jivg.org

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
from torch.autograd import Variable

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    

#### ---------------- Drawing functions ------------------ ####

def showGrid(grid, nbv = None, predicted_nbv = None):
    # receives a plain grid and plots the 3d voxel map
    grid3d = np.reshape(grid, (32,32,32))

    unknown = (grid3d == 0.5)
    occupied = (grid3d > 0.5)

    # combine the objects into a single boolean array
    voxels = unknown | occupied

    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[unknown] = 'yellow'
    colors[occupied] = 'blue'

    # and plot everything
    fig = plt.figure(figsize=(15,15))
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    #ax.axis('equal')
     
    # plot the NBV
    # the view sphere was placed at 0.4 m from the origin, the voxelmap has an aproximated size of 0.25
    scale = 32/2
    rate_voxel_map_sphere = 0.25
    center = np.ones(3) * scale
    
    if nbv is not None:
        position = nbv[:3]
        position = position * (scale / rate_voxel_map_sphere) + center
        direction = center - position
        #print(position)
        ax.quiver(position[0], position[1], position[2], direction[0], direction[1], direction[2], length=9.0, normalize=True, color = 'g')  
    
    if predicted_nbv is not None:
        position = predicted_nbv[:3]
        position = position * (scale / rate_voxel_map_sphere) + center
        direction = center - position
        #print(position)
        ax.quiver(position[0], position[1], position[2], direction[0], direction[1], direction[2], length=9.0, normalize=True, color = 'r')
    
    plt.pause(0.001)  # pause a bit so that plots are updated
    #plt.show()
    

def showGrid4(grid, nbv = None, predicted_nbv = None):
     # receives a plain grid and plots the 3d voxel map
    grid3d = np.reshape(grid, (32,32,32))

    unknown = (grid3d == 0.5)
    occupied = (grid3d > 0.5)

    # combine the objects into a single boolean array
    voxels = unknown | occupied

    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[unknown] = 'yellow'
    colors[occupied] = 'blue'
    
    # plot the NBV
    # the view sphere was placed at 0.4 m from the origin, the voxelmap has an aproximated size of 0.25
    scale = 32/2
    rate_voxel_map_sphere = 0.25
    center = np.ones(3) * scale
    
    if nbv is not None:
        position = nbv[:3]
        position_nbv = position * (scale / rate_voxel_map_sphere) + center
        direction_nbv = center - position_nbv
    if predicted_nbv is not None:
        position = predicted_nbv[:3]
        position_pred = position * (scale / rate_voxel_map_sphere) + center
        direction_pred = center - position_pred    
    
    fig = plt.figure(figsize=(20, 20))
    
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    if nbv is not None:
        ax.quiver(position_nbv[0], position_nbv[1], position_nbv[2], direction_nbv[0], direction_nbv[1], direction_nbv[2], length=9.0, normalize=True, color = 'g') 
    if predicted_nbv is not None:
        ax.quiver(position_pred[0], position_pred[1], position_pred[2], direction_pred[0], direction_pred[1], direction_pred[2], length=9.0, normalize=True, color = 'r')

        
        
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    if nbv is not None:
        ax.quiver(position_nbv[0], position_nbv[1], position_nbv[2], direction_nbv[0], direction_nbv[1], direction_nbv[2], length=9.0, normalize=True, color = 'g') 
    if predicted_nbv is not None:
        ax.quiver(position_pred[0], position_pred[1], position_pred[2], direction_pred[0], direction_pred[1], direction_pred[2], length=9.0, normalize=True, color = 'r')
    ax.view_init(elev=0.0, azim=180.0)

    
    
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    if nbv is not None:
        ax.quiver(position_nbv[0], position_nbv[1], position_nbv[2], direction_nbv[0], direction_nbv[1], direction_nbv[2], length=9.0, normalize=True, color = 'g') 
    if predicted_nbv is not None:
        ax.quiver(position_pred[0], position_pred[1], position_pred[2], direction_pred[0], direction_pred[1], direction_pred[2], length=9.0, normalize=True, color = 'r')
    ax.view_init(elev=0.0, azim=90.0)
    
    
    
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    if nbv is not None:
        ax.quiver(position_nbv[0], position_nbv[1], position_nbv[2], direction_nbv[0], direction_nbv[1], direction_nbv[2], length=9.0, normalize=True, color = 'g') 
    if predicted_nbv is not None:
        ax.quiver(position_pred[0], position_pred[1], position_pred[2], direction_pred[0], direction_pred[1], direction_pred[2], length=9.0, normalize=True, color = 'r')
    ax.view_init(elev=90.0, azim=0.0)
    
    
    
def showScanLocations(grid, scan_locations, elevation = 45, azimut = 45):
    # receives a plain grid and plots the 3d voxel map
    grid3d = np.reshape(grid, (32,32,32))

    unknown = (grid3d == 0.5)
    occupied = (grid3d > 0.5)

    # combine the objects into a single boolean array
    voxels = unknown | occupied

    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[unknown] = 'yellow'
    colors[occupied] = 'blue'

    # and plot everything
    fig = plt.figure(figsize=(15,15))
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')
    #ax.axis('equal')
     
    # plot the NBV
    # the view sphere was placed at 0.4 m from the origin, the voxelmap has an aproximated size of 0.25
    scale = 32/2
    rate_voxel_map_sphere = 0.25
    center = np.ones(3) * scale
    
    for location in scan_locations:
        position = location[:3]
        position = position * (scale / rate_voxel_map_sphere) + center
        direction = center - position
        #print(position)
        ax.quiver(position[0], position[1], position[2], direction[0], direction[1], direction[2], length=9.0, normalize=True, color = 'r')  
    
    ax.view_init(elev=elevation, azim=azimut)
    plt.pause(0.001)  # pause a bit so that plots are updated
    #plt.show()
    
    
    
    
#### ---------------- Transformations ------------------ ####
    
def spherical2cartesian(spherical):
    # receives spherical = (r, yaw, pitch)
    r, yaw, pitch = spherical
    x = r * np.sin(pitch) * np.cos(yaw)
    y = r * np.sin(pitch) * np.sin(yaw)
    z = r * np.cos(pitch)
    return np.array([x,y,z])
    
    
def normPos2Angles(values):
    angles = values * 2 * np.pi
    angles = angles - np.pi
    return angles
    
    
class NBVPredictionDatasetFull(Dataset):
    """NBV dataset."""

    def __init__(self, grid_file, nbv_file, transform=None):
        """
        Args:
            poses_file (string): Path to the sensor poses (next-best-views).
            root_dir (string): Directory with all the grids.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.root_dir = root_dir
        self.grid_data = np.load(grid_file)
        self.pose_data = np.load(nbv_file)
        
        self.transform = transform

    def __len__(self):
        return len(self.pose_data)

    def __getitem__(self, idx):
        #grid_name = os.path.join(self.root_dir, 'grid_' + str(idx) + '.npy')
        grid = self.grid_data[idx] # np.load(grid_name)
        #image = io.imread(img_name)
        pose = self.pose_data[idx]
        #landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'grid': grid, 'nbv': pose}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
class NBVPredictionDataset(Dataset):
    """NBV dataset."""

    def __init__(self, pose_file, root_dir, transform=None):
        """
        Args:
            poses_file (string): Path to the sensor poses (next-best-views).
            root_dir (string): Directory with all the grids.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.pose_data = np.load(pose_file)
        self.transform = transform

    def __len__(self):
        return len(self.pose_data)

    def __getitem__(self, idx):
        grid_name = os.path.join(self.root_dir,
                                'grid_' + str(idx) + '.npy')
        grid = np.load(grid_name)
        #image = io.imread(img_name)
        pose = self.pose_data[idx]
        #landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'grid': grid, 'nbv': pose}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
class To3DGrid(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        grid, nbv = sample['grid'], sample['nbv']

        # swap color axis because
        # numpy image: H i x W j x C k
        # torch image: C k X H i X W j
        grid = np.reshape(grid, (32,32,32))
        return {'grid': grid,
                'nbv': nbv}
    
    
class ToNormalizedPositiveAngles(object):
    """Convert the 6D pose to single spherical coordinates (yaw, pitch)."""

    def __call__(self, sample):
        grid, nbv = sample['grid'], sample['nbv']

        # swap color axis because
        # numpy image: H i x W j x C k
        # torch image: C k X H i X W j
        #grid = grid.transpose((2, 0, 1))
        nbv = nbv + np.pi
        nbv = (1/(2*np.pi)) * nbv
        return {'grid':grid,
                'nbv': nbv}
    
class ToOrientationSpherical(object):
    """Convert the 6D pose to single spherical coordinates (yaw, pitch)."""

    def __call__(self, sample):
        grid, nbv = sample['grid'], sample['nbv']

        # swap color axis because
        # numpy image: H i x W j x C k
        # torch image: C k X H i X W j
        #grid = grid.transpose((2, 0, 1))
        x,y,z = nbv[:3]
        r = np.sqrt(x**2 + y**2 + z**2)
        yaw = np.arctan2(y,x)
        pitch = np.arctan2(np.sqrt(x**2 + y**2),z)
        new_nbv = np.array([yaw, pitch])
        return {'grid':grid,
                'nbv': new_nbv}
    
class ToReducedSpherical(object):
    """Convert the 6D pose to single spherical coordinates (r, yaw, pitch)."""

    def __call__(self, sample):
        grid, nbv = sample['grid'], sample['nbv']

        # swap color axis because
        # numpy image: H i x W j x C k
        # torch image: C k X H i X W j
        #grid = grid.transpose((2, 0, 1))
        x,y,z = nbv[:3]
        r = np.sqrt(x**2 + y**2 + z**2)
        yaw = np.arctan2(y,x)
        pitch = np.arctan2(np.sqrt(x**2 + y**2),z)
        new_nbv = np.array([r, yaw, pitch])
        return {'grid':grid,
                'nbv': new_nbv}
    
class ToPositionOnly(object):
    """Convert the 6D pose to single spherical coordinates (r, yaw, pitch)."""
   
    def __call__(self, sample):
        grid, nbv = sample['grid'], sample['nbv']

        return {'grid':grid,
                'nbv': nbv[:3]}
    
class RandomXflip(object):
    """Rotates the grid randomly to 90, 180 or 270 degrees."""
    
    def __init__(self, probability, grid_size):
        self.size = grid_size
        self.p = probability

    def __call__(self, sample):
        grid, nbv = sample['grid'], sample['nbv']
        # I am assuming that the to3d has already been applied
        # 
        
        if (np.random.rand() <= self.p):
            grid = np.flip(grid, axis = 0)
            nbv[0] = -1 * nbv[1]
        
        return {'grid':grid,
                'nbv': nbv}

    
class RandomYflip(object):
    """Rotates the grid randomly to 90, 180 or 270 degrees."""
    
    def __init__(self, probability, grid_size):
        self.size = grid_size
        self.p = probability

    def __call__(self, sample):
        grid, nbv = sample['grid'], sample['nbv']
        # I am assuming that the to3d has already been applied
        # 
        if (np.random.rand() <= self.p):          
            grid = np.flip(grid, axis = 1)
            nbv[1] = -1 * nbv[1]
        
        return {'grid':grid,
                'nbv': nbv}
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        grid, nbv = sample['grid'], sample['nbv']

        # swap color axis because
        # numpy image: H i x W j x C k
        # torch image: C k X H i X W j
        #grid = grid.transpose((2, 0, 1))
        return {'grid': torch.from_numpy(np.array([grid])),
                'nbv': torch.from_numpy(nbv)}

class PositionToUnitVector(object):
    def __call__(self, sample):
        grid, nbv = sample['grid'], sample['nbv']
        norm = np.linalg.norm(nbv[:3])
        nbv = (1/norm) * nbv[:3]
        
        return {'grid': grid,
                'nbv': nbv}
    
    
class VxlFreeRandomNoise(object):
    """Inserts to the free voxels random noise but keeping them as free"""
    
    def __init__(self, probability):
        self.p = probability
    
    def __call__(self, sample):
        grid, nbv = sample['grid'], sample['nbv']
        
        if (np.random.rand() <= self.p): 
            for i, p_i in enumerate(grid):
                if p_i < 0.5:
                    grid[i] = np.random.random() / 2
        
        return {'grid':grid,
                'nbv': nbv}
