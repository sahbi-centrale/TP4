import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import trimesh


# Hoppe surface reconstruction
def compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel):
    x = min_grid[0] + size_voxel[0] * np.arange(grid_resolution+1)
    y = min_grid[1] + size_voxel[1] * np.arange(grid_resolution+1)
    z = min_grid[2] + size_voxel[2] * np.arange(grid_resolution+1)
    X, Y, Z = np.meshgrid(x, y, z)
    XYZ = np.stack((X,Y,Z), 3).reshape(-1,3)

    tree = KDTree(points)
    closest = tree.query(XYZ, 1, return_distance=False).squeeze()
    hoppe_fun = np.sum(normals[closest] * (XYZ-points[closest]), axis=1)
    scalar_field[:,:] = hoppe_fun.reshape(grid_resolution+1, grid_resolution+1, grid_resolution+1)

				
# EIMLS surface reconstruction
def compute_eimls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,knn):
    x = min_grid[0] + size_voxel[0] * np.arange(grid_resolution+1)
    y = min_grid[1] + size_voxel[1] * np.arange(grid_resolution+1)
    z = min_grid[2] + size_voxel[2] * np.arange(grid_resolution+1)
    X, Y, Z = np.meshgrid(x, y, z)
    XYZ = np.stack((X,Y,Z), 3).reshape(-1,3)

    
    
    tree = KDTree(points)
    closest = tree.query(XYZ, knn, return_distance=False).squeeze()
    
    xpi = XYZ[:,np.newaxis,:] - points[closest]
    xpi_norm = np.linalg.norm(xpi, axis=2)
    h = np.clip(xpi_norm, 0.003, None)

    theta = np.exp(-(xpi_norm**2)/h**2)
    
    nxpi = np.sum(normals[closest] * xpi, axis=2)
    eiml_fun = np.sum(nxpi * theta, axis=1) / np.sum(theta, axis = 1)

    scalar_field[:,:] = eiml_fun.reshape(grid_resolution+1, grid_resolution+1, grid_resolution+1)

if __name__ == '__main__':

    # Path of the file
    file_path = 'bunny_normals.ply'
    t0 = time.time()

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

    # Compute the min and max of the data points
    min_grid = np.copy(points[0, :])
    max_grid = np.copy(points[0, :])
    for i in range(1,points.shape[0]):
        for j in range(0,3):
            if (points[i,j] < min_grid[j]):
                min_grid[j] = points[i,j]
            if (points[i,j] > max_grid[j]):
                max_grid[j] = points[i,j]

    number_cells = 128
	# Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - (max_grid-min_grid) / number_cells
    max_grid = max_grid + (max_grid-min_grid) / number_cells

	# Number_cells is the number of voxels in the grid in x, y, z axis
    length_cell = np.array([(max_grid[0]-min_grid[0])/number_cells,(max_grid[1]-min_grid[1])/number_cells,(max_grid[2]-min_grid[2])/number_cells])
	
	# Create a volume grid to compute the scalar field for surface reconstruction
    volume = np.zeros((number_cells+1,number_cells+1,number_cells+1),dtype = np.float32)

	# Compute the scalar field in the grid
    #compute_hoppe(points,normals,volume,number_cells,min_grid,length_cell)
    compute_eimls(points,normals,volume,number_cells,min_grid,length_cell,30)

	# Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes(volume, level=0.0, spacing=(length_cell[0],length_cell[1],length_cell[2]))
    verts += min_grid
	
    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices = verts, faces = faces)
    mesh.export(file_obj='bunny_mesh_hoppe_17.ply', file_type='ply')
	
    print("Total time for surface reconstruction : ", time.time()-t0)
	
