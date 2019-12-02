import open3d
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--path', type=str, required=True, help='file path to use open3d upon')
    file_path = parser.parse_args().path
    points = np.loadtxt(file_path).astype(np.float32)
    points = points[:, :3]
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    open3d.visualization.draw_geometries([pcd])