import pptk
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--path', type=str, required=True, help='file path to use open3d upon')
    file_path = parser.parse_args().path
    points = np.loadtxt(file_path, skiprows=1).astype(np.float32)
    print(np.min(points[:, 0], axis=0), np.max(points[:, 0], axis=0), np.max(points[:, 0], axis=0) - np.min(points[:, 0], axis=0))
    print(np.min(points[:, 1], axis=0), np.max(points[:, 1], axis=0), np.max(points[:, 1], axis=0) - np.min(points[:, 1], axis=0))
    print(np.min(points[:, 2], axis=0), np.max(points[:, 2], axis=0), np.max(points[:, 2], axis=0) - np.min(points[:, 2], axis=0))
    v = pptk.viewer(points[:, :3])
    if points.shape[1] == 4 or points.shape[1] == 7:
        v.attributes(points[:, -1])
    
    print(v.get('selected'))