"""
Util to get files from annotated dir, 
add there labels from s3d_cat2num file,
and compile into one file like full room.

After this processing the result file will
be looked into and sliced into 1*1*1m cubes
checked for amount of points inside, randomly
sampled for train points and pushed into the net.
"""

import os
import numpy as np

from utils import get_s3d_cat2num


if __name__ == "__main__":
    cat2num = get_s3d_cat2num()
    roots = ['test', 'train']
    for root in roots:
        areas = os.listdir(root)
        for area in areas:
            rooms = os.listdir(os.path.join(root, area))
            rooms = [r for r in rooms if os.path.isdir(os.path.join(root, area, r))]
            for room in rooms:
                elements_buffer =[]
                files = os.listdir(os.path.join(root, area, room, 'Annotations'))
                files = list(filter(lambda x: 'label' not in x and 'pred' not in x, files))
                for f in files: 
                    f_path = os.path.join(root, area, room, 'Annotations', f)
                    l_path = os.path.join(root, area, room, 'Annotations', f.split('.')[0]+'_labels.txt')
                    points = np.loadtxt(f_path).astype(np.float32)
                    clss = np.zeros((points.shape[0], 1)) + cat2num[f.split('_')[0]]
                    points = np.concatenate([points, clss], 1)
                    elements_buffer.append(points)
                np.savetxt(os.path.join(root, area, room, room+'_full.txt'), np.concatenate(elements_buffer, 0), fmt='%2.3f %2.3f %2.3f %i %i %i %i')
                
