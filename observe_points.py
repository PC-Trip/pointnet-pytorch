import os
import numpy as np


def observe_grouped_areas_points(root):
    classes = {}
    areas = os.listdir(root)
    for area in areas:
        rooms = os.listdir(os.path.join(root, area))
        rooms = [r for r in rooms if os.path.isdir(os.path.join(root, area, r))]
        for room in rooms:
            files = os.listdir(os.path.join(root, area, room, 'Annotations'))
            files = list(filter(lambda x: 'label' not in x and 'pred' not in x, files))
            for f in files: 
                f_path = os.path.join(root, area, room, 'Annotations', f)
                clss = f.split('_')[0]
                if clss not in classes:
                    classes[clss] = 0
                classes[clss] += np.loadtxt(f_path).astype(np.float32).shape[0]

    return classes


def observe_grouped_areas_elements(root):
    classes = {}
    areas = os.listdir(root)
    for area in areas:
        rooms = os.listdir(os.path.join(root, area))
        rooms = [r for r in rooms if os.path.isdir(os.path.join(root, area, r))]
        for room in rooms:
            files = os.listdir(os.path.join(root, area, room, 'Annotations'))
            files = list(filter(lambda x: 'label' not in x and 'pred' not in x, files))
            for f in files: 
                f_path = os.path.join(root, area, room, 'Annotations', f)
                clss = f.split('_')[0]
                if clss not in classes:
                    classes[clss] = 1
                else:
                    classes[clss] += 1
    return classes

if __name__ == "__main__":
    print("train folder")
    train_classes = observe_grouped_areas_elements('train')
    print(train_classes)

    print('test folder')
    test_classes = observe_grouped_areas_elements('test')
    print(test_classes)
