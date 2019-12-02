import os



if __name__ == '__main__':
    root = os.getcwd()
    file_name = 's3d_cat2num.txt'
    units = set()
    for d1 in ['train', 'test']:
        root1 = os.path.join(root, d1)
        for d2 in os.listdir(root1):
            root2 = os.path.join(root1, d2)
            for scene in filter(lambda s: os.path.isdir(os.path.join(root2, s)), os.listdir(root2)):
                root3 = os.path.join(root, d1, d2, scene, 'Annotations')
                for f in filter(lambda s: not s.startswith('.DS'), os.listdir(root3)):
                    units.add(f.split('_')[0])

    with open(file=file_name, mode='w') as f:
        for i, unit in enumerate(list(units)):
            f.write('{} {}\n'.format(unit, i))