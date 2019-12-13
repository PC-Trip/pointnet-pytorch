import time
import pickle

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree, KDTree
from sklearn.neighbors import KDTree as sKDTree
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    try:
        print('Loading')
        lss = np.load('lss.npy')
        ns = np.load('ns.npy')
        cts = np.load('cts.npy')
        qts = np.load('qts.npy')
        # with open('open3d.pkl', 'rb') as f:
        #     okdtree = pickle.load(f)
        with open('skdtree.pkl', 'rb') as f:
            skdtree = pickle.load(f)
        with open('ckdtree.pkl', 'rb') as f:
            ckdtree = pickle.load(f)
        with open('kdtree.pkl', 'rb') as f:
            kdtree = pickle.load(f)
    except Exception:
        fn = 's3d2\/train\/rooms\hallway_37.txt'
        print('Reading')
        t0 = time.time()
        data = np.loadtxt(fn)
        dt = time.time() - t0
        print('points: {}'.format(data.shape[0]))
        print('reading time: {}'.format(dt))
        print('points per second: {}'.format(data.shape[0] / dt))
        ps = data[:, :3].astype(np.float)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ps)
        print('Creating')
        lss = np.concatenate((np.arange(1, 10, 1),
                              np.arange(10, 100, 10),
                              np.arange(100, 1000, 100),
                              np.arange(1000, 10000, 1000),
                              np.arange(10000, 100000, 10000),
                              np.arange(100000, 1100000, 100000)))
        # lss = [100]
        print(lss)
        ns = np.concatenate((np.arange(1, 10, 1),
                             np.arange(10, 100, 10),
                             np.arange(100, 1000, 100),
                             np.arange(1000, 10000, 1000),
                             np.arange(10000, 100000, 10000),
                             np.arange(100000, 1100000, 100000)))
        # ns = [100]
        print(ns)
        cts = np.zeros((4, len(lss)))
        qts = np.zeros((4, len(lss), len(ns)))
        for i, ls in enumerate(lss):
            print('leaf: {}'.format(ls))
            t0 = time.time()
            okdtree = o3d.geometry.KDTreeFlann(pcd)
            dt = time.time() - t0
            cts[0, i] = dt
            print('open3d: {}'.format(dt))
            t0 = time.time()
            skdtree = sKDTree(ps, leaf_size=ls)
            dt = time.time() - t0
            cts[1, i] = dt
            print('sklearn KDTree: {}'.format(dt))
            t0 = time.time()
            ckdtree = cKDTree(ps, leafsize=ls)
            dt = time.time() - t0
            cts[2, i] = dt
            print('scipy cKDTree: {}'.format(dt))
            t0 = time.time()
            kdtree = KDTree(ps, leafsize=ls)
            dt = time.time() - t0
            cts[3, i] = dt
            print('scipy KDTree: {}'.format(dt))
            print('Query')
            for j, n in enumerate(ns):
                print('neighbours: {}'.format(n))
                p = [0, 0, 0]
                t0 = time.time()
                n, ds, idxs = okdtree.search_knn_vector_3d(p, n)
                dt = time.time() - t0
                qts[0, i, j] = dt
                print('open3d: {}'.format(dt))
                t0 = time.time()
                ds, idxs = skdtree.query([p], n)
                dt = time.time() - t0
                qts[1, i, j] = dt
                print('sklearn KDTree: {}'.format(dt))
                t0 = time.time()
                ds, idxs = ckdtree.query(p, n)
                dt = time.time() - t0
                qts[2, i, j] = dt
                print('scipy cKDTree: {}'.format(dt))
                t0 = time.time()
                ds, idxs = kdtree.query(p, n)
                dt = time.time() - t0
                qts[3, i, j] = dt
                print('scipy KDTree: {}'.format(dt))
        print('Saving')
        np.save('lss.npy', lss)
        np.save('ns.npy', ns)
        np.save('cts.npy', cts)
        np.save('qts.npy', qts)
        # with open('open3d.pkl', 'wb') as f:
        #     pickle.dump(okdtree, f)  # FIXME can't pickle
        with open('skdtree.pkl', 'wb') as f:
            pickle.dump(skdtree, f)
        with open('ckdtree.pkl', 'wb') as f:
            pickle.dump(ckdtree, f)
        with open('kdtree.pkl', 'wb') as f:
            pickle.dump(kdtree, f)
    print(cts)
    print(qts)
    print(lss)
    print(ns)
    p = [0, 0, 0]
    n = 3
    # print(okdtree.search_knn_vector_3d(p, n))
    print(skdtree.query([p], n))
    print(ckdtree.query(p, n))
    print(kdtree.query(p, n))
    labels = ['open3d', 'sklearn', 'scipy cKDTree', 'scipy KDTree']
    for ct in cts:
        plt.plot(lss, ct)
    plt.legend(labels)
    plt.xlabel('leaf')
    plt.ylabel('time, s')
    plt.savefig('cts.png', dpi=300)
    plt.clf()
    fig, axs = plt.subplots(2, 2)
    for i, qt in enumerate(qts):
        ax = axs.ravel()[i]
        im = ax.imshow(qt, cmap='Blues',
                       interpolation='nearest')
        ax.set_xticks(np.arange(len(ns)))
        ax.set_yticks(np.arange(len(lss)))
        ax.set_xticklabels(ns, rotation=90, fontsize=6)
        ax.set_yticklabels(lss, fontsize=6)
        ax.set_xlabel('neighbours', fontsize=6)
        ax.set_ylabel('leaf', fontsize=6)
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(labelsize=6)
        ax.set_title(labels[i])
    plt.tight_layout()
    plt.savefig('qt.png', dpi=300)
