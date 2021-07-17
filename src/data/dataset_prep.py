import os
import pickle

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(points, n_points):
    """
    Input:
        xyz: point cloud data, [N, D]
        n_points: number of samples
    Return:
        centroids: sampled point cloud index, [n_points, D]
    """
    N, D = points.shape
    xyz = points[:, :3]
    centroids = np.zeros((n_points,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_points):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    points = points[centroids.astype(np.int32)]
    return points


class ModelNetDataLoader(Dataset):
    """
    Prepares a ModelNetDataset
    """

    def __init__(self, root, args, split='train', process_data=False):
        """

        :param root: Root Directory
        :param args: Arguments
        :param split: split type train or test
        :param process_data: Boolean
        """
        self.root = root
        self.n_points = args.num_point
        self.process_data = args.process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        if self.num_category == 10:
            self.categories_file = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.categories_file = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.categories = [line.rstrip() for line in open(self.categories_file)]
        self.classes = dict(zip(self.categories, range(len(self.categories))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train_new.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test_new.txt'))]
            shape_ids['val'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_val_new.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train_new.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test_new.txt'))]
            shape_ids['val'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_val_new.txt'))]

        assert (split in ['train', 'test', 'val'])
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.data_path = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                          in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.data_path)))

        if self.uniform:
            self.save_path = os.path.join(root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.n_points))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.n_points))
        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.data_path)
                self.list_of_labels = [None] * len(self.data_path)

                for index in tqdm(range(len(self.data_path)), total=len(self.data_path)):
                    fn = self.data_path[index]
                    cls = self.classes[self.data_path[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.n_points)
                    else:
                        point_set = point_set[0:self.n_points, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.data_path)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.data_path[index]
            cls = self.classes[self.data_path[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.n_points)
            else:
                point_set = point_set[0:self.n_points, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)
