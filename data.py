#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

# Part of the code is referred from: https://github.com/charlesq34/pointnet

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud

from hypericp import compute_components

def check(data_type, pc, num_points):
    kEpsilon = 5e-1
    (_, T), (_, S) = compute_components(pc[:num_points].T, np.random.permutation(jitter_pointcloud(pc[:num_points]).T))
    assert np.allclose(T, S, atol=1e-1)
    
    # First two singular values are equal.
    if data_type == 'f2':
        return np.isclose(S[0], S[1], atol=kEpsilon)

    # Last two singular values are equal.
    if data_type == 'l2':
        return np.isclose(S[1], S[2], atol=kEpsilon)
    
    # First two are last two singular values are equal.
    if data_type == 's2':
        return np.isclose(S[0], S[1], atol=kEpsilon) or np.isclose(S[1], S[2], atol=kEpsilon)
    
    # All singular values are distinct pair-wise.
    if data_type == 'distinct':
        return not (np.isclose(S[0], S[1], atol=kEpsilon) or np.isclose(S[1], S[2], atol=kEpsilon))
    assert 0
    return False
    
class ModelNet40(Dataset):
    def __init__(self, usage, num_points, data_size=None, partition='train', data_type=None, gaussian_noise=False, unseen=False, permute=False, factor=4, max_epochs=250):
        self.data, self.label = load_data(partition)
        self.data_size = data_size

        same_sv_count = 0
        for index in range(len(self.data)):
            if check('s2', self.data[index], num_points):
                same_sv_count += 1
        print(f'~~~~~~~~ Original statistics: SAME = {same_sv_count / len(self.data)}, DISTINCT = {1 - same_sv_count / len(self.data)} ~~~~~~~~')

        if data_type is not None:
            ls = []
            for index in range(len(self.data)):
                if check(data_type, self.data[index], num_points):
                    ls.append(index)
            
            ls = np.asarray(ls)
            self.data = self.data[ls]
            self.label = self.label[ls]

        # Bound the data. This is deterministic, i.e., we can easily debug.
        if data_size is not None:
            self.data = self.data[:data_size]
            self.label = self.label[:data_size]

        # Generate indices for point clouds where the singular values are the same.
        def generate_sv_indices():
            same_sv_indices = []
            for index in range(len(self.data)):
                if check('s2', self.data[index], num_points):
                    same_sv_indices.append(index)
            distinct_sv_indices = list(set(range(len(self.data))) - set(same_sv_indices))

            # Compute bounds.
            same_sv_bound = int(usage['percentage'] * len(same_sv_indices) / 100)
            distinct_sv_bound = int(usage['percentage'] * len(distinct_sv_indices) / 100)

            # And return.
            return (same_sv_indices, same_sv_bound), (distinct_sv_indices, distinct_sv_bound)

        bound = int(usage['percentage'] * len(self.data) / 100)         
        if usage['type'] == 'train' or usage['type'] == 'test':
            if data_type is not None:
                self.data = self.data[:bound]
                self.label = self.label[:bound]
            else:
                # Compute indices.
                (same_sv_indices, same_sv_bound), (distinct_sv_indices, distinct_sv_bound) = generate_sv_indices()

                # Slice.
                same_sv_indices = np.asarray(same_sv_indices)[:same_sv_bound]
                distinct_sv_indices = np.asarray(distinct_sv_indices)[:distinct_sv_bound]

                # And take the point clouds at those indices.
                self.data = self.data[np.concatenate((same_sv_indices, distinct_sv_indices), axis=0)]
                self.label = self.label[np.concatenate((same_sv_indices, distinct_sv_indices), axis=0)]
        else:
            if data_type is not None:
                self.data = self.data[len(self.data) - bound:]
                # In case you want to update this line, make sure you don't write `len(self.data) - bound`.
                # In that case, `self.data` has been already modified.
                self.label = self.label[len(self.label) - bound:]
            else:
                # Compute indices.
                (same_sv_indices, same_sv_bound), (distinct_sv_indices, distinct_sv_bound) = generate_sv_indices()

                # Slice.
                same_sv_indices = np.asarray(same_sv_indices)[len(same_sv_indices) - same_sv_bound:]
                distinct_sv_indices = np.asarray(distinct_sv_indices)[len(distinct_sv_indices) - distinct_sv_bound:]

                # And take the point clouds at those indices.
                self.data = self.data[np.concatenate((same_sv_indices, distinct_sv_indices), axis=0)]
                self.label = self.label[np.concatenate((same_sv_indices, distinct_sv_indices), axis=0)]

        new_same_sv_count = 0
        for index in range(len(self.data)):
            if check('s2', self.data[index], num_points):
                new_same_sv_count += 1
        print(f'~~~~~~~~ New statistics: SAME = {new_same_sv_count / len(self.data)}, DISTINCT = {1 - new_same_sv_count / len(self.data)} ~~~~~~~~')

        print(f'**************** DATA usage={usage["type"]} = {len(self.data)} *****************')

        self.num_points = num_points
        self.usage = usage
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.permute = permute
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        if len(self) > 100:
            assert self.permute and self.gaussian_noise

        # TODO: maybe vary the number of inversions in the permutation.
        if self.permute:
            pointcloud1 = np.random.permutation(pointcloud1.T).T
            pointcloud2 = np.random.permutation(pointcloud2.T).T

        # TODO: maybe vary the noise.
        # if self.gaussian_noise:
        #     pointcloud1 = jitter_pointcloud(pointcloud1)
        #     pointcloud2 = jitter_pointcloud(pointcloud2)

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data in train:
        print(len(data))
        break
