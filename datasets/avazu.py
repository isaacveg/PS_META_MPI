import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import shutil
import struct
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import lmdb
import sys
# sys.path.append('/data/slwang/FL_MPI_CV_REC')
from config import cfg


class AvazuDataset(Dataset):
    """
    Avazu Click-Through Rate (CTR) Prediction Dataset

    Dataset preparation
        Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature

    :param dataset_path: avazu train path
    :param cache_path: lmdb cache path
    :param rebuild_cache: If True, lmdb cache is refreshed
    :param min_threshold: infrequent feature threshold

    Reference
        https://www.kaggle.com/c/avazu-ctr-prediction
    """

    def __init__(self, dataset_path=cfg['dataset_path'], cache_path=cfg['cache_path'], rebuild_cache=False, min_threshold=4, is_training=True):
        super().__init__()
        self.train_num = 90000
        self.test_num = 10000
        self.NUM_FEATS = 22
        self.min_threshold = min_threshold
        self.is_training = is_training
        labels = np.array(pd.read_csv(dataset_path)['click'])
        if self.is_training:
            self.labels = labels[:self.train_num]
        else:
            self.labels = labels[self.train_num: self.train_num + self.test_num]
        print(np.where(self.labels == 0)[0].shape)
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)

        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        if not self.is_training:
            index += self.train_num
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.int64)
        
        one_hot = np.empty(0, dtype=np.float32)
        for i, feature_index in enumerate(np_array[1:]):
            one_hot_field = np.zeros(self.field_dims[i], dtype=np.float32)
            one_hot_field[feature_index] = 1
            one_hot = np.concatenate((one_hot, one_hot_field))
            
        # one_hot = F.const(one_hot.flatten().tolist(), [int(np.sum(self.field_dims))], F.data_format.NCHW)
        # label = F.const([np_array[0]], [], F.data_format.NCHW, F.dtype.uint8)
        one_hot = torch.from_numpy(one_hot)
        label = torch.tensor(np_array[0], dtype=torch.float32)

        return one_hot, label

    def __len__(self):
        if self.is_training:
            return self.train_num
        else:
            return self.test_num

    def __build_cache(self, dataset_path, cache_path):
        feat_mapper, defaults = self.__get_feat_mapper(dataset_path)
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                field_dims[i - 1] = len(fm) + 1
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())
            for buffer in self.__yield_buffer(dataset_path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __get_feat_mapper(self, dataset_path):
        feat_cnts = defaultdict(lambda: defaultdict(int))  # the value is defaultdict(int)
        count = 0
        with open(dataset_path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create avazu dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2:
                    continue
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i + 1]] += 1  # the number of each feature
                count += 1
                if count >= self.train_num + self.test_num:
                    break
        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in
                       feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        return feat_mapper, defaults

    def __yield_buffer(self, dataset_path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(dataset_path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create avazu dataset cache: setup lmdb')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2:
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
                np_array[0] = int(values[1])
                for i in range(1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i + 1], defaults[i])
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
                if item_idx >= self.train_num + self.test_num:
                    break
            yield buffer


# dataset = AvazuDataset(dataset_path='/data/slwang/datasets/avazu/mini_set.csv', rebuild_cache=True, is_training=False)
# print(dataset.__getitem__(9999))
# print(len(dataset))
# print(dataset.__getitem__(1)[1])

def create_dataset(data_path):
    return  AvazuDataset(dataset_path=data_path, is_training=True), AvazuDataset(dataset_path=data_path, is_training=False)

