import numpy as np
import torch as T
from src.metric.evaluation import probability_table


class SCMDataTypes:
    BINARY = "binary"
    REP_BINARY = "rep_binary"
    BINARY_ONES = "binary_ones"
    REP_BINARY_ONES = "rep_binary_ones"
    ONE_HOT = "one_hot"
    REAL = "real"
    IMAGE = "image"


class SCMDataGenerator:
    def __init__(self, mode="sampling", normalize=True):
        self.v_size = {}
        self.v_type = {}
        self.cg = None
        self.mode = mode
        self.normalize = normalize

    def generate_samples(self, n):
        return None


class SCMDataset(T.utils.data.Dataset):
    def __init__(self, datagen: SCMDataGenerator, n: int, augment_transform=None, use_tau=False, soft_sampling=False,
                 soft_sampling_mode=None):
        self.datagen = datagen
        self.v_type = datagen.v_type
        self.n = n
        self.data = None
        if not self.datagen.evaluating:
            self.data = datagen.generate_samples(n, use_tau=use_tau, soft_sampling=soft_sampling,
                                                 soft_sampling_mode=soft_sampling_mode)
            if soft_sampling:
                _, i_type_sampler, o_name, _, o_type, _ = datagen.get_sampler_metadata(mode=soft_sampling_mode)
                self.v_type = i_type_sampler
                self.v_type[o_name] = o_type
            elif use_tau:
                self.v_type = datagen.v_type_high_level
        self.augment_transform = augment_transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        out_data = {}
        for var in self.data:
            if self.v_type[var] == SCMDataTypes.IMAGE:
                if self.augment_transform is not None:
                    out_data[var] = self.augment_transform(self.data[var][idx])
                else:
                    out_data[var] = self.data[var][idx]
            else:
                out_data[var] = self.data[var][idx]
        return out_data

    def get_image_batch(self, batch_size=64):
        data = self[:batch_size]
        out_data = {}
        for var in self.data:
            if self.v_type[var] == SCMDataTypes.IMAGE:
                out_data[var] = data[var]

        return out_data

    def get_prob_table(self):
        if self.datagen.evaluating:
            return None
        dat_noimg = {k: v for (k, v) in self.data.items() if self.v_type[k] != SCMDataTypes.IMAGE}
        return probability_table(dat=dat_noimg)
