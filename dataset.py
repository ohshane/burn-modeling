from datalib.dataset import ABDataset, ClinicalTrialDataset

from transforms import *

supervised_abdataset = \
    ABDataset(transform=transform,
              bbox_transform=bbox_transform,
              coords_transform=list,
              filter=supervised)

supervised_abdataset_train = \
    ABDataset(transform=transform,
              bbox_transform=bbox_transform,
              coords_transform=list,
              filter=supervised_train)

supervised_abdataset_val = \
    ABDataset(transform=transform,
              bbox_transform=bbox_transform,
              coords_transform=list,
              filter=supervised_val)

supervised_abdataset_test = \
    ABDataset(transform=transform,
              bbox_transform=bbox_transform,
              coords_transform=list,
              filter=supervised_test)

unsupervised_abdataset = \
    ABDataset(transform=transform,
              bbox_transform=bbox_transform,
              coords_transform=list,
              filter=unsupervised)

supervised_ctdataset_test = \
    ClinicalTrialDataset(transform=transform,
                         filter=supervised)

