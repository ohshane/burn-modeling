import datalib.transforms as DT
from sklearn.model_selection import train_test_split


def dropna():
    def _f(df):
        df = df.dropna()
        return df
    return _f

def label_exists():
    def _f(df):
        df = df[df.label != -1]
        return df
    return _f

def label_not_exists():
    def _f(df):
        df = df[df.label == -1]
        return df
    return _f

def split(mode, seed=42):
    assert mode in ['train', 'val', 'test']
    def _f(df):
        train_df, _df = train_test_split(df, test_size=0.2, random_state=seed)
        val_df, test_df = train_test_split(_df, test_size=0.5, random_state=seed)
        if mode == 'train':
            return train_df
        elif mode == 'val':
            return val_df
        elif mode == 'test':
            return test_df
    return _f

supervised = DT.Compose([
    dropna(),
    label_exists(),
])

supervised_train = DT.Compose([
    dropna(),
    label_exists(),
    split(mode='train'),
])

supervised_val = DT.Compose([
    dropna(),
    label_exists(),
    split(mode='val'),
])

supervised_test = DT.Compose([
    dropna(),
    label_exists(),
    split(mode='test'),
])

unsupervised = DT.Compose([
    dropna(),
    label_not_exists(),
])

bbox_transform = transform = DT.Compose([
    DT.exif_transpose(),
    DT.squish_resize(224),
    DT.to_tensor(),
    DT.normalize(mode='imagenet'),
])

import torchvision.transforms as T

bbox_transform_train = transform_train = DT.Compose([
    DT.exif_transpose(),
    DT.squish_resize(384),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    DT.to_tensor(),
    DT.normalize(mode='imagenet'),
])

coords_transform = DT.Compose([
    DT.exif_transpose(),
    DT.squish_resize(384),
    DT.to_tensor(),
    DT.normalize(mode='imagenet'),
])

