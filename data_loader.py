import tensorflow as tf
import tensorflow_io as tfio
import pandas as pd
from glob import glob
from config import CFG
import os

def build_decoder(with_labels=True, target_size=CFG.image_size, augment=False):
    def decode_image(paths):
        img_array = tf.TensorArray(dtype=tf.uint8, size=len(paths))
        for i in range(len(paths)):
            file_bytes = tf.io.read_file(paths[i])
            img0 = tfio.experimental.image.decode_tiff(file_bytes)[..., 0:1]
            img_array = img_array.write(i, img0[...,0])
        img = tf.transpose(img_array.stack(), perm=(1, 2, 0))
        img = tf.cast(img, tf.float32)
        img -= tf.reduce_min(img)
        img /= tf.reduce_max(img) + 0.001
        del img_array
        return img
    
    def decode_mask(mask_path):
        file_bytes = tf.io.read_file(mask_path)
        msk = tfio.experimental.image.decode_tiff(file_bytes)[...,0:1]
        msk = tf.cast(msk, tf.float32) / 255.0
        return msk

    def decode_without_labels(img_path):
        img = decode_image(img_path)
        img = tf.reshape(img, [*target_size, 3])
        return img
    
    def decode_with_labels(img_path, msk_path):
        img_msk = tf.concat([decode_image(img_path), decode_mask(msk_path)], axis=-1)
        img_msk = tf.image.random_crop(img_msk, [*target_size, 4])
        if augment:
            img_msk = apply_augmentations(img_msk)
        img = tf.reshape(img_msk[...,0:3], [*target_size, 3])
        msk = tf.reshape(img_msk[...,3:4], [*target_size, 1])
        return (img, msk)
    
    def apply_augmentations(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        return img
    
    return decode_with_labels if with_labels else decode_without_labels

def build_dataset(img_paths, msk_paths=None, batch_size=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, repeat=True, shuffle=1024, 
                  cache_dir="", drop_remainder=False):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    if decode_fn is None:
        decode_fn = build_decoder(msk_paths is not None, augment=augment)
    
    AUTO = tf.data.experimental.AUTOTUNE
    slices = img_paths if msk_paths is None else (img_paths, msk_paths)
    
    ds = tf.data.Dataset.from_tensor_slices(slices)
    ds = ds.map(decode_fn, num_parallel_calls=AUTO)
    ds = ds.cache(cache_dir) if cache else ds
    ds = ds.repeat() if repeat else ds
    if shuffle: 
        ds = ds.shuffle(shuffle, seed=CFG.seed)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTO)
    return ds

def load_data(base_path):
    mask_paths = sorted(glob(f"{base_path}/train/*/labels/*tif"))
    df = pd.DataFrame({"mask_path": mask_paths})
    df['dataset'] = df.mask_path.map(lambda x: x.split('/')[-3])
    df['slice'] = df.mask_path.map(lambda x: x.split('/')[-1].replace(".tif",""))

    df = df[~df.dataset.str.contains("kidney_3_sparse")]
    df['image_path'] = df.mask_path.str.replace("label","image")
    df['image_path'] = df.image_path.str.replace("kidney_3_dense","kidney_3_sparse")

    CHANNELS = 3 
    STRIDE = 3 
    for i in range(CHANNELS):
        df[f'image_path_{i:02}'] = df.groupby(['dataset'])['image_path'].shift(-i*STRIDE).ffill()
    df['image_paths'] = df[[f'image_path_{i:02d}' for i in range(CHANNELS)]].values.tolist()

    train_df = df[~df.dataset.str.contains('kidney_3')]
    valid_df = df[df.dataset.str.contains('kidney_3')]

    return train_df, valid_df