import keras
import numpy as np
from config import CFG
from data_loader import load_data, build_dataset
from model import build_model
from losses import DiceLoss, DiceCoef
from utils import get_lr_callback

def train():

    train_df, valid_df = load_data("/kaggle/input/blood-vessel-segmentation")

    # Create datasets
    train_ds = build_dataset(train_df.image_paths.tolist(), train_df.mask_path.tolist(), 
                             batch_size=CFG.batch_size, cache=CFG.cache, augment=True)
    valid_ds = build_dataset(valid_df.image_paths.tolist(), valid_df.mask_path.tolist(), 
                             batch_size=CFG.batch_size, cache=CFG.cache, repeat=False, 
                             shuffle=False, augment=False)


    model = build_model()

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    loss = DiceLoss()
    metrics = [DiceCoef(), keras.metrics.BinaryAccuracy(name="accuracy")]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    lr_callback = get_lr_callback(CFG.batch_size)
    ckpt_callback = keras.callbacks.ModelCheckpoint("best_model.keras",
                                                    monitor='val_dice_coef',
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    mode='max')
    history = model.fit(
        train_ds, 
        epochs=CFG.epochs,
        callbacks=[lr_callback, ckpt_callback], 
        steps_per_epoch=len(train_df)//CFG.batch_size,
        validation_data=valid_ds, 
        verbose=CFG.verbose
    )

    return history, model

if __name__ == "__main__":
    history, model = train()
