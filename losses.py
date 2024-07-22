import keras
from keras import ops

class DiceLoss(keras.losses.Loss):
    def __init__(self, smooth=1e-4, name="dice_loss"):
        super().__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = ops.ravel(y_true)
        y_pred = ops.ravel(y_pred)
        
        intersection = ops.sum(y_true * y_pred)
        union = ops.sum(y_true) + ops.sum(y_pred)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice
    
class DiceCoef(keras.metrics.Metric):
    def __init__(self, name='dice_coef', smooth=1e-4, threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.threshold = threshold
        self.intersection_sum = self.add_weight(name='intersection_sum', initializer='zeros')
        self.union_sum = self.add_weight(name='union_sum', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = ops.cast(y_pred > self.threshold, dtype="float32")
        y_true = ops.ravel(y_true)
        y_pred = ops.ravel(y_pred)

        intersection = ops.sum(y_true * y_pred)
        union = ops.sum(y_true) + ops.sum(y_pred)

        self.intersection_sum.assign_add(intersection)
        self.union_sum.assign_add(union)

    def result(self):
        dice = (2 * self.intersection_sum + self.smooth) / (self.union_sum + self.smooth)
        return dice

    def reset_states(self):
        self.intersection_sum.assign(0)
        self.union_sum.assign(0)
        
    def get_config(self):
        config = super().get_config()
        config.update({'smooth': self.smooth, 'threshold': self.threshold})
        return config