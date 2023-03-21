from pytorch_lightning.callbacks import Callback
import gc

class GarbageCallback(Callback):

    def on_train_epoch_end(self, *args, **kwargs):
        gc.collect()
