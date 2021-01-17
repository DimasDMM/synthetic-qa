import numpy as np
import re
import tensorflow as tf

class ToggleMetrics(tf.keras.callbacks.Callback):
    '''On test begin (i.e. when evaluate() is called or 
     validation data is run during fit()) toggle metric flag '''
    def __init__(self, metric):
        self.metric = metric

    def on_test_begin(self, logs):
        self.metric.on.assign(True)

    def on_test_end(self,  logs):
        self.metric.on.assign(False)

# See: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
class ValLossLogger(tf.keras.callbacks.Callback):
    def __init__(self, logger, metric_index='exact_match'):
        self.logger = logger
        self.metric_index = metric_index
        self.best_score = None
        super().__init__()
    
    def on_train_begin(self, logs=None):
        self.logger.info('Start training')
    
    def on_epoch_end(self, epoch, logs=None):
        self._compare_and_update(logs)
        
        msg = 'Epoch: %d - Loss: %.4f (score: %.4f) - Val. loss:  %.4f (score: %.4f) - Best score: %.4f' % (
                epoch, logs['loss'], logs[self.metric_index],
                logs['val_loss'], logs['val_%s' % self.metric_index],
                self.best_score)
        self.logger.info(msg)
    
    def _compare_and_update(self, logs):
        if best_score is None or best_score < logs['val_%s' % self.metric_index]:
            best_score = logs['val_%s' % self.metric_index]
