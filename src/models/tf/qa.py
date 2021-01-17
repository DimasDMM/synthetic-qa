import json
import os
import pickle
import tensorflow as tf
from transformers import TFBertForQuestionAnswering

class ModelManager:
    def __init__(self, pretrained_model):
        self.pretrained_model = pretrained_model

    def build(self):
        model = TFBertForQuestionAnswering.from_pretrained(self.pretrained_model)
        return model

    def load(self, strategy, model_ckpt):
        filepath = os.path.join(model_ckpt, 'model_data.pickle')
        with open(filepath, 'rb') as fp:
            model_data = pickle.load(fp)
            config = model_data['config']
            weights = model_data['weights']

        # Build base model and set weights
        manager = self
        with strategy.scope():
            model = manager.build()
            model.set_weights(weights)

        return model, config

    def save(self, model, config, model_ckpt):
        if not os.path.exists(config.model_ckpt):
            os.makedirs(config.model_ckpt)

        model_data = {
            'config': config,
            'weights': model.get_weights(),
        }

        filepath = os.path.join(model_ckpt, 'model_data.pickle')
        with open(filepath, 'wb') as fp:
            pickle.dump(model_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
