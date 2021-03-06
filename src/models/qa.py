from .. import *
import json
import os
import pickle
import torch
from transformers import AutoModelForQuestionAnswering

class ModelManager:
    def __init__(self):
        pass

    def build(self, lm_name, device=None):
        pretrained_model = get_project_path('artifacts', lm_name)
        model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)
        model.to(device)
        return model

    def load(self, model_ckpt, device=None):
        config = self.load_config(model_ckpt)
        model = self.load_model(model_ckpt, config.lm_name, device=device)
        return model, config
    
    def load_config(self, model_ckpt):
        if not os.path.exists(model_ckpt):
            raise Exception('Path does not exist: %s' % model_ckpt)

        filepath = os.path.join(model_ckpt, 'model_data.pickle')
        with open(filepath, 'rb') as fp:
            model_data = pickle.load(fp)
            config = model_data['config']
        
        return config
    
    def load_model(self, model_ckpt, lm_name, device=None):
        if not os.path.exists(model_ckpt):
            raise Exception('Path does not exist: %s' % model_ckpt)

        filepath = os.path.join(model_ckpt, 'model.pt')
        model = self.build(lm_name, device=device)
        model.load_state_dict(torch.load(filepath))
        model.eval()
        return model

    def save(self, model, config, model_ckpt):
        if not os.path.exists(model_ckpt):
            os.makedirs(model_ckpt)

        model_data = {'config': config}
        filepath = os.path.join(model_ckpt, 'model_data.pickle')
        with open(filepath, 'wb') as fp:
            pickle.dump(model_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        filepath = os.path.join(model_ckpt, 'model.pt')
        torch.save(model.state_dict(), filepath)
