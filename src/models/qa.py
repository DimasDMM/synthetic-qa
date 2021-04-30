from .. import *
import json
import os
import pickle
import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering


class MuseQA(nn.Module):
    def __init__(self, word_embeddings, max_length=512, device=None):
        super().__init__()
        self.device = device
        
        self.hidden_size = word_embeddings.shape[-1]
        self.set_embedding_layer(word_embeddings)
        
        self.hidden_layer = nn.Linear(self.hidden_size, self.hidden_size).to(device=device)
        
        self.start_span = nn.Linear(self.hidden_size, 1).to(device=device)
        self.end_span = nn.Linear(self.hidden_size, 1).to(device=device)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def set_embedding_layer(self, word_embeddings):
        if word_embeddings.shape[-1] != self.hidden_size:
            raise Exception('Bad word embedding dimension! Expected %d, given %d.' % (
                    self.hidden_size, word_embeddings.shape[-1]))
        word_embeddings = torch.from_numpy(word_embeddings)
        self.emb_layer = nn.Embedding.from_pretrained(word_embeddings, freeze=True).to(device=device)
    
    def forward(self, input_ids, attention_mask, start_positions=None, end_positions=None, **kwargs):
        x = self.emb_layer(input_ids)
        attention_mask = attention_mask.type(torch.FloatTensor)
        
        x = torch.relu(self.hidden_layer(x))

        x_start = torch.relu(self.start_span(x))
        x_start = torch.flatten(x_start, start_dim=1)
        x_start = attention_mask * x_start
        
        x_end = torch.relu(self.end_span(x))
        x_end = torch.flatten(x_end, start_dim=1)
        x_end = attention_mask * x_end
        
        if start_positions is None or end_positions is None:
            return x_start, x_end
        else:
            loss_start = self.criterion(x_start, start_positions)
            loss_end = self.criterion(x_end, end_positions)
            loss = loss_start + loss_end
            return loss, x_start, x_end
    
    def get_top_weights(self):
        return self.hidden_layer.weight, self.start_span.weight, self.end_span.weight
    
    def set_top_weights(self, hidden_layer_weights, start_span_weights, end_span_weights):
        self.hidden_layer.weight = hidden_layer_weights
        self.start_span.weight = start_span_weights
        self.end_span.weight = end_span_weights

class ModelManager:
    def __init__(self):
        pass

    def build(self, device=None, **kwargs):
        raise NotImplementedError()

    def load(self, model_ckpt, device=None):
        config = self.load_config(model_ckpt)
        model = self.load_model(model_ckpt, lm_name=config.lm_name, device=device)
        return model, config
    
    def load_config(self, model_ckpt):
        if not os.path.exists(model_ckpt):
            raise Exception('Path does not exist: %s' % model_ckpt)

        filepath = os.path.join(model_ckpt, 'model_data.pickle')
        with open(filepath, 'rb') as fp:
            model_data = pickle.load(fp)
            config = model_data['config']
        
        return config
    
    def load_model(self, model_ckpt, device=None, **kwargs):
        if not os.path.exists(model_ckpt):
            raise Exception('Path does not exist: %s' % model_ckpt)

        filepath = os.path.join(model_ckpt, 'model.pt')
        model = self.build(device=device, **kwargs)
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

class BertModelManager(ModelManager):
    def __init__(self):
        ModelManager.__init__(self)

    def build(self, device=None, **kwargs):
        pretrained_model = get_project_path('artifacts', kwargs['lm_name'])
        model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model)
        model.to(device)
        return model

class MuseModelManager(ModelManager):
    def __init__(self):
        ModelManager.__init__(self)

    def build(self, device=None, **kwargs):
        model = MuseQA(kwargs['word_embeddings'], kwargs['word2id'], device=device)
        model.to(device)
        return model
