import copy

class Config:
    def __init__(
        self,
        cased=True,
        ckpt_name='model_ckpt',
        dataset_name='squad',
        hidden_dim=768,
        batch_size=32,
        max_epoches=10,
        max_length=512,
        learning_rate=1e-5,
        lm_name='bert-base-cased',
        continue_training=False,
        device=None,
        current_score=0.,
        current_epoch=0,
        *args,
        **kargs,
    ):
        self.cased = cased
        self.ckpt_name = ckpt_name
        self.dataset_name = dataset_name
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_epoches = max_epoches
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.lm_name = lm_name
        self.continue_training = continue_training
        self.device = device
        self.current_score = current_score
        self.current_epoch = current_epoch
        
    def __call__(self, **kargs):
        obj = copy.copy(self)
        for k, v in kargs.items():
            setattr(obj, k, v)
        return obj
    
    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)
