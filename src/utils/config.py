import copy

class Config:
    def __init__(
        self,
        cased=True,
        model_type='transformers',
        ckpt_name='model_ckpt',
        dataset_train_path='train.json',
        dataset_train_lang='en',
        dataset_dev_path='dev.json',
        dataset_dev_lang='en',
        dataset_test_path='test.json',
        dataset_test_lang='en',
        output_pred_path='artifacts/predictions',
        output_pred_file='predictions.json',
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
        self.model_type = model_type
        self.ckpt_name = ckpt_name
        self.dataset_train_path = dataset_train_path
        self.dataset_train_lang = dataset_train_lang
        self.dataset_dev_path = dataset_dev_path
        self.dataset_dev_lang = dataset_dev_lang
        self.dataset_test_path = dataset_test_path
        self.dataset_test_lang = dataset_test_lang
        self.output_pred_path = output_pred_path
        self.output_pred_file = output_pred_file
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
    
    def copy_from(self, obj):
        for attr_name in dir(obj):
            if not attr_name.startswith('__') and not callable(getattr(obj, attr_name)):
                setattr(self, attr_name, getattr(obj, attr_name))
    
    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)
