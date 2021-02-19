import json
import gc
import torch
from torch.utils.data import DataLoader

from .. import *
from ..data import *
from ..data.dataset import *
from ..data.squad import *
from ..data.tokenizers import *
from ..models.metrics import *
from ..models.qa import *
from ..utils.config import *
from ..utils.strategy import *

def run_qa_predict(logger, config: Config):
    logger.info(config.__dict__)
    save_path = get_project_path('artifacts', config.ckpt_name)

    logger.info('== SET UP ==')

    # Load config and model
    manager = ModelManager()
    logger.info('Loading model...')
    args_config = config
    config = manager.load_config(save_path)
    config.dataset_name = args_config.dataset_name
    config.device = args_config.device
    model = manager.load_model(save_path, config.lm_name, device=config.device)

    logger.info('Loading tokenizer')
    tokenizer = get_tokenizer(lm_name=config.lm_name, lowercase=(not config.cased))

    # Training step
    logger.info('== MODEL PREDICT ==')
    
    dataset_paths = get_dataset_path(args_config.dataset_name, 'test')
    if not isinstance(dataset_paths, dict):
        dataset_paths = {args_config.dataset_name: dataset_paths}

    for dataset_name, dataset_path in dataset_paths.items():
        # Make predictions for each dataset
        logger.info('Loading dataset: %s' % dataset_name)
        squad_preprocess = SquadPreprocess(tokenizer, max_length=config.max_length)
        test_dataset = SquadDataset(squad_preprocess, dataset_path)
        logger.info('- Test data: %d' % len(test_dataset))
        
        logger.info('Creating data loader...')
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        
        logger.info('Computing predictions...')
        qa_predict = QAPredictions(test_dataset)
        model.eval() # Set model to eval mode

        for i_batch, batch_data in enumerate(test_dataloader):
            input_ids = batch_data['input_ids'].to(device=config.device)
            token_type_ids = batch_data['token_type_ids'].to(device=config.device)
            attention_mask = batch_data['attention_mask'].to(device=config.device)
            start_token_idx = batch_data['start_token_idx'].to(device=config.device)
            end_token_idx = batch_data['end_token_idx'].to(device=config.device)
            
            attention_mask_context = batch_data['attention_mask_context']

            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            y_pred = [torch.argmax(outputs[0].cpu().detach() * attention_mask_context, dim=-1).numpy(),
                      torch.argmax(outputs[1].cpu().detach() * attention_mask_context, dim=-1).numpy()]

            indices = list(range(i_batch * config.batch_size, (i_batch + 1) * config.batch_size))
            qa_predict.add_predictions(y_pred, indices, attention_mask)
            
            # Clear some memory
            if config.device == 'cuda':
                del input_ids
                del token_type_ids
                del attention_mask
                del attention_mask_context
                del start_token_idx
                del end_token_idx
                del outputs
                gc.collect()
                torch.cuda.empty_cache()
        
        result = qa_predict.get_result()

        # Get results ans store
        logger.info('Saving results...')
        save_path = get_project_path('artifacts', config.ckpt_name, 'predictions')
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, '%s.json' % dataset_name)
        with open(filepath, 'w') as fp:
            json.dump(result, fp)

    logger.info('End training')
