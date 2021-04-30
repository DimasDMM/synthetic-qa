import json
import gc
import torch
from torch.utils.data import DataLoader

from .. import *
from ..data import *
from ..data.dataset import *
from ..data.embeddings import *
from ..data.squad import *
from ..data.tokenizers import *
from ..models.metrics import *
from ..models.qa import *
from ..utils.config import *

def run_qa_predict(logger, config: Config):
    logger.info(config.__dict__)
    save_path = get_project_path('artifacts', config.ckpt_name)

    logger.info('== SET UP ==')

    args_config = config
    dataset_test_path = args_config.dataset_test_path
    dataset_test_lang = args_config.dataset_test_lang
    output_pred_file = args_config.output_pred_file
    
    if not os.path.exists(dataset_test_path):
        raise Exception('Test dataset does not exist: %s' % dataset_test_path)
    elif args_config.model_type != config.model_type:
        raise Exception('Expected model "%s", but "%s" found!' % (args_config.model_type, config.model_type))

    # Load config and model
    manager = ModelManager()
    logger.info('Loading model...')
    config = manager.load_config(save_path)
    config.device = args_config.device
    logger.info(config.__dict__)

    if config.model_type == 'transformers':
        train_tokenizer = get_piece_tokenizer(lm_name=config.lm_name, lowercase=(not config.cased))
        train_model = manager.load_model(save_path, lm_name=config.lm_name, device=config.device)
        test_tokenizer = train_tokenizer
        test_model = train_model
    else:
        train_word_embeddings, _, train_word2id = load_embeddings(config.dataset_train_lang, embeddings_path='muse')
        train_tokenizer = get_word_tokenizer(lang_code=config.dataset_train_lang, word2id=train_word2id)
        train_model = manager.load_model(save_path, word_embeddings=train_word_embeddings, device=config.device)

        test_word_embeddings, _, test_word2id = load_embeddings(dataset_test_lang, embeddings_path='muse')
        test_tokenizer = get_word_tokenizer(lang_code=dataset_test_lang, word2id=test_word2id)
        test_model = manager.load_model(save_path, word_embeddings=train_word_embeddings, device=config.device)
        test_model.set_embedding_layer(test_word_embeddings)

    # Training step
    logger.info('== MODEL PREDICT ==')

    logger.info('Loading dataset: %s' % dataset_test_path)
    squad_preprocess = SquadPreprocess(test_tokenizer, max_length=config.max_length)
    test_dataset = SquadDataset(squad_preprocess, dataset_test_path)
    logger.info('- Test data: %d' % len(test_dataset))
    
    logger.info('Creating data loader...')
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    logger.info('Computing predictions...')
    qa_metric = ExactMatch(test_dataset, device=config.device)
    test_model.eval() # Set model to eval mode

    for batch_data in test_dataloader:
        input_ids = batch_data['input_ids'].to(device=config.device)
        token_type_ids = batch_data['token_type_ids'].to(device=config.device)
        attention_mask = batch_data['attention_mask'].to(device=config.device)
        start_token_idx = batch_data['start_token_idx'].to(device=config.device)
        end_token_idx = batch_data['end_token_idx'].to(device=config.device)
        
        attention_mask_context = batch_data['attention_mask_context']

        outputs = test_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        y_pred_start = outputs[0].cpu().detach().numpy()
        y_pred_end = outputs[1].cpu().detach().numpy()
        y_pred = [y_pred_start, y_pred_end]

        indices = batch_data['idx']
        qa_metric.add_predictions(indices, y_pred)
        
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
    
    result = qa_metric.result()
    logger.info('Score: %.4f' % result)

    predictions = qa_metric.get_predictions()

    # Get predictionss ans store
    save_path = get_project_path('artifacts', 'predictions', config.ckpt_name)
    logger.info('Saving predictions at: %s' % save_path)
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, output_pred_file)
    with open(filepath, 'w') as fp:
        json.dump(predictions, fp)

    logger.info('Done')
