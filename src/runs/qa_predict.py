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
    output_pred_path = args_config.output_pred_path
    output_pred_file = args_config.output_pred_file
    
    if not os.path.exists(dataset_test_path):
        raise Exception('Test dataset does not exist: %s' % dataset_test_path)
    elif args_config.model_type != config.model_type:
        raise Exception('Expected model "%s", but "%s" found!' % (args_config.model_type, config.model_type))

    # Load config and model
    logger.info('Loading model...')
    if args_config.model_type == 'transformers':
        manager = BertModelManager()
    else:
        manager = MuseModelManager()
    config = Config()
    config.copy_from(manager.load_config(save_path))
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
    
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)
    model_predictions = ModelPredictions(device=config.device)
    qa_em_metric = ExactMatchMetric()
    qa_f1_metric = F1ScoreMetric()
    test_model.eval() # Set model to eval mode

    test_predictions = model_predictions.get_predictions(test_model, test_dataloader, test_dataset)
    em_score, _ = qa_em_metric.eval(test_dataset, test_predictions)
    f1_score, _ = qa_f1_metric.eval(test_dataset, test_predictions)
    
    logger.info('ExactMatch: %.4f | F1-Score: %.4f' % (em_score, f1_score))

    # Get predictionss ans store
    save_path = get_project_path(output_pred_path, config.ckpt_name)
    logger.info('Saving predictions at: %s' % save_path)
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, output_pred_file)
    with open(filepath, 'w') as fp:
        json.dump(test_predictions, fp)

    logger.info('Done')
