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


def run_qa_training(logger, config: Config):
    logger.info(config.__dict__)
    save_path = get_project_path('artifacts', config.ckpt_name)
    best_score = 0.

    logger.info('Base path: %s' % BASE_PATH)

    logger.info('== SET UP ==')

    logger.info('Loading tokenizer')
    if config.model_type == 'transformers':
        train_tokenizer = get_piece_tokenizer(lm_name=config.lm_name, lowercase=(not config.cased))
        dev_tokenizer = train_tokenizer
    else:
        train_word_embeddings, _, train_word2id = load_embeddings(config.dataset_train_lang, embeddings_path='muse')
        dev_word_embeddings, _, dev_word2id = load_embeddings(config.dataset_dev_lang, embeddings_path='muse')
        train_tokenizer = get_word_tokenizer(lang_code=config.dataset_train_lang, word2id=train_word2id)
        dev_tokenizer = get_word_tokenizer(lang_code=config.dataset_train_lang, word2id=dev_word2id)
    
    dataset_train_path = config.dataset_train_path
    dataset_dev_path = config.dataset_dev_path
    if not os.path.exists(dataset_train_path):
        raise Exception('Train dataset does not exist: %s' % dataset_train_path)
    elif not os.path.exists(dataset_dev_path):
        raise Exception('Dev dataset does not exist: %s' % dataset_dev_path)

    logger.info('Loading train dataset: %s' % dataset_train_path)
    train_squad_preprocess = SquadPreprocess(train_tokenizer, max_length=config.max_length)
    train_dataset = SquadDataset(train_squad_preprocess, dataset_train_path, save_contexts=False)
    train_skipped = train_dataset.get_skipped_items()
    logger.info('- Train data: %d (skipped: %d)' % (len(train_dataset), len(train_skipped)))

    logger.info('Loading dev dataset: %s' % dataset_dev_path)
    dev_squad_preprocess = SquadPreprocess(dev_tokenizer, max_length=config.max_length)
    dev_dataset = SquadDataset(dev_squad_preprocess, dataset_dev_path)
    dev_skipped = dev_dataset.get_skipped_items()
    logger.info('- Dev data: %d (skipped: %d)' % (len(dev_dataset), len(dev_skipped)))
    
    logger.info('Creating data loaders...')
    set_seed(DEFAULT_SEED)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    # Load trained model or build a new one
    if config.model_type == 'transformers':
        manager = BertModelManager()
    else:
        manager = MuseModelManager()

    if config.continue_training:
        logger.info('Loading model...')
        args_config = config
        train_model, config = manager.load(save_path, config.device)
        dev_model = train_model
        logger.info(config.__dict__)
        best_score = config.current_score
        if args_config.model_type != config.model_type:
            raise Exception('Expected model "%s", but "%s" found!' % (args_config.model_type, config.model_type))
    else:
        logger.info('Building model...')
        if config.model_type == 'transformers':
            train_model = manager.build(lm_name=config.lm_name, device=config.device)
            dev_model = train_model
        else:
            train_model = manager.build(word_embeddings=train_word_embeddings, word2id=train_word2id, device=config.device)
            dev_model = manager.build(word_embeddings=dev_word_embeddings, word2id=dev_word2id, device=config.device)

    # Training step
    logger.info('== MODEL TRAINING ==')
    
    logger.info('Creating metrics and optimizer...')
    optimizer = torch.optim.Adam(train_model.parameters(), lr=config.learning_rate)

    dev_dataloader = DataLoader(dev_dataset, batch_size=8, shuffle=True, num_workers=0)
    model_predictions = ModelPredictions(device=config.device)
    dev_exact_match = ExactMatchMetric()

    logger.info('Start training')
    n_batches = len(train_dataloader)

    for _ in range(config.max_epoches):
        run_loss = 0
        i_epoch = config.current_epoch
        config.current_epoch += 1

        # Set model to training mode
        train_model.train()

        for i_batch, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Get inputs
            input_ids = batch_data['input_ids'].to(device=config.device)
            token_type_ids = batch_data['token_type_ids'].to(device=config.device)
            attention_mask = batch_data['attention_mask'].to(device=config.device)
            start_token_idx = batch_data['start_token_idx'].to(device=config.device)
            end_token_idx = batch_data['end_token_idx'].to(device=config.device)

            # Inference
            loss, outputs1, outputs2 = train_model(input_ids=input_ids,
                                                   token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask,
                                                   start_positions=start_token_idx,
                                                   end_positions=end_token_idx,
                                                   return_dict=False)
            
            # Compute loss
            loss.backward()
            optimizer.step()

            run_loss += loss.cpu().data.numpy()

            if i_batch % 50 == 0:
                logger.info("Epoch %d of %d | Batch %d of %d | Loss = %.3f" % (
                        i_epoch + 1, config.max_epoches, i_batch + 1, n_batches, run_loss / (i_batch + 1)))
            
            # Clear some memory
            if config.device == 'cuda':
                del input_ids
                del token_type_ids
                del attention_mask
                del start_token_idx
                del end_token_idx
                del outputs1
                del outputs2
                gc.collect()
                torch.cuda.empty_cache()

        logger.info("Epoch %d of %d | Loss = %.3f" % (i_epoch + 1, config.max_epoches,
                                                      run_loss / len(train_dataloader)))

        logger.info('Evaluating model...')
        
        if config.model_type == 'muse':
            # In the case of MUSE, we have to transfer the weights to the dev model
            hidden_layer_weights, start_span_weights, end_span_weights = train_model.get_top_weights()
            dev_model.set_top_weights(hidden_layer_weights, start_span_weights, end_span_weights)
        
        dev_model.eval() # Set model to eval mode
        dev_predictions = model_predictions.get_predictions(dev_model, dev_dataloader, dev_dataset)
        dev_score, _ = dev_exact_match.eval(dev_dataset, dev_predictions)
        logger.info('Dev Score: %.4f | Best: %.4f' % (dev_score, best_score))
        
        if dev_score > best_score:
            logger.info('Score Improved! Saving model...')
            best_score = dev_score
            config.current_score = best_score
            manager.save(train_model, config, save_path)
    
    logger.info('End training')
