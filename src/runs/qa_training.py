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


def run_qa_training(logger, config: Config):
    logger.info(config.__dict__)
    save_path = get_project_path('artifacts', config.ckpt_name)
    best_score = 0.

    logger.info('Base path: %s' % BASE_PATH)

    logger.info('== SET UP ==')

    logger.info('Loading tokenizer')
    tokenizer = get_tokenizer(lm_name=config.lm_name, lowercase=(not config.cased))

    logger.info('Loading dataset: %s' % config.dataset_name)
    squad_preprocess = SquadPreprocess(tokenizer, max_length=config.max_length)

    dataset_path = get_dataset_path(config.dataset_name, 'train')
    train_dataset = SquadDataset(squad_preprocess, dataset_path, save_contexts=False)
    train_skipped = train_dataset.get_skipped_items()
    logger.info('- Train data: %d (skipped: %d)' % (len(train_dataset), len(train_skipped)))

    dataset_path = get_dataset_path(config.dataset_name, 'dev')
    dev_dataset = SquadDataset(squad_preprocess, dataset_path)
    dev_skipped = dev_dataset.get_skipped_items()
    logger.info('- Dev data: %d (skipped: %d)' % (len(dev_dataset), len(dev_skipped)))
    
    logger.info('Creating data loaders...')
    set_seed(DEFAULT_SEED)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    # Load trained model or build a new one
    manager = ModelManager()
    if config.continue_training:
        logger.info('Loading model...')
        model, config = manager.load(save_path, config.device)
        logger.info(config.__dict__)
        best_score = config.current_score
    else:
        logger.info('Building model...')
        model = manager.build(config.lm_name, device=config.device)

    # Training step
    logger.info('== MODEL TRAINING ==')
    
    logger.info('Creating metrics...')
    #train_exact_match = ExactMatch(train_dataset, device=config.device)
    dev_exact_match = ExactMatch(dev_dataset, device=config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    n_batches = len(train_dataloader)

    logger.info('Start training')

    for _ in range(config.max_epoches):
        run_loss = 0
        i_epoch = config.current_epoch
        config.current_epoch += 1

        # Set model to training mode
        model.train()

        for i_batch, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Get inputs
            input_ids = batch_data['input_ids'].to(device=config.device)
            token_type_ids = batch_data['token_type_ids'].to(device=config.device)
            attention_mask = batch_data['attention_mask'].to(device=config.device)
            start_token_idx = batch_data['start_token_idx'].to(device=config.device)
            end_token_idx = batch_data['end_token_idx'].to(device=config.device)

            # Inference
            loss, outputs1, outputs2 = model(input_ids=input_ids,
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
        model.eval() # Set model to eval mode

        #train_score = train_exact_match.eval(model)
        dev_score = dev_exact_match.eval(model)

        #logger.info('Train Score: %.4f | Dev Score: %.4f | Best: %.4f' % (
        #            train_score, dev_score, best_score))
        logger.info('Dev Score: %.4f | Best: %.4f' % (dev_score, best_score))
        
        if dev_score > best_score:
            logger.info('Score Improved! Saving model...')
            best_score = dev_score
            config.current_score = best_score
            manager.save(model, config, save_path)
    
    logger.info('End training')
