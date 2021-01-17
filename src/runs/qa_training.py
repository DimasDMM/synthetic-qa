import json

from .. import *
from ..data.loader import *
from ..data.tokenizers import *
from ..models.tf.callbacks import *
from ..models.tf.metrics import *
from ..models.tf.qa import *
from ..utils.config import *

def run_qa_training(logger, strategy, config: Config):
    logger.info('== SET UP ==')

    artifacts_path = get_project_path('artifacts')

    logger.info('Loading tokenizer')
    tokenizer = get_tokenizer(lm_name=config.lm_name, lowercase=(not config.cased))

    logger.info('Loading dataset: %s' % config.dataset_name)
    squad_training = build_squad_dataset('%s-train' % config.dataset_name, tokenizer, config.max_length)
    X_train, y_train = split_inputs_targets(squad_training)

    squad_testing = build_squad_dataset('%s-test' % config.dataset_name, tokenizer, config.max_length)
    X_test, y_test = split_inputs_targets(squad_testing)

    logger.info('Building model...')
    pretrained_model = os.path.join(artifacts_path, config.lm_name)
    manager = ModelManager(pretrained_model)
    with strategy.scope():
        # Build model
        model = manager.build()

        # Build callbacks
        loss_logger = ValLossLogger(logger)
        exact_match = ExactMatch(X_test, y_test, squad_testing)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_exact_match', patience=config.early_stop, verbose=0)

        ckpt_path = os.path.join(artifacts_path, config.ckpt_name)
        m_ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_exact_match', verbose=0,
                                                    save_best_only=True, save_weights_only=True)

        # Add loss functions and compile model
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
        optimizer = tf.keras.optimizers.Adam(lr=config.learning_rate)
        model.compile(optimizer=optimizer,
                      loss={'start_logits': loss_fn, 'end_logits': loss_fn},
                      metrics=['accuracy'])

    logger.info('== TRAINING ==')
    with strategy.scope():
        model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=[loss_logger, exact_match, early_stop, m_ckpt],
        )

    logger.info('Done')
