import tensorflow as tf

def get_strategy(logger):
    # Detect hardware, return appropriate distribution strategy.
    strategy = tf.distribute.get_strategy()
    n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    logger.info('GPUs: %d' % n_gpus)

    logger.info('Number of replicas: %d' % strategy.num_replicas_in_sync)
    return strategy
