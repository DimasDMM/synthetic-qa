import tensorflow as tf

def get_strategy(logger):
    # Detect hardware, return appropriate distribution strategy.
    # You can see that it is pretty easy to set up.
    try:
        # TPU detection: no parameters necessary if TPU_NAME environment
        # variable is set (always set in Kaggle)
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        logger.info('Running on TPU ', tpu.master())
    except ValueError:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()
        n_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
        logger.info('GPUs:', n_gpus)

    logger.info('Number of replicas:', strategy.num_replicas_in_sync)
    return strategy
