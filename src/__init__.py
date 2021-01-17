import numpy as np
import os
import tensorflow as tf

np.random.seed(42)

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

def get_project_path(partial_path):
    return os.path.join(BASE_PATH, partial_path)

def setup_logger(logger, to_file=False):
    if to_file:
        logger.basicConfig(filename=get_project_path('logger.log'),
                           filemode='a',
                           format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                           datefmt='%H:%M:%S',
                           level=logger.DEBUG)
    else:
        logger.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                           datefmt='%H:%M:%S',
                           level=logger.DEBUG)

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
        logger.info('Running on TPU: ' + tpu.master())
    except ValueError:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
            logger.info('Running on GPU')
        else:
            logger.info('Running on CPU')
        strategy = tf.distribute.get_strategy()

    #print('Number of replicas:', strategy.num_replicas_in_sync)
    return strategy