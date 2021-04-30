import io
import numpy as np
import os

from . import *
from .. import *

def load_embeddings(lang_code, embeddings_path, nmax=1000000, data_path='data/',
                    unknown_token=UNKNOWN_TOKEN, sep_token=SEP_TOKEN, pad_token=PAD_TOKEN):
    embeddings_path = get_project_path(data_path, embeddings_path)
    
    vectors = []
    word2id = {}
    emb_path = os.path.join(embeddings_path, 'wiki.multi.%s.vec' % lang_code)
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ', dtype='float32')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax - 3:
                break

    # Add token to separate items
    vect_random = np.ones(vectors[-1].shape[-1], dtype='float32')
    vectors.append(vect_random)
    word2id[sep_token] = len(word2id)
    
    # Add token for padding
    vect_random = np.zeros(vectors[-1].shape[-1], dtype='float32')
    vectors.append(vect_random)
    word2id[pad_token] = len(word2id)
    
    # Add token for unknown words
    vect_random = np.random.random(vectors[-1].shape[-1]).astype('float32')
    vectors.append(vect_random)
    word2id[unknown_token] = len(word2id)
    
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id
