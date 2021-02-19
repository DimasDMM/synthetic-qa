from .. import *

def get_dataset_path(dataset_name, dataset_type):
    # SQuAD dataset
    if dataset_name == 'squad' and dataset_type == 'train':
        return get_project_path('data/squad/train-v1.1.json')
    elif dataset_name == 'squad' and dataset_type == 'dev':
        return get_project_path('data/squad/dev-v1.1.json')
    # SQuAD-2 dataset
    elif dataset_name == 'squad2' and dataset_type == 'train':
        return get_project_path('data/squad/train-v2.0.json')
    elif dataset_name == 'squad2' and dataset_type == 'dev':
        return get_project_path('data/squad/dev-v2.0.json')
    # Spanish machine-translation of SQuAD and SQuAD-2
    elif dataset_name == 'squad_tar_es' and dataset_type == 'train':
        return get_project_path('data/squad_tar/train-v1.1-es.json')
    elif dataset_name == 'squad_tar_es' and dataset_type == 'dev':
        return get_project_path('data/squad_tar/dev-v1.1-es.json')
    elif dataset_name == 'squad2_tar_es' and dataset_type == 'train':
        return get_project_path('data/squad_tar/train-v2.0-es.json')
    elif dataset_name == 'squad2_tar_es' and dataset_type == 'dev':
        return get_project_path('data/squad_tar/dev-v2.0-es.json')
    # XQuAD dataset
    elif dataset_name == 'xquad' and dataset_type == 'test':
        return {
            'xquad_ar': get_project_path('data/xquad/xquad.ar.json'),
            'xquad_de': get_project_path('data/xquad/xquad.de.json'),
            'xquad_el': get_project_path('data/xquad/xquad.el.json'),
            'xquad_en': get_project_path('data/xquad/xquad.en.json'),
            'xquad_es': get_project_path('data/xquad/xquad.es.json'),
            'xquad_hi': get_project_path('data/xquad/xquad.hi.json'),
            'xquad_ru': get_project_path('data/xquad/xquad.ru.json'),
            'xquad_th': get_project_path('data/xquad/xquad.th.json'),
            'xquad_tr': get_project_path('data/xquad/xquad.tr.json'),
            'xquad_vi': get_project_path('data/xquad/xquad.vi.json'),
            'xquad_zh': get_project_path('data/xquad/xquad.zh.json'),
        }
    # Synthetic data
    elif dataset_name == 'synthetic' and dataset_type == 'train':
        return get_project_path('data/synthetic/train-synthetic-v2.json')
    elif dataset_name == 'synthetic' and dataset_type == 'dev':
        return get_project_path('data/squad_tar/dev-v1.1-es.json') # Note that it uses SQuAD-v1-spanish as dev set
    else:
        raise Exception('Unknown dataset "%s" and/or type "%s"' % (dataset_name, dataset_type))
