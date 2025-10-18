import random

import torch

def obatin_train_test_tasks(dataset):
    tox21_train_tasks = list(range(9))
    tox21_test_tasks = list(range(9, 12))
    sider_train_tasks = list(range(21))
    sider_test_tasks = list(range(21, 27))
    muv_train_tasks = list(range(12))
    muv_test_tasks = list(range(12, 17))
    if dataset == "sider":
        return sider_train_tasks, sider_test_tasks
    elif dataset == "tox21":
        return tox21_train_tasks, tox21_test_tasks
    elif dataset == "muv":
        return muv_train_tasks, muv_test_tasks
    else:
        return None, None


def obtain_distr_list(dataset):
    '''Set the number of samples here.'''
    if dataset == "sider":
        return [[684, 743], [431, 996], [1405, 22], [551, 876], [276, 1151], [430, 997], [129, 1298], [1176, 251],
                [403, 1024], [700, 727], [1051, 376], [135, 1292], [1104, 323], [1214, 213], [319, 1108], [542, 885],
                [109, 1318], [1174, 253], [421, 1006], [367, 1060], [411, 1016], [516, 911], [1302, 125], [768, 659],
                [439, 988], [123, 1304], [481, 946]]


    elif dataset == "tox21":
        return [[6956, 309], [6521, 237], [5781, 768], [5521, 300], [5400, 793], [6605, 350], [6264, 186], [4890, 942],
                [6808, 264], [6095, 372], [4892, 918], [6351, 423]]


    elif dataset == "muv":
        return [[14813, 27],[14705, 29],[14698, 30], [14593, 30], [14873, 29], [14572, 29], [14614, 30], [14383, 28],
                [14807, 29],[14654, 28],[14662, 29], [14615, 29], [14637, 30], [14681, 30], [14622, 29], [14745, 29],
                [14722, 24]]
    else:
        return None

def sample_inds(data, size):
    len_data = len(data)
    if len_data >= size:
        return random.sample(data, size)
    else:
        return random.sample(data, len_data) + sample_inds(data, size - len_data)


def sample_meta_datasets(data, dataset, task, n_shot, n_query):
    distri_list = obtain_distr_list(dataset)
    thresh = distri_list[task][0]
    neg_sample = sample_inds(range(0, thresh), n_shot)
    pos_sample = sample_inds(range(thresh, len(data)), n_shot)

    s_list_1 = neg_sample[:n_shot] + pos_sample[:n_shot]

    l = [i for i in range(0, len(data)) if i not in s_list_1]
    random.shuffle(l)

    q_sample = sample_inds(l, n_query)
    q_list_1 = q_sample[:n_query]


    s_adapt = data[torch.tensor(s_list_1)]
    q_adapt = data[torch.tensor(q_list_1)]

    return s_adapt, q_adapt

def sample_test_datasets(data, dataset, task, n_shot, n_query, update_step=1):
    distri_list = obtain_distr_list(dataset)
    thresh = distri_list[task][0]

    neg_sample = sample_inds(range(0, thresh), n_shot)
    pos_sample = sample_inds(range(thresh, len(data)), n_shot)

    s_list = neg_sample + pos_sample

    q_list = [i for i in range(0, len(data)) if i not in s_list]

    s_data = data[torch.tensor(s_list)]
    q_data = data[torch.tensor(q_list)]

    q_sample = sample_inds(q_list, update_step * n_query)
    q_data_adapt = data[torch.tensor(q_sample)]

    return s_data, q_data,q_data_adapt
