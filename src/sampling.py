import numpy as np
import matplotlib.pyplot as plt


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_split_dataset(dataset, num_users, split):
    print("splitting the dataset by: ", split)
    num_items = int(len(dataset) / num_users * split)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

    

def cifar_noniid_dirichlet(dataset, num_users, beta=0.4, labels=None, vis=False):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    if labels is None:
        labels = np.array(dataset.targets)

    dict_users = dirichlet_sampling(
        labels, num_users, beta, vis=vis, fig_name="y_shift"
    )
    for user, indices in dict_users.items():
        user_labels = labels[list(indices)]
        unique_classes = np.unique(user_labels)
        print(f"User {user} has {len(unique_classes)} unique classes.")
        
    return dict_users

def dirichlet_sampling(labels, num_users, alpha, vis=False, fig_name="cluster"):
    """
    Sort labels and use dirichlet resampling to split the labels
    :param dataset:
    :param num_users:
    :return:
    """
    K = len(np.unique(labels))
    N = labels.shape[0]
    threshold = 0.5
    min_require_size = N / num_users * (1 - threshold)
    max_require_size = N / num_users * (1 + threshold)
    min_size, max_size = 0, 1e6
    iter_idx = 0

    while (
        min_size < min_require_size or max_size > max_require_size
    ) and iter_idx < 1000:
        idx_batch = [[] for _ in range(num_users)]
        plt.clf()
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))

            # avoid adding over
            proportions = np.array(
                [
                    p * (len(idx_j) < N / num_users)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]

            min_size = min([len(idx_j) for idx_j in idx_batch])
            max_size = max([len(idx_j) for idx_j in idx_batch])

        iter_idx += 1

    # divide and assign
    dict_users = {i: idx for i, idx in enumerate(idx_batch)}
    return dict_users
