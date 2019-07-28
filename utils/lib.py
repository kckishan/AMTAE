from texttable import Texttable
import networkx as nx
import pandas as pd
import torch
import numpy as np
from scipy import sparse, io
from sklearn.model_selection import train_test_split


def table_printer(args):
    """
    Print the parameters of the model in a Tabular format
    Parameters
    ---------
    args: argparser object
        The parameters used for the model
    """
    args = vars(args)
    keys = sorted(args.keys())
    table = Texttable()
    table.set_precision(4)
    table.add_rows([["Parameter", "Value"]] +
                   [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(table.draw())


def load_network(filename, num_nodes, mtrx='adj'):
    """
    Load the adjacency list to a adjacency matrix of shape [num_nodes, num_nodes]
    Parameters
    ----------
    filename : string
        Path to the adjacency list file
    num_nodes : int
        Number of nodes in the network
    mtrx : string
        Description of parameter `mtrx` (the default is 'adj').

    Returns
    -------
    A: Matrix, shape [num_nodes, num_nodes]
        The adjacency matrix of the network

    """
    A = 0
    if mtrx == 'adj':
        i, j, val = np.loadtxt(filename).T
        i = i.astype(int)
        j = j.astype(int)
        A = sparse.coo_matrix((val, (i-1, j-1)), shape=(num_nodes, num_nodes), dtype=np.float32)
        A = A.todense()
        A = np.squeeze(np.asarray(A))
        if A.min() < 0:
            print("### Negative entries in the matrix are not allowed!")
            A[A < 0] = 0
            print("### Matrix converted to nonnegative matrix.")
            print()
        if (A.T == A).all():
            pass
        else:
            print("### Matrix not symmetric!")
            A = A + A.T
            print("### Matrix converted to symmetric.")
        A = A - np.diag(np.diag(A))
    else:
        print("### Wrong mtrx type. Possible: {'adj', 'inc'}")

    return A


def load_networks(path_to_string_nets, num_nodes, mtrx='adj'):
    """
    Load networks from adjacency lists
    Parameters
    ----------
    path_to_string_nets : path

    mtrx : strings
        Type of the matrix (the default is 'adj').

    Returns
    -------
    Nets: list
        The list of adjacency matrix

    """
    Nets = []
    adj_mat = []
    string_nets = ['neighborhood', 'fusion', 'cooccurence',
                   'coexpression', 'experimental', 'database']
    for net in string_nets:
        filename = path_to_string_nets + 'yeast_string_' + net + '_adjacency.txt'
        Net = load_network(filename, num_nodes, mtrx)
        # print(np.count_nonzero(Net))
        np.fill_diagonal(Net, 1)
        adj_mat.append(Net)
        Nets.append(torch.from_numpy(Net))
    adjs = torch.stack(Nets, dim=2)
    A = torch.sum(adjs, dim=1)
    A = A/A.max(dim=1, keepdim=True)[0]
    A[torch.isnan(A)] = 0
    return adj_mat, A


def split_data(X, test_size=0.2, noise_factor=0.5, std=1.0):
    """
    Splits the dataset into train and test.

    Parameters
    ----------
    X : list or matrix
        The list of adjacency matrix
    test_size : float
        The percentage of samples as test samples (the default is 0.2).
    noise_factor : float
        The value that weights the random noise (the default is 0.5).
    std : float
        Standard deviation of the Gaussian noise distribution. (the default is 1.0).

    Returns
    -------
    X_train_noisy: list or matrix
        Training data with added noise
    X_train: list or matrix
        Training data output
    X_test_noisy:  list or matrix
        Testing set with added noise
    X_test:   list or matrix
        Testing set

    """
    if isinstance(X, list):
        Xs = train_test_split(*X, test_size=test_size)
        X_train = []
        X_test = []
        for jj in range(0, len(Xs), 2):
            X_train.append(Xs[jj])
            X_test.append(Xs[jj+1])
        X_train_noisy = list(X_train)
        X_test_noisy = list(X_test)
        for ii in range(0, len(X_train)):
            X_train_noisy[ii] = X_train_noisy[ii] + noise_factor * \
                np.random.normal(loc=0.0, scale=std, size=X_train[ii].shape)
            X_test_noisy[ii] = X_test_noisy[ii] + noise_factor * \
                np.random.normal(loc=0.0, scale=std, size=X_test[ii].shape)
            X_train_noisy[ii] = np.clip(X_train_noisy[ii], 0, 1)
            X_test_noisy[ii] = np.clip(X_test_noisy[ii], 0, 1)
    else:
        X_train, X_test = train_test_split(X, test_size=0.2)
        X_train_noisy = X_train.copy()
        X_test_noisy = X_test.copy()
        X_train_noisy = X_train_noisy + noise_factor * \
            np.random.normal(loc=0.0, scale=std, size=X_train.shape)
        X_test_noisy = X_test_noisy + noise_factor * \
            np.random.normal(loc=0.0, scale=std, size=X_test.shape)
        X_train_noisy = np.clip(X_train_noisy, 0, 1)
        X_test_noisy = np.clip(X_test_noisy, 0, 1)

    return X_train_noisy, X_train, X_test_noisy, X_test


def list_to_gpu(X, device):
    """
    Transfer the data to CPU/GPU

    Parameters
    ----------
    X : list or array
        The dataset to load
    device : cpu or GPU
        Description of parameter `device`.

    Returns
    -------
    X: list or array
        The dataset on CPU/GPU
    """
    if isinstance(X, list):
        if isinstance(X[0], np.ndarray):
            return [torch.from_numpy(X[i]).type(torch.FloatTensor).to(device) for i in range(len(X))]
        else:
            return [X[i].type(torch.FloatTensor).to(device) for i in range(len(X))]
    else:
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X).type(torch.FloatTensor).to(device)
        else:
            return X.type(torch.FloatTensor).to(device)


def list_to_cpu(X):
    """
    Transfer the data to CPU

    Parameters
    ----------
    X : list or array
        The dataset to load
    Returns
    -------
    X: list or array
        The dataset on CPU
    """
    if isinstance(X, list):
        return [X[i].cpu().detach() for i in range(len(X))]
    else:
        return X.cpu().detach()


def criterion_for_list(criterion, target, output):
    loss = 0
    for i in range(len(output)):
        l = criterion(output[i], target[i])
        loss += l
    return loss
