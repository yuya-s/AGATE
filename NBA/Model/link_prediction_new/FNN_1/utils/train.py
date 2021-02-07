from torch.autograd import Variable
import torch
import numpy as np
import random

def _random_subset(seq,m):
    targets=set()
    while len(targets)<m:
        x=random.choice(seq)
        targets.add(x)
    return targets

def balancer(target, mask):
    target = target[0 < mask]
    n_positive = int(target.sum())
    n_negative = int(len(target) - n_positive)
    if n_positive <= n_negative:
        sample_idx_list = np.where(target==1)[0].tolist()
        negative_idx = np.where(target==0)[0]
        sample_idx_list.extend(list(_random_subset(negative_idx, n_positive)))
    else:
        sample_idx_list = np.where(target==0)[0].tolist()
        positive_idx = np.where(target==1)[0]
        sample_idx_list.extend(list(_random_subset(positive_idx, n_negative)))
    return sample_idx_list

def train(epoch, dataloader, net, criterion, optimizer, opt):
    train_loss = 0
    net.train()
    for i, (sample_idx, data_arrays_link, dists, ind_train_A, ind_train_X, nodes_keep, A, X, label, mask) in enumerate(dataloader, 0):
        sample_idx, data_arrays_link, dists, ind_train_A, ind_train_X, nodes_keep, A, X, label, mask = sample_idx[0], data_arrays_link, dists[0], ind_train_A[0], ind_train_X[0], nodes_keep[0], A[0], X[0], label[0], mask[0]
        indic_train_ones = data_arrays_link[0][0]
        indic_train_zeros = data_arrays_link[1][0]
        indic_valid_ones = data_arrays_link[2][0]
        indic_valid_zeros = data_arrays_link[3][0]
        indic_test = data_arrays_link[4][0]
        """
        print(sample_idx, dists.shape, ind_train_A.shape, ind_train_X.shape, nodes_keep.shape, A.shape, X.shape, label.shape, mask.shape)
        tensor(46) torch.Size([3976, 3976]) torch.Size([3859, 3859]) torch.Size([3859, 35]) torch.Size([439]) torch.Size([3976, 3976]) torch.Size([3976, 35]) torch.Size([3976, 3976]) torch.Size([3976, 3976])

        print(indic_train_ones.shape)
        print(indic_train_zeros.shape)
        print(indic_valid_ones.shape)
        print(indic_valid_zeros.shape)
        print(indic_test.shape)
        torch.Size([2817, 2])
        torch.Size([2817, 2])
        torch.Size([313, 2])
        torch.Size([313, 2])
        torch.Size([465192, 2])
        """
        net.zero_grad()

        train_loss_ = 0
        for j in range(indic_train_ones.shape[0]):
            X_p_0 = X[indic_train_ones[j, 0]]
            X_p_1 = X[indic_train_ones[j, 1]]
            X_n_0 = X[indic_train_zeros[j, 0]]
            X_n_1 = X[indic_train_zeros[j, 1]]
            X_batch = torch.stack([torch.stack([X_p_0, X_p_1]), torch.stack([X_n_0, X_n_1])])
            output = net(X_batch)
            loss = criterion(output, torch.DoubleTensor([[1], [0]]))
            loss.backward()
            optimizer.step()
            train_loss_ += loss.item()
        train_loss += train_loss_ / (indic_train_ones.shape[0])
    train_loss /= (len(dataloader.dataset) / opt.batchSize)
    print('Train set: Average loss: {:.4f}'.format(train_loss))

    return train_loss
