import torch.nn as nn
import torch
from layers.ST_blocks import ST_blocks
from layers.GatedCNN import GatedCNN
import os
import sys
current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append( str(current_dir) + '/../../../' )

from setting_param import L
from setting_param import all_node_num_expanded as all_node_num

def get_A_in_per_batch(A, batch):
    A_in = A[batch][0]  # (3, max_nnz)
    nnz = int(A_in[0].sum().item())
    A_in = A_in[:, :nnz]  # (3, nnz)
    return A_in

def get_adj_per_t(A_in, t):
    col = A_in[2]
    idx = (t * all_node_num) <= col
    idx = idx * (col < ((t + 1) * all_node_num))
    A_t = A_in[:, idx]  # (3, nnz_per_t)
    A_t[2] = A_t[2] % all_node_num  # adjacency matrix per t
    return A_t

def get_cur_adj(A_t):
    cur_adj = {}
    cur_adj['vals'] = A_t[0]  # (nnz_per_t, )
    cur_adj['idx'] = A_t[1:].t().long()  # (nnz_per_t, 2)
    return cur_adj

def make_sparse_eye(size):
    eye_idx = torch.arange(size)
    eye_idx = torch.stack([eye_idx,eye_idx],dim=1).t()
    vals = torch.ones(size)
    eye = torch.sparse.FloatTensor(eye_idx,vals,torch.Size([size,size]))
    return eye

def normalize_adj(adj, num_nodes):
    '''
    takes an adj matrix as a dict with idx and vals and normalize it by:
        - adding an identity matrix,
        - computing the degree vector
        - multiplying each element of the adj matrix (aij) by (di*dj)^-1/2
    A¥hat{A}(ー(vals))
    1
    '''
    idx = adj['idx']
    vals = adj['vals']
    sp_tensor = torch.sparse.FloatTensor(idx.t(), vals.type(torch.float), torch.Size([num_nodes, num_nodes]))
    sparse_eye = make_sparse_eye(num_nodes)
    sp_tensor = sparse_eye + sp_tensor
    idx = sp_tensor._indices()
    vals = sp_tensor._values()
    degree = torch.sparse.sum(sp_tensor, dim=1).to_dense()
    di = degree[idx[0]]
    dj = degree[idx[1]]
    vals = vals * ((di * dj) ** -0.5)
    return {'idx': idx.t(), 'vals': vals}

def make_sparse_tensor(adj,tensor_type,torch_size):
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size*2)

    if tensor_type == 'float':
        test = torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
        return torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
    elif tensor_type == 'long':
        return torch.sparse.LongTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.long),
                                      tensor_size)
    else:
        raise NotImplementedError('only make floats or long sparse tensors')

def sparse_prepare_tensor(tensor,torch_size):
    tensor = make_sparse_tensor(tensor,
                                tensor_type = 'float',
                                torch_size = torch_size)
    return tensor

def get_A_hat(A):
    """
    1. （）
    2.
    """
    batch_adj_list = []
    for batch in range(A.shape[0]):
        A_in = get_A_in_per_batch(A, batch) # (3, nnz)
        for t in range(L):
            A_t = get_adj_per_t(A_in, t) # (3, nnz_per_t)
            if t == 0:
                A_ = A_t
            else:
                A_ = torch.cat((A_, A_t), 1) # (3, nnz)
        cur_adj = get_cur_adj(A_)
        cur_adj = normalize_adj(adj=cur_adj, num_nodes=all_node_num) # ー
        cur_adj = sparse_prepare_tensor(cur_adj, torch_size=[all_node_num])
        batch_adj_list.append(cur_adj)
    A_hat = torch.stack(batch_adj_list, 0)
    return A_hat


class STGCN(nn.Module):

    def __init__(self, opt, kernel_size=2, n_blocks=1, state_dim_bottleneck=64, annotation_dim_bottleneck=64):
        super(STGCN, self).__init__()

        self.batchSize = opt.batchSize
        self.state_dim = opt.state_dim
        self.n_node = opt.n_node

        # ST-Block
        self.st_blocks = ST_blocks(opt, kernel_size=kernel_size, n_blocks=n_blocks, state_dim_bottleneck=state_dim_bottleneck, annotation_dim_bottleneck=annotation_dim_bottleneck)
        opt.L = opt.L - 2 * n_blocks * (kernel_size - 1)

        #  GCNN ()
        self.gcnn = GatedCNN(opt, in_channels=self.state_dim, out_channels=self.state_dim, kernel_size=opt.L)
        opt.L = 1

        #  FC ()
        self.out = nn.Sequential(
            nn.Linear(self.state_dim, opt.output_dim),
            nn.Sigmoid()
        )

        opt.L = opt.init_L
        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, prop_state, annotation, A):
        """"
        prop_state :(batch_size, n_node, L, state_dim)
        annotation :(batch_size, n_node, L, annotation_dim)
        A          :(batch_size, 2, 3, max_nnz)
        A_hat      :(batch_size, n_node, n_node) *sparse
        output     :(batch_size, n_node, output_dim)
        """
        A_hat = get_A_hat(A)
        output = self.st_blocks(prop_state, annotation, A_hat)             # (batch_size, n_node, L-2*n_blocks(kernel_size-1), state_dim)
        if output.shape[2] > 1:
            output = self.gcnn(output)                                     # (batch_size, n_node, 1, state_dim)
        output = output.view(self.batchSize, self.n_node, self.state_dim)  # (batch_size, n_node, state_dim)
        output = self.out(output)                                          # (batch_size, n_node, output_dim)
        return output