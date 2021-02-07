import torch.nn as nn
import torch

class FNN(nn.Module):

    def __init__(self, opt):
        super(FNN, self).__init__()

        self.batchSize = opt.batchSize
        self.state_dim = opt.state_dim
        self.n_node = opt.n_node
        self.mlp = nn.Linear(self.state_dim, self.state_dim)

        self.out = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

        self._initialization()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def cosine_sim(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim = (a_norm * b_norm).sum(1)[:, None]
        return sim

    def forward(self, prop_state):
        """"
        prop_state :(2 (pos, neg), 2 (source, target), state_dim)
        """
        output = self.mlp(prop_state)  # (2, 2, state_dim)
        output = self.cosine_sim(output[:, 0], output[:, 1]) # (2, 1)
        output = self.out(output) # (2, 1)
        return output