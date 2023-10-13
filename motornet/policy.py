import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PolicyGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = 1

        self.gru = th.nn.GRU(input_dim, hidden_dim, 1, batch_first=True)
        self.fc = th.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = th.nn.Sigmoid()

        # the default initialization in torch isn't ideal
        for name, param in self.named_parameters():
            if name == "gru.weight_ih_l0":
                th.nn.init.xavier_uniform_(param)
            elif name == "gru.weight_hh_l0":
                th.nn.init.orthogonal_(param)
            elif name == "gru.bias_ih_l0":
                th.nn.init.zeros_(param)
            elif name == "gru.bias_hh_l0":
                th.nn.init.zeros_(param)
            elif name == "fc.weight":
                th.nn.init.xavier_uniform_(param)
            elif name == "fc.bias":
                th.nn.init.constant_(param, -5.)
            else:
                raise ValueError

        self.to(device)

    def forward(self, x, h0):
        y, h = self.gru(x[:, None, :], h0)
        u = self.sigmoid(self.fc(y)).squeeze(dim=1)
        return u, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden


class ModularPolicyGRU(nn.Module):
    def __init__(self, input_size: int, module_size: list, output_size: int,
                 vision_mask: list, proprio_mask: list, task_mask: list,
                 connectivity_mask: np.ndarray, output_mask: list,
                 vision_dim: list, proprio_dim: list, task_dim: list,
                 device=th.device("cpu"), random_seed=None, activation='tanh'):
        super(ModularPolicyGRU, self).__init__()

        # Store class info
        hidden_size = sum(module_size)
        self.activation = activation
        self.device = device
        self.num_modules = len(module_size)
        self.input_size = input_size
        self.module_size = module_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Set the random seed
        if random_seed:
            self.rng = np.random.default_rng(seed=random_seed)
        else:
            self.rng = np.random.default_rng()

        # Make sure that all sizes check out
        assert len(vision_mask) == self.num_modules
        assert len(proprio_mask) == self.num_modules
        assert len(task_mask) == self.num_modules
        assert connectivity_mask.shape[0] == connectivity_mask.shape[1] == self.num_modules
        assert len(output_mask) == self.num_modules
        assert len(vision_dim) + len(proprio_dim) + len(task_dim) == self.input_size

        # Initialize all GRU parameters
        # Update gate
        self.Wz = nn.Parameter(th.cat((nn.init.xavier_uniform_(th.Tensor(hidden_size, input_size), gain=1),
                                       nn.init.orthogonal_(th.Tensor(hidden_size, hidden_size))), dim=1))
        self.bz = nn.Parameter(th.zeros(hidden_size))
        # Reset gate
        self.Wr = nn.Parameter(th.cat((nn.init.xavier_uniform_(th.Tensor(hidden_size, input_size), gain=1),
                                       nn.init.orthogonal_(th.Tensor(hidden_size, hidden_size))), dim=1))
        self.br = nn.Parameter(th.zeros(hidden_size))
        # Candidate hidden state
        self.Wh = nn.Parameter(th.cat((nn.init.xavier_uniform_(th.Tensor(hidden_size, input_size), gain=1),
                                       nn.init.orthogonal_(th.Tensor(hidden_size, hidden_size))), dim=1))
        self.bh = nn.Parameter(th.zeros(hidden_size))

        # Initialize all output parameters
        self.Y = nn.Parameter(nn.init.xavier_uniform_(th.Tensor(output_size, hidden_size), gain=1))
        self.bY = nn.Parameter(nn.init.constant_(th.Tensor(output_size), -4.))

        # create indices for indexing modules
        module_dims = []
        for m in range(self.num_modules):
            if m > 0:
                module_dims.append(np.arange(module_size[m]) + module_dims[-1][-1] + 1)
            else:
                module_dims.append(np.arange(module_size[m]))
        self.module_dims = module_dims

        # Create sparsity mask for GRU
        h_probability_mask = np.zeros((hidden_size, input_size + hidden_size), dtype=np.float32)
        i_module = 0
        j_module = 0
        for i in range(self.Wz.shape[0]):
            for m in range(self.num_modules):
                if i in module_dims[m]:
                    i_module = m
            for j in range(self.Wz.shape[1]):
                module_type = 'hidden'
                if j < input_size:
                    if j in vision_dim:
                        module_type = 'vision'
                    elif j in proprio_dim:
                        module_type = 'proprio'
                    elif j in task_dim:
                        module_type = 'task'
                if module_type == 'hidden':
                    for m in range(self.num_modules):
                        if j in (module_dims[m] + input_size):
                            j_module = m
                        h_probability_mask[i, j] = connectivity_mask[i_module, j_module]
                elif module_type == 'vision':
                    h_probability_mask[i, j] = vision_mask[i_module]
                elif module_type == 'proprio':
                    h_probability_mask[i, j] = proprio_mask[i_module]
                elif module_type == 'task':
                    h_probability_mask[i, j] = task_mask[i_module]

        # Create sparsity mask for output
        y_probability_mask = np.zeros((output_size, hidden_size), dtype=np.float32)
        j_module = 0
        for j in range(self.Y.shape[1]):
            for m in range(self.num_modules):
                if j in module_dims[m]:
                    j_module = m
            y_probability_mask[:, j] = output_mask[j_module]

        # Initialize masks with desired sparsity
        mask_connectivity = self.rng.binomial(1, h_probability_mask)
        mask_output = self.rng.binomial(1, y_probability_mask)

        # Masks for weights and biases
        self.mask_Wz = nn.Parameter(th.tensor(mask_connectivity), requires_grad=False)
        self.mask_Wr = nn.Parameter(th.tensor(mask_connectivity), requires_grad=False)
        self.mask_Wh = nn.Parameter(th.tensor(mask_connectivity), requires_grad=False)
        self.mask_Y = nn.Parameter(th.tensor(mask_output), requires_grad=False)
        # No need to mask any biases for now
        self.mask_bz = nn.Parameter(th.ones_like(self.bz), requires_grad=False)
        self.mask_br = nn.Parameter(th.ones_like(self.br), requires_grad=False)
        self.mask_bh = nn.Parameter(th.ones_like(self.bh), requires_grad=False)
        self.mask_bY = nn.Parameter(th.ones_like(self.bY), requires_grad=False)

        # Zero out weights and biases that we don't want to exist
        self.Wz = nn.Parameter(th.mul(self.Wz, self.mask_Wz.data))
        self.Wr = nn.Parameter(th.mul(self.Wr, self.mask_Wr.data))
        self.Wh = nn.Parameter(th.mul(self.Wh, self.mask_Wh.data))
        self.Y = nn.Parameter(th.mul(self.Y, self.mask_Y.data))
        self.bz = nn.Parameter(th.mul(self.bz, self.mask_bz.data))
        self.br = nn.Parameter(th.mul(self.br, self.mask_br.data))
        self.bh = nn.Parameter(th.mul(self.bh, self.mask_bh.data))
        self.bY = nn.Parameter(th.mul(self.bY, self.mask_bY.data))

        # Registering a backward hook to apply mask on gradients during backward pass
        self.Wz.register_hook(lambda grad: grad * self.mask_Wz.data)
        self.Wr.register_hook(lambda grad: grad * self.mask_Wr.data)
        self.Wh.register_hook(lambda grad: grad * self.mask_Wh.data)
        self.bz.register_hook(lambda grad: grad * self.mask_bz.data)
        self.br.register_hook(lambda grad: grad * self.mask_br.data)
        self.bh.register_hook(lambda grad: grad * self.mask_bh.data)
        self.Wh.register_hook(lambda grad: grad * self.mask_Wh.data)
        self.Y.register_hook(lambda grad: grad * self.mask_Y.data)
        self.bY.register_hook(lambda grad: grad * self.mask_bY.data)

        self.to(device)

    def forward(self, x, h_prev):
        # TODO: Add time delays between neural modules. This would be possible by looping through an h_prev time
        #  buffer and then concatenating h together from the results of different time delays
        concat = th.cat((x, h_prev), dim=1)
        z = th.sigmoid(F.linear(concat, self.Wz, self.bz))
        r = th.sigmoid(F.linear(concat, self.Wr, self.br))
        concat_hidden = th.cat((x, r * h_prev), dim=1)
        h_tilda = th.tanh(F.linear(concat_hidden, self.Wh, self.bh))
        if self.activation == 'rect_tanh':
            h_tilda = th.max(th.zeros_like(h_tilda), h_tilda)
        h = (1 - z) * h_prev + z * h_tilda
        y = th.sigmoid(F.linear(h, self.Y, self.bY))

        return y, h
