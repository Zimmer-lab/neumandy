
import numpy as np
import slim
import torch


class CDLDSModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_points, num_subdyn, control_size=1, sparsity_threshold=0.1):
        super().__init__()

        self.control_size = control_size

        self.coeffs = torch.nn.Parameter(torch.tensor(
            np.random.rand(num_subdyn, time_points), requires_grad=True))

        self.F = torch.nn.ParameterList()

        # Control matrix B (learnable)
        self.B = torch.nn.Linear(
            control_size, input_size, bias=False)
        
        self.bias = torch.nn.Linear(
            num_subdyn, input_size, bias=False)
        

        # Learnable control input U (initialize with small values)
        # U will have the shape: (batch_size, seq_len, control_size)
        self.U = torch.nn.Parameter(torch.randn(
            control_size, time_points, requires_grad=True))

        for _ in range(num_subdyn):
            
            f_i = torch.nn.Linear(input_size, output_size, bias=False)
            self.F.append(f_i)

    def forward(self, x, idx):

        batch_size = x.shape[0]

        dim = 1 if batch_size > 1 else 0
        
        x = torch.stack([self.coeffs[i, idx].unsqueeze(dim)*f_i(x) + (self.B(self.effective_U[:, idx].permute(*torch.arange(self.effective_U[:, idx].ndim - 1, -1, -1)))) + self.bias(self.coeffs[:, idx].permute(*torch.arange(self.coeffs[:, idx].ndim - 1, -1, -1)))
                         for i, f_i in enumerate(self.F)]).sum(dim=0)

        return x

    @ property
    def effective_U(self):
        return torch.relu(self.U)

    def control_sparsity_loss(self):
        # L1 regularization for sparsity on the control matrix U
        return torch.sum(torch.abs(self.effective_U[:self.control_size, :]))
    
    def coeff_sparsity_loss(self):
        return torch.sum(torch.abs(self.coeffs))
    
    

class DeepDLDS(torch.nn.Module):
    """Deep Dynamic Linear Dynamical System (DeepDLDS) model with spectral linear layers.
    Args:
        input_size (int): Size of the input features.
        output_size (int): Size of the output features. 
        num_subdyn (int): Number of sub-dynamics.
        time_points (int): Number of time points in the sequence.
        fixed_point_change (bool): If True, the model will learn fixed point changes.
        softmax_temperature (float): Temperature for softmax normalization of coefficients.
    """

    def __init__(self, input_size, output_size, num_subdyn, time_points, fixed_point_change=False, softmax_temperature=1):
        super().__init__()

        self.softmax_temperature = softmax_temperature

        self.F = torch.nn.ParameterList()  # can't be a simple list

        self.coeffs = torch.nn.Parameter(torch.tensor(
            np.random.rand(num_subdyn, time_points), requires_grad=True))


        for i in range(num_subdyn):


            f_i = slim.linear.SpectralLinear(
                input_size, input_size, bias=fixed_point_change, sigma_max=1.0, sigma_min=0)

            self.F.append(f_i)

    def forward(self, x_t, t):

        y_t = torch.stack([self.coeffs[i, t]*f_i(x_t.unsqueeze(0))  
                          for i, f_i in enumerate(self.F)]).sum(dim=0)

        return y_t

    @property
    def soft_coeffs(self):
        return torch.nn.functional.softmax(self.coeffs / self.softmax_temperature, dim=0)

    def multi_step(self):

        _coeffs = self.soft_coeffs
        Y = torch.zeros((self.step_ahead, self.input_size))
        y0 = self.batch
        Y[0, :] = y0

        for t in range(self.step_ahead):
            # combination of all f_i * c_i
            y = torch.stack([_coeffs[i, t]*f_i(y0)
                            for i, f_i in enumerate(self.F)]).sum(dim=0)
            Y[t, :] = y
            y0 = y

        return Y


class SimpleNN(torch.nn.Module):
    """a simple fully connected neural network with ReLU activation functions and a linear output layer
    """

    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        in_size = input_size
        self.layers = []

        # define multiple layers in a loop
        for hidden_size in hidden_sizes:
            # linear layer computes output = input * weights + bias
            self.layers.append(torch.nn.Linear(in_size, hidden_size))
            self.layers.append(torch.nn.ReLU())
            in_size = hidden_size

        self.layers.append(torch.nn.Linear(in_size, output_size))
        # no final activation function because we have a regression problem

        self.network = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        # this forward function is always called when we call the model (it's somewhere in __call__ method)
        return self.network(x)
