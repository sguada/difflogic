import torch
from pallas import autograd as pallas_autograd
import numpy as np
import jax
import jax.numpy as jnp
from jax.nn import softmax, one_hot
from .functional import bin_op_s, get_unique_connections, GradFactor
from .packbitstensor import PackBitsTensor


from pallas import compile as pallas_compile

@pallas_compile(backend="cuda")
def logic_layer_kernel(x, a, b, w, y):
    i = pallas_autograd.program_id(0)
    y[i] = x[i]


########################################################################################################################


class LogicLayer(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        """
        super().__init__()
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, 16, device=device))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor

        """
        The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only 
        available for device='cuda'. The `python` implementation exists for 2 reasons:
        1. To provide an easy-to-understand implementation of differentiable logic gate networks 
        2. To provide a CPU implementation of differentiable logic gate networks 
        """
        self.implementation = implementation
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'python'], self.implementation

        self.connections = connections
        assert self.connections in ['random', 'unique'], self.connections
        self.indices = self.get_connections(self.connections, device)

        self.num_neurons = out_dim
        self.num_weights = out_dim

    def forward(self, x, training: bool):
        if isinstance(x, PackBitsTensor):
            assert not training, 'PackBitsTensor is not supported for the differentiable training mode.'
            assert self.device == 'cuda', 'PackBitsTensor is only supported for CUDA, not for {}. '.format(self.device) + \
                                          'If you want fast inference on CPU, please use CompiledDiffLogicModel.'

        else:
            if self.grad_factor != 1.:
                x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == 'cuda':
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x, training)
        elif self.implementation == 'python':
            return self.forward_python(x, training)
        else:
            raise ValueError(self.implementation)

    def forward_python(self, x, training: bool):
        assert x.shape[-1] == self.in_dim, (x.shape[-1], self.in_dim)

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        weights = jnp.array(self.weights)  # Convert to JAX array
        if training:
            x = bin_op_s(a, b, softmax(weights, axis=-1))
        else:
            weights = one_hot(jnp.argmax(weights, axis=-1), 16).astype(jnp.float32)
            x = bin_op_s(a, b, weights)
        return x

    def forward_cuda(self, x, training: bool):
        x = jnp.array(x)
        a = jnp.array(self.indices[0])
        b = jnp.array(self.indices[1])
        w = jnp.array(self.weights)
        y = jnp.zeros((x.shape[0], self.out_dim), dtype=x.dtype)

        grid_dim = (x.shape[0],)
        block_dim = (min(x.shape[0], 1024),)

        logic_layer_kernel[grid_dim, block_dim](x[:,0], a, b, w, y)

        return y

    def forward_cuda_eval(self, x: PackBitsTensor):
        raise NotImplementedError("`forward_cuda_eval` is not yet implemented for the JAX version. "
                                  "PackBitsTensor is currently not supported in JAX.")

    def extra_repr(self):
        return '{}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval')

    def get_connections(self, connections, device='cuda'):
        assert self.out_dim * 2 >= self.in_dim, 'The number of neurons ({}) must not be smaller than half of the ' \
                                                'number of inputs ({}) because otherwise not all inputs could be ' \
                                                'used or considered.'.format(self.out_dim, self.in_dim)
        if connections == 'random':
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        elif connections == 'unique':
            return get_unique_connections(self.in_dim, self.out_dim, device)
        else:
            raise ValueError(connections)


########################################################################################################################


class GroupSum:
    """
    The GroupSum module.
    """
    def __init__(self, k: int, tau: float = 1., device='cuda'):
        self.k = k
        self.tau = tau
        self.device = device

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            raise NotImplementedError("PackBitsTensor is not yet supported in JAX.")

        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(axis=-1) / self.tau

    def __repr__(self):
        return f'GroupSum(k={self.k}, tau={self.tau})'

