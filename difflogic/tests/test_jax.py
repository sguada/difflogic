import jax
import jax.numpy as jnp
from jax.nn import softmax, one_hot
from difflogic import LogicLayer, GroupSum
from difflogic.functional import bin_op_s
import numpy as np

def test_logic_layer_forward():
    in_dim = 4
    out_dim = 2
    layer = LogicLayer(in_dim=in_dim, out_dim=out_dim, implementation="python", connections="unique")
    x = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    layer.weights = np.random.randn(out_dim, 16)
    output = layer(x, training=True)
    assert output.shape == (2, 2)

def test_group_sum_forward():
    k = 2
    group_sum = GroupSum(k=k)
    x = jnp.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    output = group_sum.forward(x)
    jnp.testing.assert_allclose(output, jnp.array([[4., 6.], [12., 14.]]))


test_logic_layer_forward()
test_group_sum_forward()
