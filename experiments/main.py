import argparse
import math
import random
import os

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from tqdm import tqdm

from results_json import ResultsJSON

from difflogic import LogicLayer, GroupSum


def load_dataset(args):
    # Placeholder for data loading - replace with actual data loading later
    if args.dataset == "mnist":
        num_classes = 10
        key = jax.random.PRNGKey(0)
        train_data = jax.random.normal(key, (50000, 784))
        train_labels = jax.random.randint(key, (50000,), 0, num_classes)
        test_data = jax.random.normal(key, (10000, 784))
        test_labels = jax.random.randint(key, (10000,), 0, num_classes)

        return (train_data, train_labels), (test_data, test_labels), None
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not yet supported for JAX.")



def input_dim_of_dataset(dataset):
    return {
        'mnist': 784,
    }[dataset]


def num_classes_of_dataset(dataset):
    return {
        'mnist': 10,
    }[dataset]




class Model(nn.Module):
    in_dim: int
    out_dim: int
    num_layers: int
    num_classes: int
    tau: float
    grad_factor: float
    connections: str

    @nn.compact
    def __call__(self, x, training: bool):
        llkw = dict(grad_factor=self.grad_factor, connections=self.connections, implementation="python")

        x = x.reshape((x.shape[0], -1)) # Flatten the input

        for _ in range(self.num_layers):
            x = LogicLayer(in_dim=self.in_dim if _ == 0 else self.out_dim, out_dim=self.out_dim, **llkw)(x, training=training)
        x = GroupSum(k=self.num_classes, tau=self.tau)(x)
        return x


def get_model(args):
    key = jax.random.PRNGKey(0)

    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)

    model = Model(in_dim=in_dim, out_dim=args.num_neurons, num_layers=args.num_layers, num_classes=class_count, tau=args.tau, grad_factor=args.grad_factor, connections=args.connections)

    params = model.init(key, jnp.ones((1, in_dim)), training=True)

    # Placeholder loss function
    def loss_fn(params, x, y):
        logits = model.apply(params, x, training=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    optimizer = optax.adam(learning_rate=args.learning_rate)

    return model, params, loss_fn, optimizer


@jax.jit
def train_step(params, opt_state, x, y, loss_fn):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def eval(params, model, x, y):
    logits = model.apply(params, x, training=False)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return accuracy


def packbits_eval(model, loader):
    raise NotImplementedError("PackBitsTensor is not yet supported in JAX.")


if __name__ == '__main__':

    ####################################################################################################################

    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')

    parser.add_argument('-eid', '--experiment_id', type=int, default=None)

    parser.add_argument('--dataset', type=str, choices=[
        'mnist',
    ], required=True, help='the dataset to use')
    parser.add_argument('--tau', '-t', type=float, default=10, help='the softmax temperature tau')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')


    parser.add_argument('--implementation', type=str, default='python', choices=['cuda', 'python'],
                        help='`cuda` is the fast CUDA implementation and `python` is simpler but much slower '
                        'implementation intended for helping with the understanding.')


    parser.add_argument('--num-iterations', '-ni', type=int, default=1000, help='Number of iterations (default: 1000)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=200, help='Evaluation frequency (default: 200)')



    parser.add_argument('--connections', type=str, default='unique', choices=['random', 'unique'])
    parser.add_argument('--architecture', '-a', type=str, default='randomly_connected')
    parser.add_argument('--num_neurons', '-k', type=int)
    parser.add_argument('--num_layers', '-l', type=int)

    parser.add_argument('--grad-factor', type=float, default=1.)

    args = parser.parse_args()

    ####################################################################################################################

    print(vars(args))

    assert args.num_iterations % args.eval_freq == 0, (
        f'iteration count ({args.num_iterations}) has to be divisible by evaluation frequency ({args.eval_freq})'
    )


    key = jax.random.PRNGKey(args.seed)
    (train_data, train_labels), (test_data, test_labels), _ = load_dataset(args)
    model, params, loss_fn, optimizer = get_model(args)

    opt_state = optimizer.init(params)


    best_acc = 0

    for i in tqdm(range(args.num_iterations), desc='iteration', total=args.num_iterations):
        x = train_data[i % len(train_data)].reshape((1, -1))
        y = train_labels[i % len(train_data)].reshape((1,))

        params, opt_state, loss = train_step(params, opt_state, x, y, loss_fn)


        if (i+1) % args.eval_freq == 0:

            train_accuracy = eval(params, model, train_data, train_labels)
            test_accuracy = eval(params, model, test_data, test_labels)

            r = {
                'train_acc': train_accuracy,
                'test_acc': test_accuracy,
            }

            print(r)

            if test_accuracy > best_acc:
                best_acc = test_accuracy
                print('IS THE BEST UNTIL NOW.')

import math
import random
import os

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from tqdm import tqdm

from results_json import ResultsJSON

import mnist_dataset
import uci_datasets
from difflogic import LogicLayer, GroupSum, PackBitsTensor, CompiledLogicNet


def load_dataset(args):
    # Placeholder for data loading - replace with actual data loading later
    if args.dataset == "mnist":
        num_classes = 10
        key = jax.random.PRNGKey(0)
        train_data = jax.random.normal(key, (50000, 784))
        train_labels = jax.random.randint(key, (50000,), 0, num_classes)
        test_data = jax.random.normal(key, (10000, 784))
        test_labels = jax.random.randint(key, (10000,), 0, num_classes)

        return (train_data, train_labels), (test_data, test_labels), None
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not yet supported for JAX.")





def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


def input_dim_of_dataset(dataset):
    return {
        'adult': 116,
        'breast_cancer': 51,
        'monk1': 17,
        'monk2': 17,
        'monk3': 17,
        'mnist': 784,
        'mnist20x20': 400,
        'cifar-10-3-thresholds': 3 * 32 * 32 * 3,
        'cifar-10-31-thresholds': 3 * 32 * 32 * 31,
    }[dataset]


def num_classes_of_dataset(dataset):
    return {
        'adult': 2,
        'breast_cancer': 2,
        'monk1': 2,
        'monk2': 2,
        'monk3': 2,
        'mnist': 10,
        'mnist20x20': 10,
        'cifar-10-3-thresholds': 10,
        'cifar-10-31-thresholds': 10,
    }[dataset]


import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from difflogic import LogicLayer, GroupSum

class Model(nn.Module):
    in_dim: int
    out_dim: int
    num_layers: int
    num_classes: int
    tau: float
    grad_factor: float
    connections: str

    @nn.compact
    def __call__(self, x, training: bool):
        llkw = dict(grad_factor=self.grad_factor, connections=self.connections, implementation="python")

        x = x.reshape((x.shape[0], -1)) # Flatten the input

        for _ in range(self.num_layers):
            x = LogicLayer(in_dim=self.in_dim if _ == 0 else self.out_dim, out_dim=self.out_dim, **llkw)(x, training=training)
        x = GroupSum(k=self.num_classes, tau=self.tau)(x)
        return x


def get_model(args):
    key = jax.random.PRNGKey(0)

    in_dim = input_dim_of_dataset(args.dataset)
    class_count = num_classes_of_dataset(args.dataset)

    model = Model(in_dim=in_dim, out_dim=args.num_neurons, num_layers=args.num_layers, num_classes=class_count, tau=args.tau, grad_factor=args.grad_factor, connections=args.connections)

    params = model.init(key, jnp.ones((1, in_dim)), training=True)

    # Placeholder loss function
    def loss_fn(params, x, y):
        logits = model.apply(params, x, training=True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    optimizer = optax.adam(learning_rate=args.learning_rate)

    return model, params, loss_fn, optimizer



def train(model, x, y, loss_fn, optimizer):
    x = model(x)
    loss = loss_fn(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def eval(model, loader, mode):
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        res = np.mean(
            [
                (model(x.to('cuda').round()).argmax(-1) == y.to('cuda')).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()


def packbits_eval(model, loader):
    orig_mode = model.training
    with torch.no_grad():
        model.eval()
        res = np.mean(
            [
                (model(PackBitsTensor(x.to('cuda').reshape(x.shape[0], -1).round().bool())).argmax(-1) == y.to(
                    'cuda')).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()


if __name__ == '__main__':

    ####################################################################################################################

    parser = argparse.ArgumentParser(description='Train logic gate network on the various datasets.')

    parser.add_argument('-eid', '--experiment_id', type=int, default=None)

    parser.add_argument('--dataset', type=str, choices=[
        'adult', 'breast_cancer',
        'monk1', 'monk2', 'monk3',
        'mnist', 'mnist20x20',
        'cifar-10-3-thresholds',
        'cifar-10-31-thresholds',
    ], required=True, help='the dataset to use')
    parser.add_argument('--tau', '-t', type=float, default=10, help='the softmax temperature tau')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')
    parser.add_argument('--batch-size', '-bs', type=int, default=128, help='batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--training-bit-count', '-c', type=int, default=32, help='training bit count (default: 32)')

    parser.add_argument('--implementation', type=str, default='cuda', choices=['cuda', 'python'],
                        help='`cuda` is the fast CUDA implementation and `python` is simpler but much slower '
                        'implementation intended for helping with the understanding.')

    parser.add_argument('--packbits_eval', action='store_true', help='Use the PackBitsTensor implementation for an '
                                                                     'additional eval step.')
    parser.add_argument('--compile_model', action='store_true', help='Compile the final model with C for CPU.')

    parser.add_argument('--num-iterations', '-ni', type=int, default=100_000, help='Number of iterations (default: 100_000)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=2_000, help='Evaluation frequency (default: 2_000)')

    parser.add_argument('--valid-set-size', '-vss', type=float, default=0., help='Fraction of the train set used for validation (default: 0.)')
    parser.add_argument('--extensive-eval', action='store_true', help='Additional evaluation (incl. valid set eval).')

    parser.add_argument('--connections', type=str, default='unique', choices=['random', 'unique'])
    parser.add_argument('--architecture', '-a', type=str, default='randomly_connected')
    parser.add_argument('--num_neurons', '-k', type=int)
    parser.add_argument('--num_layers', '-l', type=int)

    parser.add_argument('--grad-factor', type=float, default=1.)

    args = parser.parse_args()

    ####################################################################################################################

    print(vars(args))

    assert args.num_iterations % args.eval_freq == 0, (
        f'iteration count ({args.num_iterations}) has to be divisible by evaluation frequency ({args.eval_freq})'
    )

    if args.experiment_id is not None:
        assert 520_000 <= args.experiment_id < 530_000, args.experiment_id
        results = ResultsJSON(eid=args.experiment_id, path='./results/')
        results.store_args(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, validation_loader, test_loader = load_dataset(args)
    model, loss_fn, optim = get_model(args)

    ####################################################################################################################

    best_acc = 0

    for i, (x, y) in tqdm(
            enumerate(load_n(train_loader, args.num_iterations)),
            desc='iteration',
            total=args.num_iterations,
    ):
        x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to('cuda')
        y = y.to('cuda')

        loss = train(model, x, y, loss_fn, optim)

        if (i+1) % args.eval_freq == 0:
            if args.extensive_eval:
                train_accuracy_train_mode = eval(model, train_loader, mode=True)
                valid_accuracy_eval_mode = eval(model, validation_loader, mode=False)
                valid_accuracy_train_mode = eval(model, validation_loader, mode=True)
            else:
                train_accuracy_train_mode = -1
                valid_accuracy_eval_mode = -1
                valid_accuracy_train_mode = -1
            train_accuracy_eval_mode = eval(model, train_loader, mode=False)
            test_accuracy_eval_mode = eval(model, test_loader, mode=False)
            test_accuracy_train_mode = eval(model, test_loader, mode=True)

            r = {
                'train_acc_eval_mode': train_accuracy_eval_mode,
                'train_acc_train_mode': train_accuracy_train_mode,
                'valid_acc_eval_mode': valid_accuracy_eval_mode,
                'valid_acc_train_mode': valid_accuracy_train_mode,
                'test_acc_eval_mode': test_accuracy_eval_mode,
                'test_acc_train_mode': test_accuracy_train_mode,
            }

            if args.packbits_eval:
                r['train_acc_eval'] = packbits_eval(model, train_loader)
                r['valid_acc_eval'] = packbits_eval(model, train_loader)
                r['test_acc_eval'] = packbits_eval(model, test_loader)

            if args.experiment_id is not None:
                results.store_results(r)
            else:
                print(r)

            if valid_accuracy_eval_mode > best_acc:
                best_acc = valid_accuracy_eval_mode
                if args.experiment_id is not None:
                    results.store_final_results(r)
                else:
                    print('IS THE BEST UNTIL NOW.')

            if args.experiment_id is not None:
                results.save()

    ####################################################################################################################

    if args.compile_model:
        print('\n' + '='*80)
        print(' Converting the model to C code and compiling it...')
        print('='*80)

        for opt_level in range(4):

            for num_bits in [
                # 8,
                # 16,
                # 32,
                64
            ]:
                os.makedirs('lib', exist_ok=True)
                save_lib_path = 'lib/{:08d}_{}.so'.format(
                    args.experiment_id if args.experiment_id is not None else 0, num_bits
                )

                compiled_model = CompiledLogicNet(
                    model=model,
                    num_bits=num_bits,
                    cpu_compiler='gcc',
                    # cpu_compiler='clang',
                    verbose=True,
                )

                compiled_model.compile(
                    opt_level=1 if args.num_layers * args.num_neurons < 50_000 else 0,
                    save_lib_path=save_lib_path,
                    verbose=True
                )

                correct, total = 0, 0
                with torch.no_grad():
                    for (data, labels) in torch.utils.data.DataLoader(test_loader.dataset, batch_size=int(1e6), shuffle=False):
                        data = torch.nn.Flatten()(data).bool().numpy()

                        output = compiled_model(data, verbose=True)

                        correct += (output.argmax(-1) == labels).float().sum()
                        total += output.shape[0]

                acc3 = correct / total
                print('COMPILED MODEL', num_bits, acc3)

