import haiku as hk
import jax
import jax.numpy as jnp


class MazeFeedForwardTorso(hk.Module):
    def __init__(self, dense_layers, name=None, **unused_kwargs):
        super(MazeFeedForwardTorso, self).__init__(name=name)
        self._dense_layers = dense_layers

    def __call__(self, inputs):
        torso_net = [lambda x: x / 255., hk.Flatten()]
        for dim in self._dense_layers:
            torso_net.append(hk.Linear(dim))
            torso_net.append(jax.nn.relu)
        torso_out = hk.Sequential(torso_net)(inputs)
        return torso_out


class MazeShallowTorso(hk.Module):
    def __init__(self, conv_layers, dense_layers, padding='SAME', name=None, **unused_kwargs):
        super(MazeShallowTorso, self).__init__(name=name)
        self._conv_layers = conv_layers
        self._dense_layers = dense_layers
        self._padding = padding

    def __call__(self, inputs):
        torso_net = [lambda x: x / 255.]
        for ch, k in self._conv_layers:
            torso_net.append(hk.Conv2D(ch, kernel_shape=[k, k], stride=[1, 1], padding=self._padding))
            torso_net.append(jax.nn.relu)
        torso_net.append(hk.Flatten())
        for dim in self._dense_layers:
            torso_net.append(hk.Linear(dim))
            torso_net.append(jax.nn.relu)
        torso_out = hk.Sequential(torso_net)(inputs)
        return torso_out


class MazeDeepTorso(hk.Module):
    def __init__(self, residual_blocks, dense_layers, name=None, w_init=None, **unused_kwargs):
        super(MazeDeepTorso, self).__init__(name=name)
        self._residual_blocks = residual_blocks
        self._dense_layers = dense_layers
        self._w_init = w_init

    def __call__(self, x):
        torso_out = x / 255.
        for i, (num_channels, num_blocks) in enumerate(self._residual_blocks):
            conv = hk.Conv2D(
                num_channels, kernel_shape=[3, 3], stride=[1, 1], padding='SAME', w_init=self._w_init)
            torso_out = conv(torso_out)
            for j in range(num_blocks):
                block = ResidualBlock(num_channels, name='residual_{}_{}'.format(i, j), w_init=self._w_init)
                torso_out = block(torso_out)

        torso_out = jax.nn.relu(torso_out)
        torso_out = hk.Flatten()(torso_out)
        for dim in self._dense_layers:
            torso_out = hk.Linear(dim, w_init=self._w_init)(torso_out)
            torso_out = jax.nn.relu(torso_out)
        return torso_out


class AtariShallowTorso(hk.Module):
    """Shallow torso for Atari, from the DQN paper."""

    def __init__(self, dense_layers, name=None, w_init=None, **unused_kwargs):
        super(AtariShallowTorso, self).__init__(name=name)
        self._dense_layers = dense_layers
        self._w_init = w_init

    def __call__(self, x):
        torso_net = [
            lambda x: x / 255.,
            hk.Conv2D(32, kernel_shape=[8, 8], stride=[4, 4], padding='VALID', w_init=self._w_init),
            jax.nn.relu,
            hk.Conv2D(64, kernel_shape=[4, 4], stride=[2, 2], padding='VALID', w_init=self._w_init),
            jax.nn.relu,
            hk.Conv2D(64, kernel_shape=[3, 3], stride=[1, 1], padding='VALID', w_init=self._w_init),
            jax.nn.relu,
            hk.Flatten(),
        ]
        for dim in self._dense_layers:
            torso_net.append(hk.Linear(dim, w_init=self._w_init))
            torso_net.append(jax.nn.relu)
        return hk.Sequential(torso_net)(x)


class ResidualBlock(hk.Module):
    """Residual block."""

    def __init__(self, num_channels, name=None, w_init=None):
        super(ResidualBlock, self).__init__(name=name)
        self._num_channels = num_channels
        self._w_init = w_init

    def __call__(self, x):
        main_branch = hk.Sequential([
            jax.nn.relu,
            hk.Conv2D(
                self._num_channels,
                kernel_shape=[3, 3],
                stride=[1, 1],
                padding='SAME',
                w_init=self._w_init,
            ),
            jax.nn.relu,
            hk.Conv2D(
                self._num_channels,
                kernel_shape=[3, 3],
                stride=[1, 1],
                padding='SAME',
                w_init=self._w_init,
            ),
        ])
        return main_branch(x) + x


class AtariDeepTorso(hk.Module):
    """Deep torso for Atari, from the IMPALA paper."""

    def __init__(self, dense_layers, name=None, w_init=None, **unused_kwargs):
        super(AtariDeepTorso, self).__init__(name=name)
        self._dense_layers = dense_layers
        self._w_init = w_init

    def __call__(self, x):
        torso_out = x / 255.
        for i, (num_channels, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
            conv = hk.Conv2D(
                num_channels, kernel_shape=[3, 3], stride=[1, 1], padding='SAME', w_init=self._w_init)
            torso_out = conv(torso_out)
            torso_out = hk.max_pool(
                torso_out,
                window_shape=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
            )
            for j in range(num_blocks):
                block = ResidualBlock(num_channels, name='residual_{}_{}'.format(i, j), w_init=self._w_init)
                torso_out = block(torso_out)

        torso_out = jax.nn.relu(torso_out)
        torso_out = hk.Flatten()(torso_out)
        for dim in self._dense_layers:
            torso_out = hk.Linear(dim, w_init=self._w_init)(torso_out)
            torso_out = jax.nn.relu(torso_out)
        return torso_out


class ProcGenDeepTorso(hk.Module):
    def __init__(self, dense_layers, name=None, w_init=None, **unused_kwargs):
        super(ProcGenDeepTorso, self).__init__(name=name)
        self._dense_layers = dense_layers
        self._w_init = w_init

    def __call__(self, x):
        torso_out = x / 255.
        for i, (num_channels, num_blocks) in enumerate([(16, 1), (32, 1), (32, 1)]):
            conv = hk.Conv2D(
                num_channels, kernel_shape=[3, 3], stride=[1, 1], padding='SAME', w_init=self._w_init)
            torso_out = conv(torso_out)
            torso_out = hk.max_pool(
                torso_out,
                window_shape=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
            )
            for j in range(num_blocks):
                block = ResidualBlock(num_channels, name='residual_{}_{}'.format(i, j), w_init=self._w_init)
                torso_out = block(torso_out)

        torso_out = jax.nn.relu(torso_out)
        torso_out = hk.Flatten()(torso_out)
        for dim in self._dense_layers:
            torso_out = hk.Linear(dim, w_init=self._w_init)(torso_out)
            torso_out = jax.nn.relu(torso_out)
        return torso_out


class ProcGenWideTorso(hk.Module):
    def __init__(self, dense_layers, name=None, w_init=None, **unused_kwargs):
        super(ProcGenWideTorso, self).__init__(name=name)
        self._dense_layers = dense_layers
        self._w_init = w_init

    def __call__(self, x):
        torso_out = x / 255.
        for i, (num_channels, num_blocks) in enumerate([(32, 1), (64, 1), (64, 1)]):
            conv = hk.Conv2D(
                num_channels, kernel_shape=[3, 3], stride=[1, 1], padding='SAME', w_init=self._w_init)
            torso_out = conv(torso_out)
            torso_out = hk.max_pool(
                torso_out,
                window_shape=[1, 3, 3, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
            )
            for j in range(num_blocks):
                block = ResidualBlock(num_channels, name='residual_{}_{}'.format(i, j), w_init=self._w_init)
                torso_out = block(torso_out)

        torso_out = jax.nn.relu(torso_out)
        torso_out = hk.Flatten()(torso_out)
        for dim in self._dense_layers:
            torso_out = hk.Linear(dim, w_init=self._w_init)(torso_out)
            torso_out = jax.nn.relu(torso_out)
        return torso_out


def torso_network(torso_type, **torso_kwargs):
    if torso_type == 'maze_shallow' or torso_type == 'maze_cnn':
        return MazeShallowTorso(**torso_kwargs)
    elif torso_type == 'maze_ff':
        return MazeFeedForwardTorso(**torso_kwargs)
    elif torso_type == 'maze_deep':
        return MazeDeepTorso(**torso_kwargs)
    elif torso_type == 'atari_shallow':
        return AtariShallowTorso(**torso_kwargs)
    elif torso_type == 'atari_deep':
        return AtariDeepTorso(**torso_kwargs)
    elif torso_type == 'procgen_deep':
        return ProcGenDeepTorso(**torso_kwargs)
    elif torso_type == 'procgen_wide':
        return ProcGenWideTorso(**torso_kwargs)
    else:
        raise KeyError
