import functools
import collections
from collections.abc import Iterable
from typing import Dict, Tuple, Callable, Union, Any, Optional

import chex
import jax
import jax.random as jrandom
import jax.numpy as jnp
import haiku as hk
import rlax
import optax

import switch_env
from haiku_networks import torso_network
from utils import pack_namedtuple_jnp, add_batch

KeyType = chex.Array

AgentOutput = collections.namedtuple(
    'AgentOutput', (
        'rnn_state',
        'logits',
        'value',
    )
)

Trajectory = collections.namedtuple(
    'Trajectory', [
        'rnn_state',
        'action_tm1',
        'logits_tm1',
        'env_state',
    ]
)

LossLog = collections.namedtuple(
    'LossLog', [
        'pi_loss',
        'baseline_loss',
        'entropy',
        'advantage',
    ]
)


class ActorCriticNet(hk.RNNCore):
    def __init__(
        self,
        num_actions: int,
        torso_type: str,
        torso_kwargs: Dict,
        use_rnn: bool,
        head_layers: Iterable,
        name: Optional[str]=None,
    ):
        super(ActorCriticNet, self).__init__(name=name)
        self._num_actions = num_actions
        self._torso_type = torso_type
        self._torso_kwargs = torso_kwargs
        self._use_rnn = use_rnn
        if use_rnn:
            # TODO: this does not work with the current conda env.
            # core = hk.GRU(512, w_h_init=hk.initializers.Orthogonal())
            core = hk.LSTM(256)
            self._initial_state = core.initial_state
        else:
            core = hk.IdentityCore()
            self._initial_state = lambda batch_size: (jnp.zeros((batch_size, 1)))
        self._core = hk.ResetCore(core)
        self._head_layers = head_layers

    def __call__(self, timesteps):
        torso_net = torso_network(self._torso_type, **self._torso_kwargs)
        torso_output = torso_net(timesteps.env_state.observation)
        rnn_state = timesteps.rnn_state
        if self._use_rnn:
            core_input = jnp.concatenate([
                hk.one_hot(timesteps.action_tm1, self._num_actions),
                timesteps.env_state.reward[:, None],
                torso_output
            ], axis=1)
            should_reset = timesteps.first
            core_output, next_state = hk.dynamic_unroll(
                self._core, (core_input, should_reset), rnn_state)
        else:
            core_output, next_state = torso_output, rnn_state
        main_head = []
        for dim in self._head_layers:
            main_head.append(hk.Linear(dim))
            main_head.append(jax.nn.relu)
        h = hk.Sequential(main_head)(core_output)
        logits = hk.Linear(self._num_actions)(h)
        value = hk.Linear(1)(h)
        agent_output = AgentOutput(
            rnn_state=next_state,
            logits=logits,
            value=value.squeeze(-1),
        )
        return agent_output

    def initial_state(self, batch_size):
        return self._initial_state(batch_size)


def sample_action(
    rngkey: KeyType,
    theta: Dict,
    timestep: Trajectory,
    apply_theta_fn,
) -> tuple[AgentOutput, chex.Array]:
    agent_output = apply_theta_fn(theta, timestep)
    a = hk.multinomial(rngkey, agent_output.logits, num_samples=1).squeeze(axis=-1)
    return agent_output, a


def select_by_terminal(s0, s):
    inner_selector = lambda x0, x: jax.lax.select(s.terminal == 1.0
                                                  * jnp.ones_like(x0), x0, x)
    return jax.tree_map(inner_selector, s0, s)


def rollout(
    key: KeyType,
    theta: Dict,
    timestep: Trajectory,
    step_env_fn: switch_env.StepFnType,
    reset_env_fn: switch_env.ResetFnType,
    apply_theta_fn,
    H: int,
    num_envs: int,
) -> Trajectory:
    sample_action_ = functools.partial(sample_action, apply_theta_fn=apply_theta_fn)
    def scan_fn(prev_state, _):
        k1, timestep = prev_state
        k1, k2 = jrandom.split(k1)
        agent_output, a = sample_action_(k2, theta, timestep)
        k1, k2 = jrandom.split(k1)
        keys = jrandom.split(k1, num_envs)
        s = jax.vmap(step_env_fn)(keys, timestep.env_state, a)
        k1, k2 = jrandom.split(k1)
        keys = jrandom.split(k1, num_envs)
        s0 = jax.vmap(reset_env_fn)(keys, timestep.env_state.hidden)
        new_s = jax.vmap(select_by_terminal)(s0, s)
        ts = Trajectory(
            rnn_state=agent_output.rnn_state,
            action_tm1=a,
            logits_tm1=agent_output.logits,
            env_state=new_s,
        )
        return (k1, ts), ts
    traj_pre = add_batch(timestep, 1)
    _, traj_post = jax.lax.scan(scan_fn, (key, timestep), jnp.arange(H))
    traj = jax.tree_multimap(lambda *xs: jnp.moveaxis(jnp.concatenate(xs), 0, 1),
                             traj_pre, traj_post)
    return traj


def surr_loss(
    theta: Dict,
    trajs: Trajectory,
    ent_coef: chex.Scalar,
    gamma: chex.Scalar,
    vf_coef: chex.Scalar,
    apply_theta_fn: Any,
) -> Tuple[chex.Scalar, LossLog]:
    rewards = trajs.env_state.reward[:, 1:]
    discounts = (1.0 - trajs.env_state.terminal[:, 1:]) * gamma
    agent_output = jax.vmap(apply_theta_fn, (None, 0))(theta, trajs)
    value = agent_output.value
    logits = agent_output.logits
    v_t = value[:, 1:]
    v_tm1 = value[:, :-1]
    returns = jax.lax.stop_gradient(jax.vmap(rlax.lambda_returns)(
        rewards, discounts, v_t, jnp.broadcast_to(1.0, rewards.shape)))
    advantage = returns - jax.lax.stop_gradient(v_tm1)
    actions = trajs.action_tm1[:, 1:]
    logpi = jax.nn.log_softmax(logits[:, :-1])
    logpi_a = rlax.batched_index(logpi, actions)
    pi_loss = -jnp.mean(advantage * logpi_a)
    baseline_loss = 0.5 * jnp.mean(jnp.square(returns - v_tm1))
    pi = jax.nn.softmax(logits[:, :-1])
    entropy = jnp.sum(-pi * logpi, axis=-1)
    ent_loss = -jnp.mean(entropy)
    total_loss = pi_loss + vf_coef * baseline_loss + ent_coef * ent_loss
    log = LossLog(pi_loss, baseline_loss, -ent_loss, advantage)
    return total_loss, log


def sample_and_update(
    key: KeyType,
    theta: Dict,
    actor_output: Trajectory,
    opt_state,
    rollout_fn: Callable,
    grad_fn: Callable,
    opt_update_fn: Callable,
) -> tuple[Dict, LossLog]:
    trajs = rollout_fn(key, theta, actor_output)
    grad, logs = grad_fn(theta, trajs)
    updates, new_opt_state = opt_update_fn(grad, opt_state)
    new_theta = optax.apply_updates(theta, updates)
    return new_theta, new_opt_state, logs


def trainable(
    config: Dict,
    reporter: Any,
):
    num_parallel_envs = config['num_parallel_envs']
    k1 = jrandom.PRNGKey(config['seed'])
    init_theta, apply_theta = hk.without_apply_rng(
        hk.transform(lambda inputs: ActorCriticNet(
            **config['network_kwargs'])(inputs)))
    _, theta_initial_state_apply = hk.without_apply_rng(
        hk.transform(lambda batch_size: ActorCriticNet(
            **config['network_kwargs']).initial_state(batch_size)))
    reset_env, step_env = switch_env.get_switch_env()
    rollout_fn = functools.partial(
        rollout,
        step_env_fn=step_env,
        reset_env_fn=reset_env,
        apply_theta_fn=apply_theta,
        H=config['H'],
        num_envs=num_parallel_envs,
    )
    loss_fn = functools.partial(
        surr_loss,
        ent_coef=config['ent_coef'],
        gamma=config['gamma'],
        vf_coef=config['vf_coef'],
        apply_theta_fn=apply_theta,
    )
    grad_fn = jax.grad(loss_fn, has_aux=True)
    opt_kwargs = config['opt_kwargs'].copy()
    learning_rate = opt_kwargs.pop('learning_rate')
    opt_init_fn, opt_update_fn = optax.chain(
        optax.scale_by_adam(**opt_kwargs),
        optax.scale(-learning_rate),
    )
    sample_and_update_ = jax.jit(functools.partial(
        sample_and_update,
        rollout_fn=rollout_fn,
        grad_fn=grad_fn,
        opt_update_fn=opt_update_fn,
    ))
    k1, k2 = jrandom.split(k1)
    keys = jrandom.split(k1, num_parallel_envs)
    switch_infos = jax.vmap(switch_env.get_random_switch_info)(keys)
    k1, k2 = jrandom.split(k1)
    keys = jrandom.split(k1, num_parallel_envs)
    env_output = jax.vmap(reset_env)(keys, switch_infos)
    agent_state = theta_initial_state_apply(None, num_parallel_envs)
    actor_output = Trajectory(
        rnn_state=agent_state,
        action_tm1=jnp.zeros((num_parallel_envs,), dtype=jnp.int32),
        logits_tm1=jnp.zeros((num_parallel_envs,
                              config['network_kwargs']['num_actions'])),
        env_state=env_output,
    )
    k1, k2 = jrandom.split(k1)
    theta = init_theta(k2, actor_output)
    opt_state = opt_init_fn(theta)
    k1, k2 = jrandom.split(k1)
    theta, opt_state, log = sample_and_update_(k2, theta, actor_output, opt_state)



if __name__ == '__main__':
    config = {
        'network_kwargs': {
            'torso_type': 'maze_shallow',
            'torso_kwargs': {
                'conv_layers': ((32, 3), (64, 3), (64, 3)),
                'dense_layers': (129,),
            },
            'use_rnn': False,
            'head_layers': (),
            'num_actions': 4,
        },
        'num_parallel_envs': 7,
        'H': 10,

        'vf_coef': 0.5,
        'ent_coef': 0.01,
        'gamma': 0.99,

        'opt_kwargs': {
            'learning_rate': 7E-4,
            'b1': 0.9,
            'b2': 0.999,
            'eps': 1E-8,
        },

        'stop_steps': 1_000,
        'seed': 0,
    }
    trainable(config, None)
