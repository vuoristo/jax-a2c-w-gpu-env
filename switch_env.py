from collections.abc import Iterable
from typing import Union, Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jrandom
import chex

KeyType = chex.Array
StateType = chex.Array
ActionType = int
RewardType = chex.Scalar
TerminalType = chex.Scalar

ResetFnType = Callable[[], StateType]
StepFnType = Callable[[KeyType, StateType, ActionType],
                      tuple[StateType, RewardType, TerminalType]]
LayerFnType = Callable[[KeyType, StateType, StateType, ActionType, int],
                        tuple[StateType, RewardType, TerminalType]]


def get_step_env(
    layer_fns:  list[LayerFnType],
) -> StepFnType:
    """Get a function for stepping a gridworld environment"""

    def step_env_fn(
        rngkey: KeyType,
        in_state: StateType,
        action: ActionType,
    ) -> tuple[StateType, RewardType, TerminalType]:
        reward = 0.0
        terminate = 0.0
        new_state = in_state
        num_layers = len(layer_fns)
        keys = jrandom.split(rngkey, num_layers)
        for i, k, layer_fn in zip(range(num_layers), keys, layer_fns):
            new_state, r, t = layer_fn(k, in_state, new_state, action, i)
            reward = reward + r
            terminate = terminate or t
        return new_state, reward, terminate

    return step_env_fn


def agent_fn(
    key: KeyType,
    in_state: StateType,
    next_state: StateType,
    action: ActionType,
    layer_index: int,
) -> tuple[StateType, RewardType, TerminalType]:
    shift_action = jax.lax.select(action < 5, action, jnp.ones_like(action) * 5)
    a_to_move = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    layer = in_state[layer_index]
    grid_shape = layer.shape
    loc = jnp.concatenate(jnp.where(layer == 1, size=1))
    new_loc = jnp.clip(loc + a_to_move[shift_action],
                        jnp.array((0, 0)), jnp.array(grid_shape))
    new_layer = layer.at[loc[0], loc[1]].set(0.0).at[
                        new_loc[0], new_loc[1]].set(1.0)
    new_state = in_state.at[layer_index].set(new_layer)
    return new_state, 0.0, 0.0


def wall_fn(
    key: KeyType,
    in_state: StateType,
    next_state: StateType,
    action: ActionType,
    layer_index: int,
) -> tuple[StateType, RewardType, TerminalType]:
    agent_after_move = next_state[0]
    walls = in_state[layer_index]
    wall_hit = jnp.any(agent_after_move * walls)
    new_state = jax.lax.select(wall_hit, in_state, next_state)
    return new_state, 0.0, 0.0



if __name__ == '__main__':
    step_fn = get_step_env([agent_fn, wall_fn])
    rngkey = jrandom.PRNGKey(0)
    state = jnp.zeros((3, 3, 4))
    state = state.at[0, 1, 1].set(1)
    state = state.at[1, 0].set(1)
    state = state.at[1, :, 0].set(1)
    print(state)
    print(step_fn(rngkey, state, 0)[0][0])
    print(step_fn(rngkey, state, 1)[0][0])
    print(step_fn(rngkey, state, 2)[0][0])
    print(step_fn(rngkey, state, 3)[0][0])
    print(step_fn(rngkey, state, 4)[0][0])
