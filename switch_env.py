import collections
from collections.abc import Iterable
from typing import Union, Any, Callable

import jax
import jax.numpy as jnp
import jax.random as jrandom
import chex

State = collections.namedtuple(
    'State',
    (
        'observation',
        'reward',
        'terminal',
        'hidden',
    )
)


SwitchInfo = collections.namedtuple(
    'SwitchInfo',
    (
        'free_positive_switch',
        'free_negative_switch',
        'positive_switch',
        'negative_switch',
    )

)

KeyType = chex.Array
ActionType = int
RewardType = chex.Scalar
TerminalType = chex.Scalar

ResetFnType = Callable[[SwitchInfo], State]
StepFnType = Callable[[KeyType, State, ActionType], State]
LayerFnType = Callable[[KeyType, State, State, ActionType, int], State]


GRID_LAYOUT = jnp.array([
    [ # Agent
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [ # Walls
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ],
    [ # Switch
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [ # Indicator
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [ # Switch
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    [ # Indicator
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
])


def get_step_env(
    layer_fns:  list[LayerFnType],
) -> StepFnType:
    """Get a function for stepping a gridworld environment"""
    def step_env_fn(
        rngkey: KeyType,
        in_state: State,
        action: ActionType,
    ) -> State:
        new_state = in_state
        num_layers = len(layer_fns)
        keys = jrandom.split(rngkey, num_layers)
        for i, k, layer_fn in zip(range(num_layers), keys, layer_fns):
            new_state = layer_fn(k, in_state, new_state, action, i)
        return new_state
    return step_env_fn


def agent_fn(
    key: KeyType,
    in_state: State,
    next_state: State,
    action: ActionType,
    layer_index: int,
) -> State:
    grid = next_state.observation
    shift_action = jax.lax.select(action < 5, action, jnp.ones_like(action) * 5)
    a_to_move = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    layer = grid[layer_index]
    grid_shape = layer.shape
    loc = jnp.stack(jnp.where(layer == 1, size=1), axis=1)
    shift_fn = lambda l: jnp.clip(l + a_to_move[shift_action],
                       jnp.array((0, 0)), jnp.array(grid_shape) - 1)
    new_loc = jax.vmap(shift_fn)(loc)
    new_layer = layer.at[loc[:, 0], loc[:, 1]].set(0.0)
    new_layer = new_layer.at[new_loc[:, 0], new_loc[:, 1]].set(1.0)
    new_grid = grid.at[layer_index].set(new_layer)
    new_state = next_state._replace(observation=new_grid)
    return new_state


def get_wall_fn(
    agent_layer_index: int,
) -> LayerFnType:
    def wall_fn(
        key: KeyType,
        in_state: State,
        next_state: State,
        action: ActionType,
        layer_index: int,
    ) -> State:
        agent_before_move = in_state.observation[agent_layer_index]
        agent_after_move = next_state.observation[agent_layer_index]
        walls = in_state.observation[layer_index]
        wall_hit = jnp.any(agent_after_move * walls)
        new_agent = jax.lax.select(wall_hit, agent_before_move, agent_after_move)
        new_obs = next_state.observation.at[agent_layer_index].set(new_agent)
        new_state = next_state._replace(observation=new_obs)
        return new_state
    return wall_fn


def get_switch_fn(
    agent_layer_index: int,
) -> LayerFnType:
    def switch_fn(
        key: KeyType,
        in_state: State,
        next_state: State,
        action: ActionType,
        layer_index: int,
    ) -> State:
        next_obs = next_state.observation
        agent_loc = jnp.stack(jnp.where(next_obs[agent_layer_index] == 1, size=1), axis=1)
        switch_info = list(map(jnp.array, zip(*next_state.hidden)))
        def switch_resetter(o, si):
            i_loc = si[1]
            o = o.at[i_loc[0], i_loc[1], i_loc[2]].set(0.0)
            return o, o
        next_obs, _ = jax.lax.scan(switch_resetter, next_obs, switch_info)
        def switch_setter(prev_state, si):
            reward, terminal, o = prev_state
            s_loc, i_loc, r, t = si
            switch_hit = jnp.all(agent_loc == jnp.array(s_loc[1:]))
            no_hit_layer = o[i_loc[0]]
            hit_layer = no_hit_layer.at[i_loc[1], i_loc[2]].set(1.0)
            new_indicator_layer = jax.lax.select(switch_hit, hit_layer, no_hit_layer)
            o = o.at[i_loc[0]].set(new_indicator_layer)
            reward = reward + jax.lax.select(switch_hit, r, 0.0)
            terminal = terminal + jax.lax.select(switch_hit, t, 0.0)
            next_state = (reward, terminal, o)
            return next_state, 0
        (reward, terminal, next_obs), _ = jax.lax.scan(
            switch_setter,
            (0.0, 0.0, next_obs),
            switch_info,
        )
        next_state = next_state._replace(observation=next_obs, reward=reward,
                                         terminal=terminal)
        return next_state
    return switch_fn


def dummy_fn(
    key: KeyType,
    in_state: State,
    next_state: State,
    action: ActionType,
    layer_index: int,
) -> State:
    return next_state


def get_switch_env() -> tuple[ResetFnType, StepFnType]:
    def reset_fn(switch_info: SwitchInfo) -> State:
        return State(GRID_LAYOUT, 0.0, 0.0, switch_info)
    wall_fn = get_wall_fn(0)
    switch_fn = get_switch_fn(0)
    layer_fns = [
        agent_fn,
        wall_fn,
        switch_fn,
        dummy_fn,
        dummy_fn,
        dummy_fn,
    ]
    step_fn = get_step_env(layer_fns)
    return reset_fn, step_fn


def get_random_switch_info(
    rngkey: KeyType,
) -> SwitchInfo:
    free_switches = (
        (2, 1, 1),
        (2, 1, 3),
    )
    reward_switches = (
        (4, 1, 5),
        (4, 1, 7),
    )
    pos_i_loc = (3, 3, 3)
    neg_i_loc = (3, 3, 1)
    pos_index = int(jrandom.bernoulli(rngkey, 0.5))
    neg_index = 1 - pos_index
    switch_info = (
        (free_switches[pos_index], pos_i_loc, 0.0, 0.0),
        (free_switches[neg_index], neg_i_loc, 0.0, 0.0),
        (reward_switches[pos_index], pos_i_loc, 1.0, 1.0),
        (reward_switches[neg_index], neg_i_loc, -1.0, 1.0),
    )
    return switch_info


if __name__ == '__main__':
    rngkey = jrandom.PRNGKey(0)
    reset_env, step_env = get_switch_env()
    switch_info = get_random_switch_info(rngkey)
    s = reset_env(switch_info)
    print(s)
    action = 0
    while True:
        action = input('enter action: ')
        try:
            action = int(action)
        except:
            break
        s = step_env(rngkey, s, action)
        print(s)
