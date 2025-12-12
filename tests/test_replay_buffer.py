import numpy as np
import pytest

from connect4_dqn.data.replay_buffer import ReplayBuffer


def test_replay_buffer_add_and_len():
    buf = ReplayBuffer(capacity=3)

    s = np.zeros(10)
    ns = np.ones(10)

    assert len(buf) == 0
    buf.add(s, 1, 0.5, ns, False)
    assert len(buf) == 1
    buf.add(s, 2, -1.0, ns, True)
    assert len(buf) == 2


def test_replay_buffer_respects_capacity():
    buf = ReplayBuffer(capacity=2)

    s0 = np.zeros(3)
    s1 = np.ones(3)
    s2 = np.full(3, 2)

    buf.add(s0, 0, 0.0, s0, False)
    buf.add(s1, 1, 1.0, s1, False)
    assert len(buf) == 2

    # adding a third element should evict the oldest
    buf.add(s2, 2, 2.0, s2, True)
    assert len(buf) == 2

    # sample should never return the evicted oldest (s0) if we sample all
    states, actions, rewards, next_states, dones = buf.sample(2)
    assert not np.any(np.all(states == s0, axis=1))


def test_replay_buffer_sample_shapes():
    buf = ReplayBuffer(capacity=10)

    for i in range(5):
        s = np.full(4, i, dtype=np.float32)
        ns = np.full(4, i + 1, dtype=np.float32)
        buf.add(s, i % 3, float(i), ns, i % 2 == 0)

    states, actions, rewards, next_states, dones = buf.sample(3)

    assert states.shape == (3, 4)
    assert next_states.shape == (3, 4)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert dones.shape == (3,)


def test_replay_buffer_sample_raises_when_too_small():
    buf = ReplayBuffer(capacity=10)
    buf.add(np.zeros(2), 0, 0.0, np.zeros(2), False)

    with pytest.raises(ValueError):
        # random.sample raises ValueError when k > len(population)
        buf.sample(2)