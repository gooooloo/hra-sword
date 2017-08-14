import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, head_reward_list, obs_tp1):
        data = (obs_t, action, head_reward_list, obs_tp1)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, head_reward_lists, obses_tp1 =  [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, head_reward_list, obs_tp1 = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            head_reward_lists.append(np.array(head_reward_list, copy=False))
            obses_tp1.append(np.array(obs_tp1, copy=False))
        return np.array(obses_t), np.array(actions), np.array(head_reward_lists), np.array(obses_tp1)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

