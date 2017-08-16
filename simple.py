import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
import zipfile

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
import build_graph
from replay_buffer import ReplayBuffer


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path, num_cpu=16):
        with open(path, "rb") as f:
            model_data, act_params = dill.load(f)
        act, _, _ = build_graph.build_act(**act_params)
        sess = U.make_session(num_cpu=num_cpu)
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            U.load_state(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path):
        """Save model to a pickle located at `path`"""
        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            dill.dump((model_data, self._act_params), f)


def load(path, num_cpu=16):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle
    num_cpu: int
        number of cpus to use for executing the policy

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load(path, num_cpu=num_cpu)


def learn(env,
          aggregator,
          num_heads,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          save_model_freq=None,
          save_model_func=None,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          num_cpu=16,
          callback=None,
          render=False,
          summary_dir=None
          ):
    # Create all the functions necessary to train the model

    sess = U.make_session(num_cpu=num_cpu)
    sess.__enter__()

    if summary_dir is not None:
        summary_writer = tf.summary.FileWriter('{}/learn.summary'.format(summary_dir))
    else:
        summary_writer = None

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name)

    lstm = build_graph.build_lstm(
        input_shape=[3],  # FIXME : donot hard code it
        size=256
    )
    q_func = build_graph.build_q_func(
        lstm=lstm,
        num_actions=env.action_space.n
    )
    act, _, _ = build_graph.build_act(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        aggregator=aggregator,
        num_actions=env.action_space.n
    )
    train, update_target, debug_info = build_graph.build_train(
        batch_size=batch_size,
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_heads=num_heads,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10
    )
    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'aggregator': aggregator,
        'num_actions': env.action_space.n,
    }
    replay_buffer = ReplayBuffer(buffer_size)
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    obs = [obs, lstm.get_initial_features()]
    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join(td, "model")
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            exp = exploration.value(t)
            action, lstm_state = act(np.array(new_obs)[None], update_eps=exp)[0]
            new_obs, rew_list, done, _ = env.step(action)
            new_obs = [new_obs, lstm_state]
            rew = rew_list[-1]
            head_rew_list = np.array(rew_list[:-1])

            if render:
                env.render()

            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, head_rew_list, new_obs)
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                if summary_writer is not None:
                    summary = tf.Summary()
                    summary.value.add(tag='info/episode_reward', simple_value=float(episode_rewards[-1]))
                    summary.value.add(tag='info/exploration', simple_value=float(exp))
                    summary_writer.add_summary(summary, t)
                    summary_writer.flush()

                print('temp model saved in ', model_file)

                obs = env.reset()
                obs = [obs, lstm.get_initial_features()]
                episode_rewards.append(0.0)

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                obses_t, lstm_states, actions, rewards, obses_tp1 = replay_buffer.sample(batch_size)
                loss = train(obses_t, lstm_states, actions, rewards, obses_tp1)

                if summary_writer is not None:
                    summary = tf.Summary()
                    summary.value.add(tag='info/loss', simple_value=float(loss))
                    summary_writer.add_summary(summary, t)
                    summary_writer.flush()

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("last episode reward", round(episode_rewards[-2], 1))
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    U.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward

            if (save_model_freq is not None and save_model_func is not None and
                    t > learning_starts and t % save_model_freq == 0):
                save_model_func(t)

        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            U.load_state(model_file)

        if save_model_func is not None:
            save_model_func(max_timesteps)

    return ActWrapper(act, act_params)
