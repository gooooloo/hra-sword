import subprocess
import sys
import envs
import numpy as np
import os
import tempfile
import signal
import tensorflow as tf

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
import build_graph
from replay_buffer import ReplayBuffer


def learn(env,
          render,
          summary_writer,
          initial_p=1.0,
          lstm_size=32,
          print_freq=1,
          checkpoint_freq=10000,
          save_model_freq=None,
          save_model_func=None,
          num_cpu=16,
          callback=None,
          head_weights=envs.HRA_WEIGHTS,
          num_heads=envs.HRA_NUM_HEADS,
          gamma=envs.HRA_GAMMAS,
          lr=1e-4,
          max_timesteps=2000000,
          buffer_size=10000,
          batch_size=32,
          exploration_fraction=0.1,
          exploration_final_eps=0.01,
          train_freq=4,
          learning_starts=10000,
          target_network_update_freq=1000,
          ):

    sess = U.make_session(num_cpu=num_cpu)
    sess.__enter__()

    graph = build_graph.HraDqnGraph(
        ob_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        num_heads=num_heads,
        head_weight=head_weights,
        head_gamma=gamma,
        batch_size_if_train=batch_size,
        lstm_size=lstm_size,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr)
    )
    replay_buffer = ReplayBuffer(buffer_size)

    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=initial_p,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    graph.update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset(graph.get_initial_lstm_state())
    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join(td, "model")
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            exp = exploration.value(t)
            action, lstm_state = graph.act_func(obs, True, exp)
            new_obs, rew_list, done, _ = env.step(action, lstm_state)
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

                obs = env.reset(graph.get_initial_lstm_state())
                episode_rewards.append(0.0)

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                obses_t, actions, rewards, obses_tp1 = replay_buffer.sample(batch_size)
                loss = graph.train(obses_t, actions, rewards, obses_tp1)

                if summary_writer is not None:
                    summary = tf.Summary()
                    summary.value.add(tag='info/loss', simple_value=float(loss))
                    summary_writer.add_summary(summary, t)
                    summary_writer.flush()

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                graph.update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("npc hp", env.npc_hp())
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


def prepare_process(summary_dir):
    processAll = []
    def shutdown(signal, frame):
        print('Received signal %s: exiting', signal)
        for p in processAll:
            p.kill()
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    port = 12346

    proc = subprocess.Popen("mkdir -p {0} && rm -rf {0} && mkdir -p {0}".format(summary_dir), shell=True)
    processAll.append(proc)
    # proc = subprocess.Popen("tensorboard --logdir {} --port {}".format(summary_dir, port), shell=True)
    # processAll.append(proc)
    # proc = subprocess.Popen("sleep 3 && open http://localhost:{}".format(port), shell=True)  # open via browser
    # processAll.append(proc)


def main():

    summary_dir = '/tmp/log'
    summary_dir = None
    render = '--visualise' in sys.argv[1:]

    prepare_process(summary_dir=summary_dir)

    if summary_dir is not None:
        summary_writer = tf.summary.FileWriter('{}/learn.summary'.format(summary_dir))
    else:
        summary_writer = None

    learn(
        env=envs.make_env(),
        render=render,
        summary_writer=summary_writer
    )


if __name__ == '__main__':
    main()