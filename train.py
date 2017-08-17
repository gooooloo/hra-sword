import subprocess
import sys
import envs
import numpy as np
import signal
import tensorflow as tf

from baselines import logger
from baselines.common.schedules import LinearSchedule
import build_graph
from replay_buffer import ReplayBuffer


def learn(env,
          render,
          summary_writer,
          lstm_size=32,
          print_freq=1,
          save_model_freq=10000,
          save_path=None,
          save_max_to_keep=10,
          restore_path=None,
          head_weights=envs.HRA_WEIGHTS,
          num_heads=envs.HRA_NUM_HEADS,
          gamma=envs.HRA_GAMMAS,
          lr=1e-4,
          max_timesteps=2000000,
          buffer_size=10000,
          batch_size=32,
          exploration_fraction=0.1,
          exploration_initial_eps=1.0,
          exploration_final_eps=0.01,
          train_freq=4,
          learning_starts=100,
          target_network_update_freq=1000,
          ):

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
                                 initial_p=exploration_initial_eps,
                                 final_p=exploration_final_eps)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if restore_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, restore_path)

        if save_path is not None:
            saver = tf.train.Saver(max_to_keep=save_max_to_keep)

            def save_model_func(tt):
                saver.save(sess=sess, save_path=save_path, global_step=tt)
        else:
            save_model_func = None

        graph.update_target()

        episode_rewards = [0.0]
        saved_mean_reward = None
        obs = env.reset(graph.get_initial_lstm_state())

        for t in range(max_timesteps):
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

            if (save_model_freq is not None and save_model_func is not None and
                        t > learning_starts and t % save_model_freq == 0):
                save_model_func(t)

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
        save_path='/tmp/hra.sword',
        restore_path='/tmp/hra.sword-50000',
        summary_writer=summary_writer
    )


if __name__ == '__main__':
    main()