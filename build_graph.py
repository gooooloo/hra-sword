import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
import models


class HraDqnGraph(object):
    def __init__(self,
                 ob_shape,
                 lstm_size,   # e.g. 256
                 num_actions,   # e.g. 18 actions
                 num_heads,  # e.g. 3
                 head_gamma,  # shape: [num_heads]
                 optimizer,  # e.g. Adam
                 batch_size_if_train,
                 scope="hra_dqn"
                 ):
        self.ob_shape = ob_shape
        self.lstm_size = lstm_size

        with tf.variable_scope(scope):
            self.ob = ob = tf.placeholder(tf.float32, [None] + list(ob_shape), name="ob")
            self.lstm_state_in = lstm_state_in = tf.placeholder(tf.float32, [None, 2, lstm_size], name="lstm_in")
            self.head_weights = head_weights = tf.placeholder(tf.float32, [None, num_heads], name="head_weight")

            # act part
            qs, qs_heads, lstm_state, var_list_q = self._q_func(
                ob=ob,
                lstm_size=lstm_size,
                lstm_state_in=lstm_state_in,
                head_weight=head_weights,
                num_actions=num_actions,
                scope="q",
                reuse=None
            )  # qs:(#B, #A),  qs_heads:(#B, #H, #A), lstm_state:(2, #B, ...)

            lstm_state = tf.transpose([lstm_state[0], lstm_state[1]], perm=[1, 0, 2])  # (#b, 2, ...)
            self.lstm_state = lstm_state[0]  # (2, ..)  I know the batch size is 0 now
            deterministic_actions = tf.argmax(qs, axis=1)  # (#B,)
            self.stochastic = tf.placeholder(tf.bool, (), name="stochastic")  # scalar
            self.eps = tf.placeholder(tf.float32, (), name="eps")  # scalar

            random_actions = tf.random_uniform(tf.stack([1]), minval=0, maxval=num_actions, dtype=tf.int64)  # (#B)
            chose_random = tf.random_uniform(tf.stack([1]), minval=0, maxval=1, dtype=tf.float32)  # (#B)
            stochastic_actions = tf.where(chose_random < self.eps, random_actions, deterministic_actions)  # (#B)

            act_of_q = tf.cond(self.stochastic, lambda: stochastic_actions, lambda: deterministic_actions)  # (#B)
            self.act_of_q = act_of_q[0]  # scalar

            # train part
            self.act = act = tf.placeholder(tf.int32, [None], name="action")  # (#B)
            self.rew = rew = tf.placeholder(tf.float32, [None, num_heads], name="reward")  # (#B, #H)
            self.ob2 = ob2 = tf.placeholder(tf.float32, [None] + list(ob_shape), name="ob2")  # (#B, ...)
            self.lstm_state_in2 = lstm_state_in2 = tf.placeholder(tf.float32, [None, 2, lstm_size], name="lstm_in2")
            self.head_weights2 = head_weights2 = tf.placeholder(tf.float32, [None, num_heads], name="head_weight2")

            act_one_hot = tf.one_hot(act, num_actions)  # (#B, #A)
            act_expanded = tf.stack([act_one_hot] * num_heads, axis=1)  # (#B, #H, #A)
            q = tf.reduce_sum(qs_heads * act_expanded, axis=2)  # (#B, #H)

            _, qs_heads_2, _, var_list_q2 = self._q_func(
                ob=ob2,
                lstm_size=lstm_size,
                lstm_state_in=lstm_state_in2,
                head_weight=head_weights2,
                num_actions=num_actions,
                scope="target_q",
                reuse=None
            )  # qs_heads_2: (#B, #H, #A)
            q2 = tf.reduce_max(qs_heads_2, 2)  # (#B, #H)

            gamma = tf.stack([head_gamma] * batch_size_if_train, axis=0)  # (#B, #H)
            q_target = rew + gamma * q2  # (#B, #H)

            td_error = q - tf.stop_gradient(q_target)  # (#B, #H)
            self.errors = tf.reduce_sum(tf.square(td_error))  # scalar
            self.train_op = optimizer.minimize(self.errors)

            # update_target_fn will be called periodically to copy Q network to target Q network
            update_target_expr = []
            for var, var_target in zip(sorted(var_list_q, key=lambda v: v.name),
                                       sorted(var_list_q2, key=lambda v: v.name)):
                update_target_expr.append(var_target.assign(var))
            self.update_target_q_expr = tf.group(*update_target_expr)

    def get_initial_lstm_state(self):
        lstm_c_init = np.zeros(self.lstm_state_size.c, np.float32)
        lstm_h_init = np.zeros(self.lstm_state_size.h, np.float32)
        return [lstm_c_init, lstm_h_init]

    def _q_func(self, ob, lstm_size, lstm_state_in, head_weight, num_actions, scope, reuse):

        new_ob = [ob[:, :12], ob[:, 12:14], ob[:, 14:16], ob[:, 16:18], ob[:, 18:20], ob[:, 20:22]]
        weight_ob = ob[:, 22:26]
        batch_size = tf.shape(self.ob)[:1]

        new_ob[0], lstm_state = self._lstm(
            x=new_ob[0],
            lstm_size=lstm_size,
            in_state=lstm_state_in,
            sequence_length=batch_size,
            scope=scope
        )

        qs = []  # (#H, #B, #A)
        h = [[4,4], [4,4], [4,4], [4,4], [4,4], [4,4]]  # (#H, ...)
        for i, ob_i in enumerate(new_ob):
            thescope = '{}_{}'.format(scope, i)
            head_q_func = models.mlp(hiddens=h[i])
            qs0 = head_q_func(ob_i, num_actions, scope=thescope, reuse=reuse)  # (#B, #A)
            qs.append(qs0)

        weight_net = models.mlp(hiddens=[4, 4])
        weight_logit = weight_net(weight_ob, len(new_ob), scope='{}_{}'.format(scope, "weight_net"), reuse=reuse)  # (#B, #H)
        weight_logit = weight_logit - tf.reduce_max(weight_logit, [1], keep_dims=True)  # (#B, #H)
        head_weight = tf.nn.softmax(weight_logit)  # (#B, #H)

        qs = tf.stack(qs, axis=1)  # (#B, #H, #A)
        q = self._arrgegate(head_weight=head_weight, num_action=num_actions, qs=qs)  # (#B, #A)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        return q, qs, lstm_state, var_list

    def _lstm(self, x, lstm_size, in_state, sequence_length, scope):
        with tf.variable_scope(scope):
            # introduce a "fake" time dimension of 1 for LSTM
            x = tf.expand_dims(x, [1])

            cell = rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
            self.lstm_state_size = cell.state_size

            # lstm_in_c = tf.placeholder(tf.float32, [1, cell.state_size.c])
            # lstm_in_h = tf.placeholder(tf.float32, [1, cell.state_size.h])
            # self.lstm_in_state = [lstm_in_c, lstm_in_h]

            in_c, in_h = in_state[:, 0, :], in_state[:, 1, :]
            initial_state = rnn.LSTMStateTuple(in_c, in_h)
            out_result, out_state = tf.nn.dynamic_rnn(
                cell, x, initial_state=initial_state,
                time_major=False)
            out_c, out_h = out_state
            out_state = [out_c[:1, :], out_h[:1, :]]
            out_result = tf.reshape(out_result, [-1, lstm_size])

            return out_result, out_state

    def _arrgegate(self, head_weight, num_action, qs):
        '''
        :param head_weight:  (1, #H)
        :param qs:  (1, #H, #A)
        :return:  (1, #A)
        '''

        weights = tf.stack([head_weight] * num_action, axis=2)  # (1, #H, #A)

        t = qs*weights  # (1, #H, #A)
        ret = tf.reduce_sum(t, axis=1)  # (1, #A)

        return ret

    def act_func(self, ob, stochastic, eps):
        sess = tf.get_default_session()
        ret = sess.run([self.act_of_q, self.lstm_state],
                       {
                           self.ob: [ob[0]],
                           self.lstm_state_in: [ob[1]],
                           self.head_weights: [ob[2]],
                           self.stochastic: stochastic,
                           self.eps: eps
                       })
        return ret[0], ret[1]

    def train(self, obs, acts, rews, obs2):
        ob0 = np.asarray([x for x in obs[:,0]])  # convert [array,array,..] to matrix
        ob1 = np.asarray([x for x in obs[:,1]])  # convert [array,array,..] to matrix
        ob2 = np.asarray([x for x in obs[:,2]])  # convert [array,array,..] to matrix

        ob20 = np.asarray([x for x in obs2[:,0]])  # convert [array,array,..] to matrix
        ob21 = np.asarray([x for x in obs2[:,1]])  # convert [array,array,..] to matrix
        ob22 = np.asarray([x for x in obs2[:,2]])  # convert [array,array,..] to matrix

        sess = tf.get_default_session()
        ret = sess.run([self.train_op, self.errors],
                       {
                           self.ob: ob0,
                           self.lstm_state_in: ob1,
                           self.head_weights: ob2,
                           self.act: acts,
                           self.rew: rews,
                           self.ob2: ob20,
                           self.lstm_state_in2: ob21,
                           self.head_weights2: ob22
                       })
        return ret[1]

    def update_target(self):
        sess = tf.get_default_session()
        sess.run([self.update_target_q_expr])
