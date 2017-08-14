import tensorflow as tf
import baselines.common.tf_util as U


def build_act(make_obs_ph, q_func, aggregator, num_actions, scope="deepq"):
    with tf.variable_scope(scope):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))  # (...)
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")  # scalar
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")  # scalar

        eps = tf.get_variable("eps", [], initializer=tf.constant_initializer(0))  # scalar

        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")  # (#B, #H, #A)
        q_values_p1 = aggregator(q_values)  # (#B, #A)
        deterministic_actions = tf.argmax(q_values_p1, axis=1)  # (#B,)

        batch_size = 1
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)  # (#B)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps  # (#B)
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)  # (#B)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)  # (#B)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))  # (#B)

        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        qs = U.function(inputs=[observations_ph], outputs=q_values)
        qs_p1 = U.function(inputs=[observations_ph], outputs=q_values_p1)
        return act, qs, qs_p1


def build_train(batch_size, make_obs_ph, q_func, num_heads, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0, double_q=True, scope="deepq"):
    with tf.variable_scope(scope):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))  # (#B, ...)
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")  # (#B)
        rew_t_ph = tf.placeholder(tf.float32, [None, num_heads], name="reward")  # (#B, #H)
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))  # (#B, ...)

        # q network evaluation
        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # (#B, #H, #A)
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # target q network evaluation
        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")  # (#B, #H, #A)
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        # q scores for actions which we know were selected in the given state.
        act_t_one_hot = tf.one_hot(act_t_ph, num_actions)  # (#B, #A)
        act_t_expanded = tf.stack([act_t_one_hot] * num_heads, axis=1)  # (#B, #H, #A)
        q_t_selected = tf.reduce_sum(q_t * act_t_expanded, axis=2)  # (#B, #H)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)  # (#B, #H, #A)
            q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 2)  # (#B, #H)
            q_tp1_best_using_online_net_one_hot = tf.one_hot(q_tp1_best_using_online_net, num_actions)  # (#B, #H, #A)
            q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_using_online_net_one_hot, 2)  # (#B, #H)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 2)  # (#B, #H)

        # compute RHS of bellman equation
        new_gamma = tf.stack([gamma] * batch_size, axis=0)  # (#B, #H)
        print('====', new_gamma)
        q_t_selected_target = rew_t_ph + new_gamma * q_tp1_best  # (#B, #H)

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)  # (#B, #H)
        errors = tf.reduce_sum(td_error, [0, 1])  # scalar
        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                errors,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(errors, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input
            ],
            outputs=errors,
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return train, update_target, {'q_values': q_values}
