
import types
import baselines.common.tf_util as U
from collections import defaultdict, deque
import gym
import simple, models
import tensorflow as tf
import sys
import signal
import subprocess
import time

import numpy as np
from easydict import EasyDict as edict
from gymgame.engine import Vector2
from gymgame.tinyrpg.sword import config, Serializer, EnvironmentGym
from gymgame.tinyrpg.framework import Skill, Damage, SingleEmitter
from gym import spaces

HRA_NUM_HEADS = 3  # 0: attack  1: defense  2: edge detect
HRA_NUM_ACTIONS = 9
HRA_WEIGHTS = [1.0, 2.0, 10.0]  # 0: attack  1: defense  2: edge detect
HRA_GAMMAS = [0.99, 0.95, 0.5]  # 0: attack  1: defense  2: edge detect
HRA_OB_INDEXES = [4, 6, 8]

OB_COUNT = 8
OB_LENGTH = OB_COUNT * HRA_OB_INDEXES[-1]
OB_SPACE_SHAPE = [OB_LENGTH]


GAME_NAME = config.GAME_NAME

config.BOKEH_MODE = "bokeh_serve"  # you need run `bokeh serve` firstly

config.MAP_SIZE = Vector2(30, 30)

config.GAME_PARAMS.fps = 24

config.GAME_PARAMS.max_steps = 300

config.NUM_PLAYERS = 1

config.NUM_NPC = 1

config.PLAYER_INIT_RADIUS = (0.0, 0.0)

config.NPC_INIT_RADIUS = (0.1, 0.9)

config.NPC_SKILL_COUNT = 1

config.SKILL_DICT = {
    'normal_attack' : Skill(
        id = 'normal_attack',
        cast_time = 0.0,#0.1,
        mp_cost = 0,
        target_required = True,
        target_relation = config.Relation.enemy,
        cast_distance = 1.0,
        target_factors = [Damage(200.0, config.Relation.enemy)]
    ),

    'normal_shoot' : Skill(
        id = 'normal_shoot',
        cast_time = 0.0, #0.3,
        mp_cost = 0,
        bullet_emitter = SingleEmitter(
            speed=0.3 * config.GAME_PARAMS.fps,
            penetration=1.0,
            max_range=config.MAP_SIZE.x * 0.8,
            radius=0.1,
            factors=[Damage(5.0, config.Relation.enemy)])
    ),

    'puncture_shoot' : Skill(
        id = 'normal_shoot',
        cast_time = 0.0,#0.3,
        mp_cost = 0,
        bullet_emitter = SingleEmitter(
            speed=0.3 * config.GAME_PARAMS.fps,
            penetration=np.Inf,
            max_range=config.MAP_SIZE.x * 0.8,
            radius=0.1,
            factors=[Damage(5.0, config.Relation.enemy)])
    ),
}

config.PLAYER_SKILL_LIST = [config.SKILL_DICT['puncture_shoot']]

config.NPC_SKILL_LIST = [config.SKILL_DICT['normal_attack']]

config.BASE_PLAYER = edict(
    id = "player-{0}",
    position = Vector2(0, 0),
    direct = Vector2(0, 0),
    speed = 0.3 * config.GAME_PARAMS.fps,
    radius = 0.5,
    max_hp = 100.0,
    camp = config.Camp[0],
    skills=config.PLAYER_SKILL_LIST
)

config.BASE_NPC = edict(
    id = "npc-{0}",
    position = Vector2(0, 0),
    direct = Vector2(0, 0),
    speed = 0.1 * config.GAME_PARAMS.fps,
    radius = 0.5,
    max_hp = 400.0,
    camp = config.Camp[1],
    skills=config.NPC_SKILL_LIST
)


def myextension(cls):

    def decorate_extension(ext_cls):
        dict = ext_cls.__dict__
        for k, v in dict.items():
            if type(v) is not types.MethodType and \
                            type(v) is not types.FunctionType and \
                            type(v) is not property:
                continue
            if hasattr(cls, k):
                setattr(cls, k+'_orig', getattr(cls, k))
            setattr(cls, k, v)
        return ext_cls

    return decorate_extension


@myextension(Serializer)
class SerializerExtension():

    DIRECTS = [Vector2.up,
               Vector2.up + Vector2.right,
               Vector2.right,
               Vector2.right + Vector2.down,
               Vector2.down,
               Vector2.down + Vector2.left,
               Vector2.left,
               Vector2.left + Vector2.up,
               ]

    def _deserialize_action(self, data):
        index, target = data
        if index < 8:
            direct = SerializerExtension.DIRECTS[index]
            actions = [('player-0', config.Action.move_toward, direct, None)]

        else:
            skill_index = index - 8
            skill_id = config.BASE_PLAYER.skills[skill_index].id
            actions = [('player-0', config.Action.cast_skill, skill_id, target, None)]

        return actions

    def _serialize_map(self, k, map):
        s_players = k.do_object(map.players, self._serialize_player)
        s_npcs = k.do_object(map.npcs, self._serialize_npc)
        s_bullets = []

        return np.hstack([s_players, s_npcs, s_bullets])

    def _serialize_character(self, k, char):

        # def norm_position_relative(v, norm):
        #     map = norm.game.map
        #     player = map.players[0]
        #     return (v - player.attribute.position) / map.bounds.max

        def norm_position_abs(v, norm):
            map = norm.game.map
            return v / map.bounds.max

        attr = char.attribute
        k.do(attr.position, None, norm_position_abs)
        k.do(attr.hp, None, k.n_div_tag, config.Attr.hp)


@myextension(EnvironmentGym)
class EnvExtension():
    def _init_action_space(self): return spaces.Discrete(9)

    def _my_current_ob(self):
        map = self.game.map
        player, npcs = map.players[0], map.npcs
        if len(npcs) == 0:
            delta = 0, 0
            npc_hp = 0
        else:
            delta = npcs[0].attribute.position - player.attribute.position  # [2]
            npc_hp = npcs[0].attribute.hp

        s = delta[0], delta[1], npc_hp, self._my_last_act, \
            delta[0], delta[1], \
            player.attribute.position[0], player.attribute.position[1]  # attack(4), defense(2), edge(2)

        assert len(s) == HRA_OB_INDEXES[-1]

        return s

    def _my_state(self):
        ret = []
        for x in self._my_all_obs: ret.extend(x[0:HRA_OB_INDEXES[0]])
        for x in self._my_all_obs: ret.extend(x[HRA_OB_INDEXES[0]:HRA_OB_INDEXES[1]])
        for x in self._my_all_obs: ret.extend(x[HRA_OB_INDEXES[1]:HRA_OB_INDEXES[2]])
        return ret

    def _my_get_hps(self):
        map = self.game.map
        player, npcs = map.players[0], map.npcs
        return player.attribute.hp / player.attribute.max_hp, sum([o.attribute.hp / o.attribute.max_hp for o in npcs])

    def _my_did_I_move(self):
        pos1 = self.last_pos
        pos2 = self._my_get_hps()
        d = pos1 - pos2
        return d[0] > 1e-5 and d[1] > 1e-5

    def _reset(self):
        self._reset_orig()

        self._my_last_act = -1
        self._my_all_obs = deque(maxlen=OB_COUNT)
        _cur_ob = self._my_current_ob()
        for _ in range(OB_COUNT): self._my_all_obs.append(_cur_ob)
        assert len(self._my_all_obs) == OB_COUNT

        return self._my_state()

    def _step(self, act):
        self.last_hps = self._my_get_hps()
        self.last_act = act
        self.last_pos = self.game.map.players[0].attribute.position

        _, r, t, i = self._step_orig((act, self.game.map.npcs[0]))

        self._my_last_act = act
        self._my_all_obs.append(self._my_current_ob())

        return self._my_state(), r, t, i

    def _reward(self):
        hps = self._my_get_hps()
        delta_hps = hps[0] - self.last_hps[0], hps[1] - self.last_hps[1]

        r_attack = -delta_hps[1]  # -1 -> 1
        r_defense = delta_hps[0]  # -1 -> -1
        r_edge = -1 if self.last_act < 8 and not self._my_did_I_move() else 0

        if hps[0] < 1e-5:
            r_game = -1
        elif hps[1] < 1e-5:
            r_game = 1
        else:
            r_game = 0

        x = [r_attack, r_defense, r_edge, r_game]
        return x


def _hra_q_func(ob, num_actions, scope, reuse=None):
    '''

    :param ob:  (#B, #H, ...)
    :param num_actions:  scalar
    :param scope:
    :param reuse:
    :return: (#B, #H, #A)
    '''

    old_shape = ob.shape
    assert len(old_shape) == 2
    assert old_shape[1] == OB_LENGTH

    new_ob = []
    new_ob.append(ob[:, 0:HRA_OB_INDEXES[0]*OB_COUNT])
    new_ob.append(ob[:, HRA_OB_INDEXES[0]*OB_COUNT:HRA_OB_INDEXES[1]*OB_COUNT])
    new_ob.append(ob[:, -2:])

    qs = []  # (#H, #B, #A)
    h = [[16,8,4], [4,4], [4,4]]  # (#H, ...)
    for i in range(HRA_NUM_HEADS):
        thescope = '{}_{}'.format(scope, i)
        head_q_func = models.mlp(hiddens=h[i])
        qs0 = head_q_func(new_ob[i], num_actions, scope=thescope, reuse=reuse)  # (#B, #A)
        qs.append(qs0)

    return tf.stack(qs, axis=1)  # (#B, #H, #A)


def do_demo(env, act, qs, qsp1):
    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            # time.sleep(0.1)
            env.render()
            action = act([obs])[0]

            # t = qs([obs])[0]
            # for i in range(3):
            #     print('head #{}:'.format(i), [round(x, 2) for x in t[i]])

            obs, rew_list, done, _ = env.step(action)

            rew = rew_list[-1]
            episode_rew += rew

        print("Episode reward", episode_rew)


def restore_checkpoint(sess, path):
    import build_graph

    def make_obs_ph(name):
        r = U.BatchInput(OB_SPACE_SHAPE, name=name)
        return r

    act, qs, qsp1 = build_graph.build_act(
        make_obs_ph=make_obs_ph,
        q_func=_hra_q_func,
        aggregator=models.arrgegator_weighted(HRA_WEIGHTS, HRA_NUM_ACTIONS),
        num_actions=HRA_NUM_ACTIONS
    )

    saver = tf.train.Saver()
    saver.restore(sess, path)

    return act, qs, qsp1

if __name__ == '__main__':

    render = '--visualise' in sys.argv[1:]
    demo = '--demo' in sys.argv[1:]
    pickle_file_path = None
    checkpoint_file_path = '/tmp/model'

    env = gym.make(GAME_NAME)
    env.observation_space = gym.spaces.Box(np.inf, np.inf, OB_SPACE_SHAPE)

    processAll = []
    def shutdown(signal, frame):
        print('Received signal %s: exiting', signal)
        for p in processAll:
            p.kill()
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if demo:

        bokehlog = open('/tmp/bokeh.log', 'w')
        # proc = subprocess.Popen("bokeh serve", shell=True, stderr=bokehlog, stdout=bokehlog)
        # processAll.append(proc)

        time_sleep = 3
        print('sleep {} sec for bokeh server'.format(time_sleep))
        time.sleep(time_sleep)

        with tf.Session() as sess:
            if pickle_file_path is not None:
                act = simple.load(pickle_file_path)
                qs = None
                qsp1 = None
            elif checkpoint_file_path is not None:
                act,qs, qsp1 = restore_checkpoint(sess, checkpoint_file_path)
            else:
                raise "bad input"

            do_demo(env, act, qs, qsp1)

    else:
        summary_dir = "./log/"
        port = 12346

        proc = subprocess.Popen("mkdir -p {0} && rm -rf {0} && mkdir -p {0}".format(summary_dir), shell=True)
        processAll.append(proc)
        # proc = subprocess.Popen("tensorboard --logdir {} --port {}".format(summary_dir, port), shell=True)
        # processAll.append(proc)
        # proc = subprocess.Popen("sleep 3 && open http://localhost:{}".format(port), shell=True)  # open via browser
        # processAll.append(proc)

        act = simple.learn(
            env,
            q_func=_hra_q_func,
            aggregator=models.arrgegator_weighted(HRA_WEIGHTS, HRA_NUM_ACTIONS),
            num_heads=HRA_NUM_HEADS,
            lr=1e-4,
            max_timesteps=2000000,
            buffer_size=10000,
            batch_size=32,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=10000,
            target_network_update_freq=1000,
            gamma=HRA_GAMMAS,
            render=render,
            summary_dir=None
        )

        # bug here
        # act.save(pickle_file_path)
