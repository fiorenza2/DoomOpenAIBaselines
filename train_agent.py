import os.path
import sys
import envs
from yaml import load
from baselines.run import get_learn_function, get_learn_function_defaults
from baselines.common.atari_wrappers import FrameStack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
import gym
import multiprocessing
from gym.envs.registration import register


def train(params_dict: dict):
    ncpu = multiprocessing.cpu_count()
    env_id = params_dict['env_params']['env']

    total_timesteps = float(params_dict['training_params']['num_timesteps'])

    learn = get_learn_function(params_dict['model_params']['alg'])
    alg_kwargs = get_learn_function_defaults(params_dict['model_params']['alg'], 'atari')
    alg_kwargs['network'] = params_dict['model_params']['network']

    env = make_vec_env(env_id, 'custom', ncpu, None)

    if 'frame_stack' in params_dict['env_params']:
        env = VecFrameStack(env, params_dict['env_params']['frame_stack'])

    model = learn(
        env=env,
        seed=None,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model


def main(yaml_name: str):

    yaml_path = os.path.join("src", "configs", yaml_name)
    config_dic = load(open(yaml_path))

    model, env = train(config_dic)

    if config_dic.save_path:
        save_path = os.path.expanduser(config_dic.save_path)
        model.save(save_path)

    if config_dic.play:
        # we'll build this later
        pass


if __name__ == "__main__":
    args = sys.argv
    assert len(args) == 2, \
        "make sure there is one argument, and that is the name of the yaml file"
    main(sys.argv[1])
