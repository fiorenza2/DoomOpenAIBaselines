from baselines.run import train
import os.path
import sys
from yaml import load
from gym.envs.registration import register


def register_vizdoom():
    register(
        id='VizDoomTrack2-v0',
        entry_point='envs.vizdoom_env:VizDoomGymTrack2'
    )


def main(yaml_name: str):
    yaml_path = os.path.join("src", "configs", yaml_name)
    config_dic = load(open(yaml_path))
    register_vizdoom()

    model, env = train(config_dic, {})

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
