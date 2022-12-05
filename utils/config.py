"""config utilities for yml file."""
from pathlib import Path
import sys
import yaml

# singletone
FLAGS = None


class AttrDict(dict):
    """Dict as attribute trick."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, list):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    @staticmethod
    def get_cfg_dict(filename):

        assert Path(filename).exists(), f'File {filename} not exist.'
        try:
            with open(filename, 'r') as f:
                cfg_dict = yaml.safe_load(f)
        except EnvironmentError:
            print(f'Please check the file with name of "{filename}"')

        return cfg_dict

    @staticmethod
    def set_dict(org_dict, target_dict):

        assert isinstance(org_dict, dict)
        for key, value in target_dict.items():
            if key in org_dict:
                if isinstance(value, dict):
                    AttrDict.set_dict(org_dict[key], value)
                elif isinstance(value, list):
                    if isinstance(value[0], dict):
                        for i, v in enumerate(value):
                            AttrDict.set_dict(org_dict[key][i], v)
                    else:
                        org_dict[key] = value
                else:
                    org_dict[key] = value
            else:
                org_dict[key] = value

    def yaml(self):
        """Convert object to yaml dict and return."""
        yaml_dict = {}
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = new_l
                else:
                    yaml_dict[key] = value
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """Print all variables."""
        ret_str = []
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, AttrDict):
                ret_str.append(f'{key}:')
                child_ret_str = value.__repr__().split('\n')
                for item in child_ret_str:
                    ret_str.append('    ' + item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append(f'{key}:')
                    for item in value:
                        # treat as AttrDict above
                        child_ret_str = item.__repr__().split('\n')
                        for child_item in child_ret_str:
                            ret_str.append('    ' + child_item)
                else:
                    ret_str.append(f'{key}: {value}')
            else:
                ret_str.append(f'{key}: {value}')
        return '\n'.join(ret_str)


class Config(AttrDict):
    """Config with yaml file.

    This class is used to config model hyper-parameters, global constants, and
    other settings with yaml file. All settings in yaml file will be
    automatically logged into file.

    Args:
        default_setting_file(str): File name.

    Examples:

        yaml file ``model.yml``::

            NAME: 'neuralgym'
            ALPHA: 1.0
            DATASET: '/mnt/data/imagenet'

        Usage in .py:

        >>> config = Config('model.yml')
        >>> print(config.NAME)
            neuralgym
        >>> print(config.ALPHA)
            1.0
        >>> print(config.DATASET)
            /mnt/data/imagenet

    """

    def __init__(self, setting_file, verbose=False):

        default_cfg = AttrDict.get_cfg_dict('apps/default_setting.yml')
        super().__init__(default_cfg)
        AttrDict.set_dict(self.__dict__, AttrDict.get_cfg_dict(setting_file))
        if verbose:
            print(' pi.cfg '.center(80, '-'))
            print(self.__repr__())
            print(''.center(80, '-'))


def app():
    """Load app via stdin from subprocess"""
    global FLAGS
    if FLAGS is None:
        job_yaml_file = None
        for arg in sys.argv:
            if arg.startswith('app:'):
                job_yaml_file = arg[4:]
        if job_yaml_file is None:
            job_yaml_file = sys.stdin.readline()
        FLAGS = Config(job_yaml_file)


app()
