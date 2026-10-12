
from envs.phlabenv import CitationEnv
try:
    from envs.lunarlander import LunarLanderWrapper
except:
    print( 'LunarLanderContinunous not available on this machine.' )


def select_env (environment_name : str):
    """ Instantiate the requested environment from its name.

    The PHLab environment name follows the pattern ``PHlab_<config>_<mode>``,
    e.g. ``PHlab_attitude_nominal`` or ``PHlab_attitude_h2000_v90`` where the
    mode itself may contain underscores. When the mode is omitted it defaults
    to ``nominal``.
    """
    _name = environment_name.lower()

    if _name.startswith('lunar'):
        wrapper = LunarLanderWrapper()
        return wrapper.env

    elif _name.startswith('phlab'):
        tokens = _name.split('_')
        if len(tokens) < 2:
            raise ValueError(
                f'Invalid PHLab environment name "{environment_name}". '
                'Expected "PHlab_<config>[_<mode>]", e.g. "PHlab_attitude_nominal".')

        phlab_config = tokens[1]
        phlab_mode = '_'.join(tokens[2:]) if len(tokens) > 2 else 'nominal'
        return CitationEnv(configuration=phlab_config, mode=phlab_mode)

    else:
        raise ValueError(f'{_name} is an unknown environment type')

if __name__ == '__main__':

    select_env('phlab_attitude')