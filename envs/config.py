
from envs.phlabenv import CitationEnv
try:
    from envs.lunarlander import LunarLanderWrapper
except:
    print( 'LunarLanderContinunous not available on this machine.' )


def select_env (environemnt_name : str):
    _name = environemnt_name

    if 'lunar'  in _name.lower():
        wrapper = LunarLanderWrapper()
        return wrapper.env
    
    elif 'ph' in _name.lower():
        tokens = _name.lower().split('_')
        phlab_mode = 'nominal'
        if len(tokens) == 3:
            _, phlab_config, phlab_mode = tokens
        else:
            phlab_config = tokens[-1]
            phlab_mode = ""

        return CitationEnv(configuration = phlab_config, mode = phlab_mode)

    else:
        raise ValueError(f'{_name} is an unknown environment type')

if __name__ == '__main__':

    select_env('phlab_attitude')