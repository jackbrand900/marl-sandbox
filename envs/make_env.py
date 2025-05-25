from pettingzoo.classic import rps_v2

def make_env():
    env = rps_v2.env()
    env.reset()
    return env
