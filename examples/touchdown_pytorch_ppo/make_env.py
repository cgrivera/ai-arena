from Touchdown import TouchdownEnv


def make_env():
    return TouchdownEnv(1, blue_obs="image", blue_actions="continuous",
                        red_obs="image", red_actions="continuous")
