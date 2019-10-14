import numpy as np


def GAE(states, actions, rewards, nexts, dones, value_sampler, T=128, y=0.99, lam=0.95, use_Q=False):

    r = np.asarray(rewards)
    d = np.asarray(dones)

    a = actions  # want this to remain a list

    all_states = np.asarray(states + [nexts[-1]])

    if use_Q:
        all_values = value_sampler(all_states, actions)
    else:
        all_values = value_sampler(all_states)
    V_s = all_values[:-1]
    V_ns = all_values[1:]

    val_target = []
    adv_target = []

    #find the target values and target advantages
    for i in range(len(states)):

        remaining = len(states)-i  # remaining experience, including this entry

        adv_returns = []
        coef = 1.0

        roll_length = min(T, remaining)
        for j in range(roll_length):

            delta_t = r[i+j] + y*V_ns[i+j]*d[i+j] - V_s[i+j]
            adv_returns.append(coef*delta_t)
            coef *= (y*lam)

            #if done, end calculation
            if d[i+j] < 0.1:  # robust check for done==0.0
                break

        adv_returns = np.asarray(adv_returns)
        adv = np.sum(adv_returns)
        adv_target.append(np.asarray(adv))
        val_target.append(np.asarray(adv + V_s[i]))

    #convert to np arrays to return
    val_target = np.squeeze(np.asarray(val_target))
    adv_target = np.squeeze(np.asarray(adv_target))

    return val_target, adv_target
