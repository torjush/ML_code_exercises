import numpy as np

ACTION_SET = [-1, 1]
N_STATES = 6


def get_next_state(current_state, current_action):
    if current_state == 0 or current_state == 5:
        return current_state
    else:
        return current_state + current_action


def get_next_reward(current_state, next_state):
    if next_state == 5 and current_state != 5:
        return 5
    elif next_state == 0 and current_state != 0:
        return 1
    else:
        return 0


def update_Q(Q, current_state, current_action, gamma):
    max_Q = -9999999
    next_state = get_next_state(current_state, current_action)
    for action in ACTION_SET:
        if Q[action][next_state] > max_Q:
            max_Q = Q[action][next_state]

    reward = get_next_reward(current_state, next_state)

    Q[current_action][current_state] = reward + gamma * max_Q
    return Q


def Q_iteration(Q, gamma):
    for i in range(1000):
        for action in ACTION_SET:
            for state in range(N_STATES):
                Q = update_Q(Q, state, action, gamma)
    return Q


# Q function Q[a, s]
Q = dict(zip(ACTION_SET, [np.zeros(N_STATES), np.zeros(N_STATES)]))

Q = Q_iteration(Q, .5)
for action in ACTION_SET:
    print("action: {}|\t {}".format(action, Q[action]))
