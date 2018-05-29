import numpy as np
import sys

# Colored terminal output
RED = "\033[1;31m"
BLUE = "\033[1;34m"
CYAN = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD = "\033[;1m"
REVERSE = "\033[;7m"

ACTION_SET = [-1, 1]
N_STATES = 6


def print_Q_function(Q):
    for action in ACTION_SET:
        sys.stdout.write("Action: {}|\t".format(action))
        for state in Q[action]:
            sys.stdout.write("{:.3f}|\t".format(state))

        sys.stdout.write("\n")
    sys.stdout.write("\n")
    sys.stdout.flush()


def Q_equal_to_previous(Q, Q_prev):
    for action in ACTION_SET:
        if not (Q[action] == Q_prev[action]).all():
            return False
        else:
            return True


def get_next_state(current_state, current_action, broken_prob):
    if np.random.random() >= broken_prob:
        if current_state == 0 or current_state == 5:
            return current_state
        else:
            return current_state + current_action
    else:
        return current_state


def get_next_reward(current_state, next_state):
    if next_state == 5 and current_state != 5:
        return 5
    elif next_state == 0 and current_state != 0:
        return 1
    else:
        return 0


def update_Q(Q, current_state, current_action, gamma, broken_prob):
    next_state = get_next_state(current_state, current_action, broken_prob)
    max_Q = np.max([Q[action][next_state] for action in ACTION_SET])

    reward = get_next_reward(current_state, next_state)

    Q[current_action][current_state] = reward + gamma * max_Q
    return Q


def Q_iteration(Q, gamma, broken_prob):
    for i in range(N_STATES - 1):
        for action in ACTION_SET:
            for state in range(N_STATES):
                Q = update_Q(Q, state, action, gamma, broken_prob)
        print(f"Iteration: {i}")
        print_Q_function(Q)
    return Q


def get_optimal_policy(Q_star):
    pi_star = [0] * N_STATES

    for i in range(N_STATES):
        pi_star[i] = -1 if Q_star[-1][i] > Q_star[1][i] else 1

    return pi_star


def main():
    broken_prob = 0
    gamma = .5
    print("\n" + BOLD + "Exercise 2" + RESET)
    # Q function Q[a, s]
    Q = dict(zip(ACTION_SET, [np.zeros(N_STATES), np.zeros(N_STATES)]))

    Q = Q_iteration(Q, gamma, broken_prob)

    optimal_policy = get_optimal_policy(Q)
    print("Optimal policy: {}".format(optimal_policy))

    print("\n" + BOLD + "Exercise 3" + RESET)
    for gamma in [0, .1, .9, 1.]:
        print("Gamma: {}".format(gamma))
        Q = dict(zip(ACTION_SET, [np.zeros(N_STATES), np.zeros(N_STATES)]))

        Q = Q_iteration(Q, gamma, broken_prob)

        optimal_policy = get_optimal_policy(Q)
        print("Optimal policy: {}".format(optimal_policy))


if __name__ == '__main__':
    main()
