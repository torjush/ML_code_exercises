import numpy as np
import matplotlib.pyplot as plt
import sys

font = {
    'family': 'helvetica',
    'size': 12
}
plt.rc('text', usetex=True)
plt.rc('font', **font)

ACTION_SET = [-1, 1]
N_STATES = 6

Q_true = np.array([
    [0.0, 1.0,   0.5,  0.625, 1.25, 0.0],
    [0.0, 0.625, 1.25, 2.5,   5.0,  0.0]
])


def Q_equal_to_previous(Q, Q_prev):
    for action in ACTION_SET:
        if not (Q[action] == Q_prev[action]).all():
            return False
        else:
            return True


def print_Q_function(Q):
    for action in ACTION_SET:
        sys.stdout.write("Action: {}|\t".format(action))
        for state in Q[action]:
            sys.stdout.write("{:.3f}|\t".format(state))

        sys.stdout.write("\n")
    sys.stdout.write("\n")
    sys.stdout.flush()


def get_next_state(current_state, current_action, broken_prob):
    if np.random.random() > broken_prob:
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


def episode(Q, current_state, epsilon, gamma, alpha, broken_prob):
    while current_state != 0 and current_state != 5:
        rand_val = np.random.random()
        if rand_val > epsilon:
            action_index = np.argmax([
                Q[-1][current_state],
                Q[1][current_state]])
        else:
            action_index = np.random.randint(0, len(ACTION_SET))

        current_action = ACTION_SET[action_index]
        next_state = get_next_state(current_state, current_action, broken_prob)

        reward = get_next_reward(current_state, next_state)
        max_Q = np.max([Q[action][next_state] for action in ACTION_SET])
        temp_diff = gamma * max_Q - Q[current_action][current_state]

        Q[current_action][current_state] += (alpha * (reward + temp_diff))

        current_state = next_state
        # alpha /= 1.005

    err = np.linalg.norm(
        Q_true - np.array([Q[action] for action in ACTION_SET]),
        ord=2)
    return Q, err


def Q_learning(Q, N_episodes, epsilon, gamma, alpha, broken_prob):
    errors = []
    for i in range(N_episodes):
        initial_state = np.random.randint(0, N_STATES)
        Q, err = episode(Q, initial_state, epsilon, gamma, alpha, broken_prob)
        errors.append(err)
    print_Q_function(Q)
    return Q, errors


def main():
    broken_prob = 0.3
    epsilons = np.arange(.01, 1., .05)
    gamma = .5
    N_episodes = 10000

    # Code to run for exercise 3
    if broken_prob == 0:
        alphas = np.arange(.1, 1., .2)
        errors = np.ndarray(shape=(len(alphas), len(epsilons)))
        for i, alpha in enumerate(alphas):
            for j, epsilon in enumerate(epsilons):
                print(f"alpha = {alpha:.2}, epsilon = {epsilon:.2}")
                # Q function Q[a, s]
                Q = dict(zip(ACTION_SET,
                             [np.zeros(N_STATES), np.zeros(N_STATES)]))
                Q, _ = Q_learning(Q, N_episodes,
                                  epsilon, gamma,
                                  alpha, broken_prob)

                error = np.linalg.norm(
                    Q_true - np.array([Q[action] for action in ACTION_SET]),
                    ord=2)
                errors[i, j] = error
                print(f"L2-distance: {error:.4}\n")

            plt.plot(epsilons, errors[i, :])
        legends = [r"$\alpha = {:.2f}$".format(alpha) for alpha in alphas]
        plt.legend(legends)
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r"$||Q_{true} - Q_{est}||_2$")
        plt.title("L2 distance between true Q function and its estimate")
        plt.grid()
        plt.savefig('../../Reports/4/figures/q_est_error.eps', dpi=300)
    else:
        alphas = [.01, .05, .1, .4, .6]
        epsilon = .3
        errors = np.ndarray(shape=(len(alphas), N_episodes))
        for i, alpha in enumerate(alphas):
            # Q function Q[a, s]
            Q = dict(zip(ACTION_SET,
                         [np.zeros(N_STATES), np.zeros(N_STATES)]))
            Q, err = Q_learning(Q, N_episodes,
                                epsilon, gamma,
                                alpha, broken_prob)
            errors[i, :] = err
            if i != 0:
                plt.plot(errors[i, :], alpha=.3)
            else:
                plt.plot(errors[0, :])
        legends = [r"$\alpha = {:.2f}$".format(alpha) for alpha in alphas]
        plt.legend(legends)
        plt.xlabel(r"Iteration \#")
        plt.ylabel(r"$||Q_{true} - Q_{est}||_2$")
        plt.title("L2 distance between true Q function and its estimate")
        plt.savefig('../../Reports/4/figures/q_est_error_broken.pdf', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
