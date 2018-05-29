import numpy as np
import sys
import matplotlib.pyplot as plt
import copy

font = {
    'family': 'helvetica',
    'size': 12
}
plt.rc('text', usetex=True)
plt.rc('font', **font)

ACTION_VAR = .01
RBF_VAR = .05
ACTION_SET = [-1, 1]
N_RBF = 2
RBF_CENTERS = np.array([1.5, 3.5])   # np.arange(0, 6, 6 / N_RBF)

epsilon = .5
gamma = .5
alpha = .3

N_EPISODES = 100000


def get_next_state(current_state, current_action):
    """Next state with random behaivour and continuous states"""
    if current_state <= 0.5 or current_state >= 4.5:
        return current_state
    else:
        rand_val = np.random.randn() * ACTION_VAR
        return current_state + current_action + rand_val


def get_next_reward(current_state, next_state):
    if next_state >= 4.5 and current_state < 4.5:
        return 5
    elif next_state <= 0.5 and current_state > 0.5:
        return 1
    else:
        return 0


def Q_approx(current_state, current_action, thetas):
    distances = np.abs(current_state - RBF_CENTERS)
    exponent = - (distances / (2 * RBF_VAR))
    rbf = np.exp(exponent)
    normalized = rbf / np.sum(rbf)
    return np.dot(thetas[current_action][:], normalized)


def Q_approx_grad(current_state):
    distances = np.abs(current_state - RBF_CENTERS)
    exponent = - (distances / (2 * RBF_VAR))
    return np.exp(exponent)


def episode(thetas, current_state, epsilon, gamma, alpha):
    actions_taken = []
    rewards = []
    states = []
    while current_state >= 0.5 and current_state <= 4.5:
        states.append(current_state)
        rand_val = np.random.random()
        if rand_val > epsilon:
            action_index = np.argmax(
                [Q_approx(current_state,
                          action,
                          thetas)
                 for action in ACTION_SET]
            )
        else:
            action_index = np.random.randint(len(ACTION_SET))

        action = ACTION_SET[action_index]
        actions_taken.append(action)
        next_state = get_next_state(current_state, action)
        reward = get_next_reward(current_state, next_state)
        rewards.append(reward)
        current_state = next_state

    states.append(current_state)

    for i, action in enumerate(actions_taken):
        max_Q = np.max([Q_approx(states[i + 1], action, thetas)
                        for action in ACTION_SET])
        grad = Q_approx_grad(states[i])
        temp_diff = gamma * max_Q - Q_approx(
            states[i],
            action,
            thetas)
        thetas[action] += alpha * (rewards[i] + temp_diff) * grad

    return thetas


def main():
    thetas = dict(zip(ACTION_SET, [np.zeros(N_RBF), np.zeros(N_RBF)]))
    theta_diffs = np.ndarray(shape=(2, N_EPISODES - 1))
    for i in range(N_EPISODES):
        init_state = np.random.randint(N_RBF)
        prev_thetas = copy.deepcopy(thetas)
        if i % 50 == 0:
            sys.stdout.write(f'Iteration: {i}\n')
            sys.stdout.write(f'Initial state: {init_state}\n')
            sys.stdout.flush()
        thetas = episode(thetas, init_state, epsilon, gamma, alpha)
        if i < N_EPISODES - 1:
            theta_diffs[0, i] = np.linalg.norm(thetas[-1] - prev_thetas[-1])
            theta_diffs[1, i] = np.linalg.norm(thetas[1] - prev_thetas[1])

    sys.stdout.write('Thetas:\n')
    for action in ACTION_SET:
        sys.stdout.write(f'Action: {action}|\t')
        for theta in thetas[action]:
            sys.stdout.write(f'{np.round(theta, decimals=4)}, ')
        sys.stdout.write('\b\b\n')

    plt.figure()
    states = np.linspace(0, 5, 200)
    for action in ACTION_SET:
        Qs = np.array([Q_approx(state, action, thetas) for state in states])
        plt.plot(states + 1, Qs)
    plt.legend(['Action: -1', 'Action:  1'])
    plt.xlabel('State')
    plt.ylabel(r'$Q(s)$')
    plt.title(f'Estimated Q-function with {N_RBF} RBF functions')
    plt.grid()
    plt.savefig(f'../../Reports/4/figures/Q_approx_{N_RBF}_rbfs.pdf', dpi=300)

    plt.figure()
    plt.plot(theta_diffs[0, :])
    plt.plot(theta_diffs[1, :])

    plt.legend(['Action: -1', 'Action:  1'])
    plt.title(r'Distance between consecutive weights $||\vec{\theta}_i - \vec{\theta}_{i-1}||_2$')
    plt.xlabel(r'$i$')

    plt.savefig(f'../../Reports/4/figures/theta_diffs_{N_RBF}_rbfs.pdf', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
