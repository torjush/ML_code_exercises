import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

# Set up matplotlib to get pretty text
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Declare some constants from assignment text
N = 1165
dim = 64
N_zeros = 554
N_ones = N - N_zeros

# Import image data from file
data = np.ndarray(shape=(N, dim))

with open('optdigitsubset.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        data[i] = [float(val) for val in line.split()]

# Divide into classes
data_zeros = data[:N_zeros - 1]
data_ones = data[N_zeros:]


# Define loss and its gradient
def loss(rep_zero, rep_one, zeros, ones, regularizer):
    total_loss = 0.0
    for x in zeros:
        total_loss += (1.0/len(zeros) * linalg.norm((x - rep_zero), ord=2))
    for x in ones:
        total_loss += (1.0/len(ones) * linalg.norm((x - rep_one), ord=2))

    total_loss += (regularizer * linalg.norm((rep_zero - rep_one), ord=1))

    return total_loss


def grad_loss(rep_zero, rep_one, zeros, ones, regularizer):
    dr_zero = np.zeros(dim)
    dr_one = np.zeros(dim)
    for x in zeros:
        dr_zero += 2.0/len(zeros) * (rep_zero - x)

    dr_zero += regularizer * np.sign(rep_zero - rep_one)

    for x in ones:
        dr_one += 2.0/len(ones) * (rep_one - x)

    dr_one -= regularizer * np.sign(rep_zero - rep_one)

    return (dr_zero, dr_one)


def gradient_descent(zeros, ones, init_lr, regularizer, stop_threshold):
    # Initialize representors randomly(normal distributed)
    rep_zero = np.random.randn(dim) * 128
    rep_one = np.random.randn(dim) * 128

    prev_loss = np.inf
    current_loss = 0

    i = 0

    # Start gradient descent
    while np.abs(prev_loss - current_loss) > stop_threshold:
        # Update gradient
        dr_zero, dr_one = grad_loss(
            rep_zero,
            rep_one,
            zeros,
            ones,
            regularizer
            )

        # Update representors
        rep_zero -= init_lr * dr_zero
        rep_one -= init_lr * dr_one

        # Update loss
        prev_loss = current_loss
        current_loss = loss(rep_zero, rep_one, zeros, ones, regularizer)

        if (i + 1) % 15 == 0:
            # Decrease learning rate regularly
            init_lr /= 2
            print("Iteration: {}\tLoss: {}".format(i, current_loss))

        i += 1

    return (rep_zero, rep_one)


def show_representor_images():
    # Task 4b
    representors = []
    regularizers = [0, 50, 100, 200, 210]

    learning_rate = .5
    threshold = .1
    for regularizer in regularizers:
        representors.append(
            gradient_descent(
                data_zeros, data_ones, learning_rate, regularizer, threshold
            )
        )

    # Show representors
    fig, axis = plt.subplots(ncols=2, nrows=len(representors), figsize=(15, 10))

    for i, ax in enumerate(axis):
        ax[0].imshow(representors[i][0].reshape(8, 8))
        ax[0].set_title('Zero representor, $\lambda = {}$'.format(regularizers[i]))
        ax[1].imshow(representors[i][1].reshape(8, 8))
        ax[1].set_title('One representor, $\lambda = {}$'.format(regularizers[i]))

    fig.tight_layout()
    plt.savefig('../../Reports/1/representors_all.eps', dpi=300)
    plt.show()


def regularization_curves():
    # Task 4c
    N_iterations = 100

    threshold = .01
    learning_rate = .5
    regularizers = [0, .1, 1, 10, 100, 1000]
    true_losses = np.ndarray(shape=(len(regularizers), N_iterations))
    apparent_losses = np.ndarray(shape=(len(regularizers), N_iterations))

    for i, regularizer in enumerate(regularizers):
        for j in range(N_iterations):
            # Get representors by optimizing over all data
            rep_zero_all, rep_one_all = gradient_descent(
                data_zeros,
                data_ones,
                learning_rate,
                regularizer,
                threshold
                )

            # Get representors by optimizing over one (random) point per class
            rep_zero_one, rep_one_one = gradient_descent(
                data_zeros[np.random.randint(0, N_zeros - 1)],
                data_ones[np.random.randint(0, N_ones - 1)],
                learning_rate,
                regularizer,
                threshold
            )

            # Estimate true loss
            true_losses[i][j] = loss(rep_zero_all, rep_one_all, data_zeros, data_ones, regularizer)

            # Estimate apparent loss
            apparent_losses[i][j] = loss(
                rep_zero_one,
                rep_one_one,
                data_zeros,
                data_ones,
                regularizer
            )

    avg_true_losses = np.mean(true_losses, axis=1)
    avg_apparent_losses = np.mean(apparent_losses, axis=1)

    true_color = 'Blue'
    apparent_color = 'Red'

    plt.xscale('log')

    plt.plot(regularizers, avg_true_losses, color=true_color)
    plt.plot(regularizers, avg_apparent_losses, color=apparent_color)
    plt.grid()
    plt.legend(['Estimated true loss', 'Estimated apparent loss'], loc='upper left')
    plt.xlabel('$\lambda$')
    plt.ylabel('Loss')

    plt.title('Estimates of True and Apparent loss - Averaged over {:d} runs'.format(N_iterations))

    plt.savefig('../../Reports/1/figures/regularization_curves.eps', dpi=300)
    plt.show()


if __name__ == '__main__':
    show_representor_images()
