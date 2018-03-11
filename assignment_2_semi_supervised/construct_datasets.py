import numpy as np
import matplotlib.pyplot as plt


def break_labelprop():
    """Constructs an artificial dataset that will break the 
    labelpropagation algorithm, 
    but will make the self-training a success"""
    x1 = np.random.randint(0, 100, 200)
    x2 = np.random.randint(0, 2, 200)

    x1_high = []
    x1_low = []

    for i, x in enumerate(x2):
        if x == 1:
            x1_high.append(x1[i])
        else:
            x1_low.append(x1[i])

    x2_low = np.zeros(len(x1_low))
    x2_high = np.ones(len(x1_high))

    fig, ax = plt.subplots()
    ax.scatter(x1_low, x2_low, c='blue', marker='.')
    ax.scatter(x1_high, x2_high, c='red', marker='.')
    ax.set_xlim([0, 100])
    ax.set_ylim([-35, 35])
    ax.legend(['Class 0', 'Class 1'])
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    plt.savefig('../../Reports/2/figures/break_labelprop.eps', dpi=300)
    plt.show()


def break_self_training():
    """Constructs an artificial dataset that will break the 
    labelpropagation algorithm, 
    but will make the self-training a success"""

def main():
    break_labelprop()


if __name__ == '__main__':
    main()
