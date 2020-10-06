import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):

    running_avg = np.zeros(len(scores))
    for i in range(len(scores)):
        running_avg[i] = np.mean(scores[max(0, i-100):i+1])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def plot_thetas(steps, thetas):
    print("Inside plot thetas")
    sin_theta, theta_dot = map(list, zip(*thetas))
    plt.figure()
    plt.plot(steps, sin_theta)
    plt.title("Sinus of thetas vs Timestamps")
    plt.savefig("plots/sin_theta.png")
    plt.figure()
    plt.plot(steps, theta_dot)
    plt.title("Theta dot vs Timestamps")
    plt.savefig("plots/theta_dot.png")
