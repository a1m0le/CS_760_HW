import matplotlib.pyplot as plt


SIGMA = [0.5, 1, 2, 4, 8]

OBJ = [903.241410422623, 1067.8111681341354, 1173.6980461436422, 1406.1074297872237, 1603.034132028563]

ACCU = [79.33333333333333, 69.0, 60.66666666666667, 40.666666666666664, 40.33333333333333]


def plotaccu():
    fig, axes = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    axes.set_ylim(0,100)
    axes.grid()
    axes.set_xlabel("Each sigma value as number")
    axes.set_ylabel("Accuracy(%)")
    axes.set_title("Accuracy of GMM clustering under different settings")
    axes.plot(SIGMA, ACCU, linestyle='-', marker='*')
    plt.savefig("plot/gmm_accu.png")


def plotobject():
    fig, axes = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    axes.set_ylim(0,4000)
    axes.grid()
    axes.set_xlabel("Each sigma value as number")
    axes.set_ylabel("Negative Log Likelihood")
    axes.set_title("Objective of GMM clustering under different settings")
    axes.plot(SIGMA, OBJ, linestyle='-', marker='*')
    plt.savefig("plot/gmm_object.png")


if __name__=="__main__":
    plotaccu()
    plotobject()
