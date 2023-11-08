import matplotlib.pyplot as plt


SIGMA = [0.5, 1, 2, 4, 8]

OBJ = [302.45851960611833, 529.8701254393789, 740.4898722711276, 1746.1388989360505, 3450.4186015356963]

ACCU = [78.66666666666666, 69.33333333333334, 63.66666666666667, 53.0, 53.333333333333336]



def plotaccu():
    fig, axes = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    axes.set_ylim(0,100)
    axes.grid()
    axes.set_xlabel("Each sigma value as number")
    axes.set_ylabel("Accuracy(%)")
    axes.set_title("Accuracy of K-means clustering under different settings")
    axes.plot(SIGMA, ACCU, linestyle='-', marker='*')
    plt.savefig("plot/kmeans_accu.png")


def plotobject():
    fig, axes = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(8)
    axes.set_ylim(0,4000)
    axes.grid()
    axes.set_xlabel("Each sigma value as number")
    axes.set_ylabel("Distance objective (J_K)")
    axes.set_title("Objective of K-means clustering under different settings")
    axes.plot(SIGMA, OBJ, linestyle='-', marker='*')
    plt.savefig("plot/kmeans_object.png")


if __name__=="__main__":
    plotaccu()
    plotobject()
