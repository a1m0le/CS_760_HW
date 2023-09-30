import matplotlib.pyplot as plt


X = [0, 0/4, 1/4, 2/4, 1]
Y = [0, 2/6, 4/6, 6/6, 1]

plt.plot(X, Y, marker='o')
plt.title('ROC curve for spam classification')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.grid()
plt.savefig("ROC.png")

