import numpy as np
import matplotlib.pyplot as plt


from kNN import myKNN



def run_cross_validation(fold_no, k, test_range, knn):
    print("Running "+str(test_range))
    knn.config_k(k)
    accuracy, precision, recall = knn.predict(test_range) 
    return accuracy




def run_multiple_cv(k, knn):
    accu_sum = 0
    accu_sum = accu_sum + run_cross_validation(1, k, (0,1000), knn)
    accu_sum = accu_sum + run_cross_validation(2, k, (999,2000), knn)
    accu_sum = accu_sum + run_cross_validation(3, k, (1999,3000), knn)
    accu_sum = accu_sum + run_cross_validation(4, k, (2999,4000), knn)
    accu_sum = accu_sum + run_cross_validation(5, k, (3999,5000), knn)
    avg_accu = accu_sum / 5
    print("Avg Accuracy for k="+str(k)+" : "+str(avg_accu))
    return avg_accu


if __name__ == "__main__":
    knn = myKNN("emails.csv")
    ks = [1,3,5,7,10]
    avg_accus=[]
    for i in range(0, len(ks)):
        print("For k = "+str(ks[i]))
        avg = run_multiple_cv(ks[i], knn)
        avg_accus.append(avg)

    plt.title("5-Fold Cross Validation with kNN")
    plt.xlabel("different k values")
    plt.ylabel("Avg. Accuracy")
    plt.plot(ks, avg_accus, linestyle='--', marker='*')
    #plt.show()
    #plt.savefig("Q4_figure.png")

