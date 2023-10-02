import numpy as np

from kNN import myKNN



def run_cross_validation(fold_no, k, test_range, knn):
    knn.config_k(k)
    accuracy, precision, recall = knn.predict(test_range) 
    print("Fold No."+str(fold_no)+":")
    print("     Accuracy = "+str(accuracy))
    print("    Precision = "+str(precision))
    print("       Recall = "+str(recall))
    print()
    print()



if __name__=="__main__":

    knn = myKNN("emails.csv")
    run_cross_validation(1, 1, (0,1000), knn)
    run_cross_validation(2, 1, (999,2000), knn)
    run_cross_validation(3, 1, (1999,3000), knn)
    run_cross_validation(4, 1, (2999,4000), knn)
    run_cross_validation(5, 1, (3999,5000), knn)

