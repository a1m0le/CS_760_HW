import numpy as np
import sys


from myLogi import myLogiClassifier



def run_cross_validation(fold_no, test_range, logi):
    logi.set_test_range(test_range)
    logi.train()
    accuracy, precision, recall = logi.predict()
    print("Fold No."+str(fold_no)+":")
    print("     Accuracy = "+str(accuracy))
    print("    Precision = "+str(precision))
    print("       Recall = "+str(recall))
    print()
    print()



if __name__=="__main__":

    if len(sys.argv) < 3:
        size = 0.1
        maxstep = 100
    else:
        size = float(sys.argv[1])
        maxstep = int(sys.argv[2])

    logi = myLogiClassifier("emails.csv", size, maxstep)
    run_cross_validation(1, (0,1000), logi)
    run_cross_validation(2, (999,2000), logi)
    run_cross_validation(3, (1999,3000), logi)
    run_cross_validation(4, (2999,4000), logi)
    run_cross_validation(5, (3999,5000), logi)

