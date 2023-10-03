import numpy as np
import sys

from kNN import myKNN
from myLogi import myLogiClassifier

import matplotlib.pyplot as plt


def calculate_ROC(confi_data):
    total_pos = 0
    total_neg = 0
    for d in confi_data:
        if d[1] == 1:
            total_pos = total_pos + 1
        elif d[1] == 0:
            total_neg = total_neg + 1
        else:
            raise Exception("Third label?????")
    TP = 0
    FP = 0
    last_TP = 0
    TPRs = [0]
    FPRs = [0]
    for i in range(0, len(confi_data)):
        confi = confi_data[i]
        if i > 0:
            prev_confi = confi_data[i-1]
            if confi[0] != prev_confi[0] and confi[1] == 0 and TP > last_TP:
                TPR = TP / total_pos
                FPR = FP / total_neg
                TPRs.append(TPR)
                FPRs.append(FPR)
                last_TP = TP
        if confi[1] == 1:
            TP = TP + 1
        elif confi[1] == 0:
            FP = FP + 1
        else:
            raise Exception("Third label again????")
    TPR = TP / total_pos # isn't it just 1?
    FPR = FP / total_neg
    TPRs.append(TPR)
    FPRs.append(FPR)
    # now calculate the areas
    assert len(TPRs) == len(FPRs)
    AUC = 0
    for i in range(1, len(TPRs)):
        # (shangdi + xiadi) * gao / 2
        top = TPRs[i-1]
        bot = TPRs[i]
        height = FPRs[i] - FPRs[i-1]
        sub_area = ((top + bot) * height) / 2
        AUC = AUC + sub_area

    return FPRs, TPRs, AUC






if __name__=="__main__":

    step_size = 1
    maxstep = 500

    logi = myLogiClassifier("emails.csv", step_size, maxstep)
    logi.set_test_range((4000,5000))
    logi.train()
    logi_confi_data = logi.predict_ROC()
    print(logi.predict()[0])

    knn = myKNN("emails.csv")
    knn.config_k(5)
    knn_confi_data = knn.predict_ROC((4000,5000))

    # now we can try to draw the curve
    logi_x, logi_y, logi_auc = calculate_ROC(logi_confi_data)
    knn_x, knn_y, knn_auc = calculate_ROC(knn_confi_data)

    fig, axes = plt.subplots()

    fig.set_figheight(8)
    fig.set_figwidth(8)

    axes.set_title("ROC curve: k-NN v.s. Logistic Reg.")
    axes.set_xlabel("False Positive Rate")
    axes.set_ylabel("Avg. Accuracy")
    logi_line, = axes.plot(logi_x, logi_y, linewidth=0.5)
    logi_line.set_label("Logistic Reg. (AUC="+str(logi_auc))
    knn_line, = axes.plot(knn_x, knn_y, linewidth=0.5)
    knn_line.set_label("k-NN (AUC="+str(knn_auc))
    axes.legend()
    plt.savefig("Q5_figure.png")




