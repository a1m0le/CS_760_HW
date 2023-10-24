import smooth
import os


def get_prior_probs(takelog=False):
    path = "languageID"
    lang_dict = {"e":0,"s":0,"j":0}
    all_files = os.listdir(path)
    for fname in all_files:
        if fname[2] == "." and int(fname[1]) >= 0 and int(fname[1]) <= 9:
            lang_dict[fname[0]] = lang_dict[fname[0]] + 1
    ret = smooth.addi_smooth(lang_dict, 0.5)
    if takelog:
        smooth.to_log(ret)
    return ret
    










if __name__=="__main__":
    res = get_prior_probs(False)
    for k in res:
        print("p^(y=" + k+") = "+str(res[k]))
