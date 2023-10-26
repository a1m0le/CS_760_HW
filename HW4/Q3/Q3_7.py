import smooth
import Q3_1
import Q3_2
import Q3_3
import Q3_4
import Q3_5
import langutil
import os
import math
import Q3_6


def get_probability_of_x(aslog=False):
    path = "languageID"
    all_files = os.listdir(path)
    ret = 1 / len(all_files)
    if aslog:
        return math.log(ret)
    return ret

def compute_log_posterior(filename):
    log_e = Q3_2.get_eng_condprob(takelog=True)
    log_s = Q3_3.get_span_condprob(takelog=True)
    log_j = Q3_3.get_nihon_condprob(takelog=True)
    esj = [log_e, log_s, log_j]
    likelihoods = Q3_5.compute_likelihood(filename, esj, retaslog=True)
    res = Q3_1.get_prior_probs(takelog=True)
    prior = []
    prior.append(res["e"])
    prior.append(res["s"])
    prior.append(res["j"])
    posterior_esj = [0,0,0]
    evidence = get_probability_of_x(aslog=True)
    for i in range(0, len(posterior_esj)):
        posterior = likelihoods[i] + prior[i] - evidence
        posterior_esj[i] = posterior
    return posterior_esj



def predict(posteriors, printout=False):
    index = None
    maxpos = None
    for i in range(0, len(posteriors)):
        if maxpos is None:
            index = i
            maxpos = posteriors[i]
        elif maxpos < posteriors[i]:
            index = i
            maxpos = posteriors[i]
    predlist = ["e","s","j"]
    printpredlist = ["e (English)","s (Spanish)","j (Japanese)"]
    if printout:
        print("Predicted to be: "+printpredlist[index])
    return predlist[index]


def gen_confusion_matrix(debugprint=False):
    # Structutre confumat[prediction][actual]
    confumat = {}
    confumat["e"] = {"e":0,"s":0,"j":0}
    confumat["s"] = {"e":0,"s":0,"j":0}
    confumat["j"] = {"e":0,"s":0,"j":0}
    # get the files to use
    path = "languageID"
    all_files = os.listdir(path)
    show_debug_mat = True
    for fname in all_files:
        if not fname[2] == ".":
            # it is a test file
            if debugprint:
                print("For "+fname, end="")
            posteriors = Q3_6.compute_log_posterior(fname)
            pred = Q3_6.predict(posteriors, printout=debugprint)
            actual = fname[0]
            confumat[pred][actual] += 1
            if debugprint and show_debug_mat:
                print(confumat)
    return confumat





if __name__=="__main__":
    mat = gen_confusion_matrix(debugprint=True)
    dictlabels = {"e":"  English  ","s":"  Spanish","j":" Japanese"}
    print("pred\\truth\tEnglish\tSpanish\tJapanese")
    key_list = predlist = ["e","s","j"]
    for i in range(0, len(key_list)):
        print(dictlabels[key_list[i]], end="")
        for j in range(0, len(key_list)):
            print("\t"+str(mat[key_list[i]][key_list[j]]), end="")
        print()




