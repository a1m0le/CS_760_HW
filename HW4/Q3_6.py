import smooth
import Q3_1
import Q3_2
import Q3_3
import Q3_4
import Q3_5
import langutil
import os
import math

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



if __name__=="__main__":
    res = compute_log_posterior("e10.txt")
    print("e10's log posterior is: ( [E, S, J] )")
    print(res)
    predict(res, printout=True)




