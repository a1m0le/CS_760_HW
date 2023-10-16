import smooth
import Q3_1
import Q3_2
import Q3_3
import Q3_4
import langutil


def compute_likelihood(fname, log_esj, retaslog=True):
    #first get the bag of word vector
    x = Q3_4.get_bagofwords_vec(fname)
    # now we compute each likelihood
    log_px_yesj = [0,0,0] # e, s, j
    for l in range(0, len(log_px_yesj)):
        for i in range(0, len(x)):
            # log of product of exponentials = addition of each multiplied log
            tmp = x[i] * log_esj[l][i]
            log_px_yesj[l] += tmp
    if not retaslog:
        langutil.vec_to_original(log_px_yesj)
        return log_px_yesj
    else:
        return log_px_yesj



if __name__=="__main__":
    log_e = Q3_2.get_eng_condprob(takelog=True)
    log_s = Q3_3.get_span_condprob(takelog=True)
    log_j = Q3_3.get_nihon_condprob(takelog=True)
    esj = [log_e, log_s, log_j]
    res = compute_likelihood("e10.txt", esj, retaslog=False)
    print(res)




