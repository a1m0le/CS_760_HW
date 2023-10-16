import math

def addi_smooth(count_dict, param):
    prob_dict = {}
    K = len(count_dict)
    count_sum = 0
    for key in count_dict:
        count_sum = count_sum + count_dict[key]
    # now each prob
    for key in count_dict:
        count = count_dict[key]
        prob = (count + param) / (count_sum + (K * param))
        prob_dict[key] = prob
    return prob_dict


def to_log(prob_dict):
    for k in prob_dict:
        prob_dict[k] = math.log(prob_dict[k])


def to_original(prob_dict):
    for k in prob_dict:
        prob_dict[k] = math.exp(prob_dict[k])
