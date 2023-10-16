import math

def get_char_list():
    chrlist = []
    for i in range(0, 26):
        nextchar = chr(ord('a')+i)
        chrlist.append(nextchar)
    chrlist.append(" ")
    return chrlist


def vec_to_log(vec):
    for i in range(0, len(vec)):
        vec[i] = math.log(vec[i])

def vec_to_original(vec):
    for i in range(0, len(vec)):
        vec[i] = math.exp(vec[i])
                                    

if __name__=="__main__":
    print(get_char_list())
