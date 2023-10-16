import smooth
import os
import langutil

def get_eng_condprob(takelog=False):
    path = "languageID"
    all_files = os.listdir(path)
    charlist = langutil.get_char_list()
    count_dict = {}
    for c in charlist:
        count_dict[c] = 0
    # now let us count
    for fname in all_files:
        if fname[0] == "e" and fname[2] == "." and int(fname[1]) >= 0 and int(fname[1]) <= 9:
            # process the file
            with open(path+"/"+fname, "r") as f:
                content = f.read()
            for c in content:
                if c in count_dict:
                    count_dict[c] += 1
    #print(count_dict)
    ret = smooth.addi_smooth(count_dict, 0.5)
    if takelog:
        smooth.to_log(ret)
    retvec = []
    for c in charlist:
        retvec.append(ret[c])
    return retvec
    










if __name__=="__main__":
    print("For English")
    res = get_eng_condprob()
    print(res)
