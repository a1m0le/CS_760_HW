import smooth
import os
import langutil

def get_bagofwords_vec(fname):
    path = "languageID"
    charlist = langutil.get_char_list()
    count_dict = {}
    for c in charlist:
        count_dict[c] = 0
    # now let us count
    # process the file
    with open(path+"/"+fname, "r") as f:
        content = f.read()
    for c in content:
        if c in count_dict:
            count_dict[c] += 1
    #print(count_dict)
    #return count_dict
    bag_vec = []
    for c in charlist:
        bag_vec.append(count_dict[c])
    return bag_vec
    










if __name__=="__main__":
    print("For e10.txt:")
    res = get_bagofwords_vec("e10.txt")
    print(res)
