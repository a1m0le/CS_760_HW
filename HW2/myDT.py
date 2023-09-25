import sys
import math


class DTNode:

    def __init__(self, isleaf, label=None, feature_dim=None, testval=None):
        self.isleaf = isleaf
        self.label = label
        self.feature_dim = feature_dim
        self.testval = testval
        self.left_node = None
        self.right_node = None

    def get_label(self):
        if self.isleaf:
            return self.label
        else:
            raise Exception("Attempting to get label from a non-leaf node")

    def do_test(features):
        if self.isleaf:
            raise Exception("Should not test on a leaf node")
        # get the value of the split happening at this node
        feature_to_test = features[self.feature_dim]
        if feature_to_test >= self.testval:
            return self.left_node
        else:
            return self.right_node
        

class CandiSplitZeroException(Exception):
     pass



def debug_print(level, message):
    for i in range(0, level*5):
    	print(" ",end="")
    print(message)
    pass


#given a list of instances, calculate its entropy
def entropy_calc(instance_list):
    # the instance list here would just be a list of (x_1, y_2, y)
    #calculating entropy
    # - sum over all y: P(y) * log_2(P(y))
    count  = 0
    for inst in instance_list:
        #taking the advantages that labels are just numbers 0 or 1
        count = count + inst[2]
    P_y1 = count / len(instance_list)
    P_y0 = 1 - P_y1 # better than recomputing the fraction
    if P_y1 == 0:
        P_y1_log = 0
    else:
        P_y1_log = math.log2(P_y1)
    if P_y0 == 0:
        P_y0_log = 0
    else:
        P_y0_log = math.log2(P_y0)
    H_y = -1 * ( (P_y1 * P_y1_log) + (P_y0 * P_y0_log)) 
    return H_y

def info_gain_ratio_calc(instances, split):
    # first get the split's entropy
    split_left = split[2] + 1
    # continue finding all values that can satisfy it
    while split_left < len(instances[split[0]]) and instances[split[0]][split_left][split[0]] == split[1]:
    	split_left = split_left + 1
    split_right = len(instances[split[0]]) - split_left
    print("Split:"+str(split),end=", ")
    

    P_left = split_left / len(instances[split[0]])
    P_right = split_right / len(instances[split[0]])
    if P_left == 0:
        P_left_log = 0
    else:
        P_left_log = math.log2(P_left)
    if P_right == 0:
        P_right_log = 0
    else:
        P_right_log = math.log2(P_right)
    H_split = -1 * ((P_left * P_left_log) + (P_right * P_right_log))
    # now the top part
    H_Y = entropy_calc(instances[split[0]])
    if split_left == 0:
        H_left = 0
    else:
        H_left = entropy_calc(instances[split[0]][0:split_left])
    if split_right == 0:
        H_right = 0
    else:
        H_right = entropy_calc(instances[split[0]][split_left:])
    H_Y_split = P_left * H_left + P_right * H_right
    info_gain = H_Y - H_Y_split
    if H_split == 0:
        print("Gain = "+str(info_gain))
        raise CandiSplitZeroException()
    gain_ratio = info_gain / H_split
    print("Gain Ratio = "+str(gain_ratio))
    return gain_ratio



# return a list of candidate splits
# each candidate split is of the format (feture_dimension, value, index)
def FindCandidateSplits(instances):
    # sort the first list of the instances using the first feature values as the key for each data point
    list0 = instances[0]
    list0.sort(key=lambda inst:inst[0], reverse=True)
    list0sets = []
    for i in range(0, len(list0)):
         if len(list0sets) == 0:
              list0sets.append([list0[i][0], i, {list0[i][2]}])
         else:
             if list0sets[-1][0] == list0[i][0]:
                 list0sets[-1][1] = i # update last seen address
                 list0sets[-1][2].add(list0[i][2]) # add the seen labels.
             else:
                 list0sets.append([list0[i][0], i, {list0[i][2]}])
    # do the same thing for the second list using the second feature values
    list1 = instances[1]
    list1.sort(key=lambda inst:inst[1], reverse=True)
    list1sets = []
    for i in range(0, len(list1)):
         if len(list1sets) == 0:
              list1sets.append([list1[i][1], i, {list1[i][2]}])
         else:
             if list1sets[-1][0] == list1[i][1]:
                 list1sets[-1][1] = i # update last seen address
                 list1sets[-1][2].add(list1[i][2]) # add the seen labels.
             else:
                 list1sets.append([list1[i][1], i, {list1[i][2]}])

    all_splits = []
    print(list0sets)
    # Find splits using the first feature
    for i in range(0, len(list0sets)-1):
        this_set = list0sets[i][2]
        next_set = list0sets[i+1][2]
        already_splited = False
        for l1 in this_set:
            if already_splited:
            	break
            for l2 in next_set:
                if l1 != l2:
            	    new_split = (0, list0sets[i][0], list0sets[i][1])
            	    all_splits.append(new_split)
            	    already_splited = True
            	    break

    for i in range(0, len(list1sets)-1):
        this_set = list1sets[i][2]
        next_set = list1sets[i+1][2]
        already_splited = False
        for l1 in this_set:
            if already_splited:
            	break
            for l2 in next_set:
                if l1 != l2:
            	    new_split = (1, list1sets[i][0], list1sets[i][1])
            	    all_splits.append(new_split)
            	    already_splited = True
            	    break
    # return all the splits
    return all_splits

# Process the splits. Return an indicator on stopping or not and a split with largest gain_ratio.
def process_splits(instances, all_splits):
    if len(instances[0]) == 0 or len(instances[1]) == 0:
        # Node is empty
        return True, None, "Empty"
    best_split = None
    largest_gain_ratio = -1
    for split in all_splits:
        try:
            gain_ratio = info_gain_ratio_calc(instances, split)
            if gain_ratio > largest_gain_ratio:
                largest_gain_ratio = gain_ratio
                best_split = split
        except CandiSplitZeroException:
            # zero split entropy
            pass
    if largest_gain_ratio > 0:
        return False, best_split, " = New split with gain ratio: "+str(largest_gain_ratio)
    if largest_gain_ratio == 0:
        return True, None, "Largest Gain Ratio is 0"
    if largest_gain_ratio < 0:
        return True, None, "All splits have entropy of zero"


def GenerateSubTree(instances, level):
    debug_print(level, "=========================")
    debug_print(level, "Processing: "+str(len(instances[0]))+" instances")
    candi_splits = FindCandidateSplits(instances)
    debug_print(level, "Found "+str(len(candi_splits))+" splits")
    should_stop, split, message = process_splits(instances, candi_splits)
    if should_stop:
        debug_print(level, "Stopping due to: "+message)
        if len(instances[0]) == 0:
            label = 1
        else:
            one_count = 0
            for inst in instances[0]:
                if inst[2] == 1:
                    one_count = one_count + 1
            zero_count = len(instances[0]) - one_count
            if one_count >= zero_count:
                label = 1
            else:
                label = 0
        new_leaf = DTNode(True, label=label)
        debug_print(level, "Labeled as: "+str(label))
        return new_leaf
    else:
        #left_half, right_half, feature_dim, split_val = FindBestSplits(isntances, candi_splits)
        debug_print(level, str(split)+message)
        split_left = split[2] + 1
    	# continue finding all values that can satisfy it
        while split_left < len(instances[split[0]]) and instances[split[0]][split_left][split[0]] == split[1]:
    	    split_left = split_left + 1
        left_list = instances[split[0]][0:split_left]
        right_list = instances[split[0]][split_left:]
        left_half = (left_list, left_list[:])
        right_half = (right_list, right_list[:])
        # find the children
        left_node = GenerateSubTree(left_half, level+1)
        right_node = GenerateSubTree(right_half, level+1)
        # construct the internal node
        new_internal = DTNode(False, feature_dim=split[0], testval=split[1])
        new_internal.left_node = left_node
        new_internal.right_node = right_node
        return new_internal



#structure of instances: 2 lists of tuples of (x_1, x_2, y)
#list[0] is to be sorted using x_1
#list[1] is to be sorted using x_2
# this is to help future usages
def LoadData(filename):
    all_instances = []
    list0 = []
    list1 = []
    with open(filename, "r") as input_file:
        all_lines = input_file.readlines()
        for line in all_lines:
            comps = line.split()
            x_1 = float(comps[0])
            x_2 = float(comps[1])
            y = int(comps[2])
            new_instance = (x_1, x_2, y)
            list0.append(new_instance)
            list1.append(new_instance)
    all_instances.append(list0)
    all_instances.append(list1)
    return all_instances



def print_tree_node(subroot, depth):
    if subroot is None:
    	raise Exception("no leaf?")
    for i in range(0, depth*7):
    	print(" ",end="")
    if subroot.isleaf:
    	print("LABEL="+str(subroot.get_label()))
    else:
        print("Braching left: X_"+str(subroot.feature_dim)+" >= "+str(subroot.testval))
        print_tree_node(subroot.left_node, depth+1)
        for i in range(0, depth*7):
            print(" ",end="")
        print("Braching right: X_"+str(subroot.feature_dim)+" < "+str(subroot.testval))
        print_tree_node(subroot.right_node, depth+1)

    


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect arg count")
    filename = sys.argv[1]
    all_instances = LoadData(filename) 
    tree_root = GenerateSubTree(all_instances, 0)
    print()
    print()
    print()
    print_tree_node(tree_root,0)
    
    
