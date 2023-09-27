import sys
import math
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import graphviz

ALPHA = 0.5


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

    def do_test(self, features):
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
    print("Splite:"+str(split),end=", ")
    

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
    # do the same thing for the second list using the second feature values
    list1 = instances[1]
    list1.sort(key=lambda inst:inst[1], reverse=True)

    all_splits = []
    # Find splits using the first feature
    for i in range(0, len(list0)-1):
        if list0[i][2] != list0[i+1][2]:
            # anything greater than or equal to i would be a split
            new_split = (0, list0[i][0], i)
            all_splits.append(new_split)
    for i in range(0, len(list1)-1):
        if list1[i][2] != list1[i+1][2]:
            new_split = (1, list1[i][1], i)
            all_splits.append(new_split)
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



# def print_tree_node(subroot, depth):
#     if subroot is None:
#     	raise Exception("no leaf?")
#     for i in range(0, depth*7):
#     	print(" ",end="")
#     if subroot.isleaf:
#     	print("LABEL="+str(subroot.get_label()))
#     else:
#         print("Braching left: X_"+str(subroot.feature_dim)+" >= "+str(subroot.testval))
#         print_tree_node(subroot.left_node, depth+1)
#         for i in range(0, depth*7):
#             print(" ",end="")
#         print("Braching right: X_"+str(subroot.feature_dim)+" < "+str(subroot.testval))
#         print_tree_node(subroot.right_node, depth+1)

# Use https://csacademy.com/app/graph_editor/
def print_tree_node(subroot, depth):
    if subroot is None:
        raise Exception("no leaf?")
    if subroot.isleaf:
        return str(subroot.label) + ":"+str(id(subroot))
    else:
        text = "x_"+str(subroot.feature_dim)+">="+str(subroot.testval)
        left_text = print_tree_node(subroot.left_node, depth+1)
        print(text+" "+left_text+" Yes")
        right_text = print_tree_node(subroot.right_node, depth+1)
        print(text+" "+right_text+" No")
        return text



def graph_tree(subroot):
    graph = graphviz.Digraph()
    graph_tree_recur(subroot, graph)
    return graph

def graph_tree_recur(subroot, graph):
    if subroot is None:
        raise Exception("no leaf?")
    if subroot.isleaf:
        graph.node(str(id(subroot)), str(subroot.label))
        return str(id(subroot))
    else:
        text = "x_"+str(subroot.feature_dim)+">="+str(subroot.testval)
        graph.node(str(id(subroot)), text)
        left_node_name = graph_tree_recur(subroot.left_node, graph)
        graph.edge(str(id(subroot)),left_node_name ,"Yes")

        right_node_name = graph_tree_recur(subroot.right_node, graph)
        graph.edge(str(id(subroot)),right_node_name ,"No")
        return str(id(subroot))

    
# range means the range of the corresponding feature value that will be classified to
def add_decision_rect(subroot, x1_range, x2_range, axes):
    if subroot is None:
        raise Exception("no leaf?")
    if subroot.isleaf:
        # it is a leaf. then we can draw the rectangle
        # the bases will be the left value of both ranges
        x = x1_range[0]
        y = x2_range[0]
        width = x1_range[1] - x1_range[0]
        height = x2_range[1] - x2_range[0]
        if subroot.get_label() == 0:
            color = "red"
        else:
            color = "green"
        rect = patches.Rectangle((x,y),width,height, color=color, alpha=ALPHA, linewidth=0)
        axes.add_patch(rect)
    else:
        # it is not a leaf. we are spliting.
        # firstly the left leaf
        if subroot.feature_dim == 0:
            # splitting x_1 range
            val = subroot.testval
            new_x1_range = [val, x1_range[1]] #TODO Is this logic correct?
            assert(val >= x1_range[0])
            add_decision_rect(subroot.left_node, new_x1_range, x2_range[:], axes)
        else:
            val = subroot.testval
            new_x2_range = [val, x2_range[1]] #TODO Is this logic correct?
            assert(val >= x2_range[0])
            add_decision_rect(subroot.left_node, x1_range[:], new_x2_range, axes)
        # now the right leaf
        if subroot.feature_dim == 0:
            # splitting x_1 range
            val = subroot.testval
            new_x1_range = [x1_range[0], val] #TODO Is this logic correct?
            assert(val <= x1_range[1])
            add_decision_rect(subroot.right_node, new_x1_range, x2_range[:], axes)
        else:
            val = subroot.testval
            new_x2_range = [x2_range[0], val] #TODO Is this logic correct?
            assert(val <= x2_range[1])
            add_decision_rect(subroot.right_node, x1_range[:], new_x2_range, axes)



def make_plot(dp_list, tree_root, filename):
    # firstly make a scatter plot
    fig, axes = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(16)

    # process points
    x_1_y1 = []
    x_2_y1 = []
    for inst in dp_list:
        if inst[2] == 1:
            x_1_y1.append(inst[0])
            x_2_y1.append(inst[1])

    x_1_y0 = []
    x_2_y0 = []
    for inst in dp_list:
        if inst[2] == 0:
            x_1_y0.append(inst[0])
            x_2_y0.append(inst[1])
    x1_max = max(max(x_1_y0),max(x_1_y1)) + 0.075
    x1_min = min(min(x_1_y0),min(x_1_y1)) - 0.075
    x2_max = max(max(x_2_y0),max(x_2_y1)) + 0.075
    x2_min = min(min(x_2_y0),min(x_2_y1)) - 0.075
    axes.set_ylim(x2_min,x2_max)
    axes.set_xlim(x1_min,x1_max)
    axes.scatter(x_1_y0, x_2_y0, color = "red")
    axes.scatter(x_1_y1, x_2_y1, color = "green")


    add_decision_rect(tree_root, [x1_min,x1_max],[x2_min,x2_max],axes)

    # # rect = patches.Rectangle((x1_min,x2_min),0.6,0.3,color="red", alpha=ALPHA, linewidth=0)
    # # rect2 = patches.Rectangle((0.6,0.6),0.2,0.2, color="black", alpha=ALPHA, linewidth=0)
    # axes.add_patch(rect)
    # axes.add_patch(rect2)
    #plt.show()
    plt.savefig("plot_"+filename+".png")





# return True if the decision meets expected label
def test_decision(feature, expected_label, root):
    node = root
    while not node.isleaf:
     next_ndoe = node.do_test(feature)
     node = next_ndoe
    return node.get_label() == expected_label 








if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Incorrect arg count")
    filename = sys.argv[1]
    all_instances = LoadData(filename) 
    tree_root = GenerateSubTree(all_instances, 0)
    print()
    print()
    print()
    #print_tree_node(tree_root,0)
    graph = graph_tree(tree_root)
    graph.render(filename+".pdf", view=True)
    make_plot(all_instances[0], tree_root,filename)
    # tester
    print()
    print()
    print()
    for inst in all_instances[0]:
        feature = (inst[0], inst[1])
        expected_label = inst[2]
        res = test_decision(feature, expected_label, tree_root)
        if not res:
            print(str(inst)+" is classified incorrectly")

    
    
