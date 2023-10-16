


class neural_node:

    def __init__(self, role, init_val):
        self.role = role # whether it is a input, output, or hidden layer node
        self.val = init_val


class neural_edge:

    # The structure is in the form of:
    # prev input --- weight ---> middle node ---> activation --> output(used as input for the next edge)

    def __init__(self, in_node, weight, out_node, activation):
        self.in_node = in_node
        self.weight = self.weight
        self.out_node = out_node
        self.activation = activation
        #middle node
        self.mid_node = neural_node(4)

    # when we do a prediction:
    def charge_forward():
        # 




    # when we update


