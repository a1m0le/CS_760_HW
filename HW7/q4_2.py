import numpy as np
import numpy.random as random

ALPHA = 0.5
STEP_COUNT = 200
GAMMA = 0.8
EPSILON = 0.5

# 0 --> A, 1 --> B
# 0 --> stay, 1--> move

def epi_greedy_act(Q, state):
    roll = random.uniform()
    if roll <= EPSILON:
        return random.randint(0,2)
    to_move = Q[state][1]
    to_stay = Q[state][0]
    if to_stay > to_move:
        return 0 #stay
    else:
        return 1 # move otherwise


def make_step(st, at):
    if at == 0: # stay
        return st, 1
    elif at == 1:
        return int(not st), 0 #move
    else:
        print("What???")
        exit()


def run():
    Q = np.array([[0.0,0.0],[0.0,0.0]])
    st = 0
    for i in range(STEP_COUNT):
        at = epi_greedy_act(Q, st)
        stp1, rst= make_step(st, at)
        maxQtp1 = np.max(Q[stp1])
        learning_leap = rst + GAMMA * maxQtp1 - Q[st][at]
        learning_step = learning_leap * ALPHA
        new_val = Q[st][at] + learning_step
        Q[st][at] = new_val
        print(st)
        print(at)
        print(stp1)
        print(rst)
        print(new_val)
        print(Q)
        print("--------------")
        st = stp1
    print("Finally, we get this Q:")
    print(Q)

if __name__=="__main__":
    run()
