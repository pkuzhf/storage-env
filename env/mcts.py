import random
import math
import hashlib
import logging
import argparse
import numpy as np
import config

# MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration.
SCALAR = 1 / math.sqrt(2.0)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('MyLogger')

actions_to_paths = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
                    [1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1],
                    [1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1]]


def choose(seq):
    choice = int(random.random() * len(seq))
    if choice == len(seq):
        choice -= 1
    return seq[choice]


class State:
    NUM_TURNS = config.Map.Width * config.Map.Width
    ACTIONS = [i for i in range(14)]
    MAX_VALUE = 1000
    num_moves = 14

    def __init__(self, value=0, moves=[], step=NUM_TURNS):
        self.value = value
        self.step = step
        self.moves = moves

    def next_state(self):
        nextmove = choose(self.ACTIONS)
        next = State(0, self.moves + [nextmove], self.step - 1)
        return next

    def terminal(self):
        if self.step == 0:
            return True
        return False

    def reward(self):
        r = 1 # TODO
        return r

    def __hash__(self):
        return int(hashlib.md5(str(self.moves)).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        s = "Value: %d; Moves: %s" % (self.value, self.moves)
        return s


class Node:
    def __init__(self, state, parent=None):
        self.visits = 0
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)
        return child

    def update(self, reward):
        self.reward += reward
        self.visits += 1

    def fully_expanded(self):
        if len(self.children) == self.state.num_moves:
            return True
        return False

    def __repr__(self):
        s = "Node; children: %d; visits: %d; reward: %f" % (len(self.children), self.visits, self.reward)
        return s

class MyMCTS:
    def __init__(self):
        self.root = Node(State())

    def SEARCHNODE(self, pathes):
        current_node = self.root
        for i in range(config.Map.Width):
            for j in range(config.Map.Height):
                if pathes[i][j][0] == -1:
                    return current_node
                action = actions_to_paths.index(pathes[i][j].tolist())
                tried_children = current_node.children
                target_child = None
                for k in range(len(tried_children)):
                    if action == tried_children[k].state.moves[-1]:
                        target_child = tried_children[k]
                        break
                if target_child is not None:
                    current_node = target_child
                else:
                    current_node = current_node.add_child(State(0, current_node.state.moves + [action], current_node.state.step - 1))

        return current_node


    def UCTSEARCH(self, budget, root):
        for iter in range(budget):
            if iter % 10000 == 9999:
                logger.info("simulation: %d" % iter)
                logger.info(root)
            front = self.TREEPOLICY(root)
            reward = self.DEFAULTPOLICY(front.state)
            self.BACKUP(front, reward)
        return self.BESTCHILD(root, 0)


    def TREEPOLICY(self, node):
        while node.state.terminal() == False:
            if node.fully_expanded() == False:
                return self.EXPAND(node)
            else:
                node = self.BESTCHILD(node, SCALAR)
        return node

    def TREEPOLICYEND(self, node):
        while node.state.terminal() == False:
            if node.fully_expanded() == False:
                node = self.EXPAND(node)
            else:
                node = self.BESTCHILD(node, SCALAR)
        return node

    def EXPAND(self, node):
        tried_children = [c.state for c in node.children]
        new_state = node.state.next_state()
        while new_state in tried_children:
            new_state = node.state.next_state()
        node.add_child(new_state)
        return node.children[-1]


    # current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
    def BESTCHILD(self, node, scalar):
        bestscore = 0.0
        bestchildren = []
        for c in node.children:
            if c.visits == 0:
                exploit = 0
                explore = 1000
            else:
                exploit = c.reward / c.visits
                explore = math.sqrt(math.log(2 * node.visits) / float(c.visits))
            score = exploit + scalar * explore
            if score == bestscore:
                bestchildren.append(c)
            if score > bestscore:
                bestchildren = [c]
                bestscore = score
        if len(bestchildren) == 0:
            logger.warn("OOPS: no best child found, probably fatal")
        return choose(bestchildren)


    def DEFAULTPOLICY(self, state):
        while state.terminal() == False:
            state = state.next_state()
        return state.reward()


    def BACKUP(self, node, reward):
        while node != None:
            node.visits += 1
            node.reward += reward
            node = node.parent
        return

    def MOVETOPATH(self, state):
        pathes = -np.ones([config.Map.Width * config.Map.Height, 4], dtype=np.int64)
        for i in range(len(state.moves)):
            pathes[i] = np.array(actions_to_paths[state.moves[i]])
        pathes = pathes.reshape((config.Map.Width, config.Map.Height, 4))
        print pathes.shape
        return pathes

