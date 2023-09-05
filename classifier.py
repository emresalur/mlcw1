# classifier.py
# Lin Li/26-dec-2021

from collections import Counter
import numpy as np
import random

class Node:
    # class to build the nodes
    def __init__(self, index=None, left=None, right=None, gain=None, value=None, isLeaf=None):

        # initialize the feature index, left and right trees, information gain, leaf node status and value
        self.featureIndex = index
        self.leftTree = left
        self.rightTree = right
        self.infoGain = gain
        self.isLeaf = isLeaf
        self.value = value

class Tree:
    # class to build the decision tree
    def __init__(self):
        # initialize the root, classes and maxDepth of the tree
        self.root = None
        self.maxDepth = 4
        self.classes = [0, 1, 2, 3]
  
    def buildTree(self, data, target, treeDepth):
        # function that recursively builds tree

        # base cases that result in leaf node
        if treeDepth <= self.maxDepth and self.isNotPure(target) and len(data) != 0:
            numFeatures = len(data[0])
            bestSplit = self.getBestSplit(data, target, numFeatures)
            # if best split is not empty then create left and right trees
            if len(bestSplit) != 0:
                # recursively build left and right trees
                rightTree = self.buildTree(bestSplit[3], bestSplit[4], treeDepth + 1)
                leftTree = self.buildTree(bestSplit[1], bestSplit[2], treeDepth + 1)
                return Node(bestSplit[0], leftTree, rightTree, bestSplit[5], isLeaf=False)

        # create a leaf node
        leafNodeValue = random.choice(list(target))
        return Node(value=leafNodeValue, isLeaf=True)

    def getBestSplit(self, data, target, numFeatures):
        # function to find best feature to split on

        # list to store information associated with split
        bestSplit = []
        maxInfoGain = -1

        # loop over all the features to determine best feature to split on
        for i in range(numFeatures):
            # split data into left and right branches
            leftData, leftTarget, rightData, rightTarget = self.split(data, target, i)
            infoGain = self.informationGain(target, leftTarget, rightTarget)
            # if both left and right data are not empty
            if leftData.size and rightData.size:
                # if information gain of new split better, then update bestSplit
                if infoGain > maxInfoGain:
                    bestSplit = [i, leftData, leftTarget, rightData, rightTarget, infoGain]
                    maxInfoGain = infoGain

        # return best split
        return bestSplit

    def split(self, data, target, featureIndex):
        # splits original data into two datasets depending on value of feature
        leftIndices = np.where(data[:, featureIndex] == 0)[0]
        rightIndices = np.where(data[:, featureIndex] == 1)[0]
        leftData, leftTarget = data[leftIndices], target[leftIndices]
        rightData, rightTarget = data[rightIndices], target[rightIndices]
        return leftData, leftTarget, rightData, rightTarget

    def informationGain(self, parent, leftChild, rightChild):
        # computes information gain of split
        leftEntropy = self.calculateEntropy(leftChild)
        rightEntropy = self.calculateEntropy(rightChild)
        childEntropy = (len(leftChild) / len(parent)) * leftEntropy + (len(rightChild) / len(parent)) * rightEntropy
        infoGain = self.calculateEntropy(parent) - childEntropy
        return infoGain

    def calculateEntropy(self, y):
        # calculates entropy of data
        counts = Counter(y)
        proportions = np.array(list(counts.values())) / len(y)
        entropy = -np.sum(proportions * np.log2(proportions))
        return entropy

    def fit(self, data, target):
        # builds decision tree using data and target
        self.root = self.buildTree(np.array(data), np.array(target), 0)

    def predict(self, data, legal=None):
        # predicts class for new data
        node = self.root
        # keep traversing tree until you reach a leaf node
        while not node.isLeaf:
            featureValue = data[node.featureIndex]
            # if value of feature is 0 then take left branch
            if featureValue == 0:
                node = node.leftTree
            # otherwise take right branch
            else:
                node = node.rightTree
        # return value of leaf node
        return node.value

    def isNotPure(self, target):
        # checks if remaining values all belong to same class
        numDiffFeatures = len(set(target))
        if numDiffFeatures == 1:
            return False
        return True

# classifier for the forest

class Classifier:

    def __init__(self):

        # two empty lists to store the forest and the predictions
        self.forest = []
        self.predictions = []
        # number of trees to predict the data
        self.numberOfTrees = 10

    def predict(self, features, legal=None):
        for i in self.forest:
            # create a tree prediction and append it in the predictions collection
            self.predictions.append(i.predict(features, legal))
        # pick the most common element of the predictions list
        counted = Counter(self.predictions)
        most_common = counted.most_common()
        return most_common[0][0]

    def bootstrapping(self, data):
        forestData = []
        for i in range(self.numberOfTrees):
            # empty collection for the data (tuples) of each tree
            treeData = []
            # create n number of tuples (same as in the dataset given)
            for j in range(len(data)):
                # append a random tuple into the tree data collection
                treeData.append(data[random.randint(0, len(data) - 1)])
            
            # append the tree data into the forest data collection
            forestData.append(treeData)
        return forestData

    def fit(self, data, target):
        # via bootsrapping create a new dataset containing 9 elements of 30 tuples to create the trees
        forestData = self.bootstrapping(data)
        self.forest = []
        for i in forestData:
            tree = Tree()
            # fit the data into the newly created tree
            tree.fit(i, target)
            # append the tree into the forest collection
            self.forest.append(tree)

    def reset(self):
        # reset the forest and the predictions
        self.forest = []
        self.predictions = []