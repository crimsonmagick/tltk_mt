from computeInputSignal import ComputeInputSignal
import numpy as np
from MTL import *
import time
from numpy import genfromtxt
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support
import os
from systemSimulator import *
import scipy
import random
from scipy import optimize
from math import cos, sin


def sampleFromDistribution(distrib, lo, hi, n):
    sampProb = 1.0
    r0 = random.random()
    s = 0
    j = 0

    for i in range(n):
        s = s + distrib[i]
        if s >= r0:
            sampProb = distrib[i]
            j = i
            break
    assert (j >= 0)
    assert (j <= n)
    r1 = random.random()
    delta = (hi - lo) / n
    sampValue = lo + (j + r1 - 1) * delta
    return sampValue, sampProb


def chooseBernoulliSamples(inpRanges,subDivs,distrib):
    nInputs = len(inpRanges)
    sampleWeight = 1.0
    sample = [0] * nInputs
    for i in range(nInputs):
        n = subDivs[i]
        sample[i], sProb = sampleFromDistribution(distrib[i], inpRanges[i][0], inpRanges[i][1], n)
    return sample


