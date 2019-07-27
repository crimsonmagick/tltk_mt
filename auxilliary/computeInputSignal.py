from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
import numpy as np
import matplotlib.pyplot as plt

# Compute Input Signal based on control points and interpolation type
# INPUTS:
#   * interpolationType: user defined interpolation function (string name)
#   * cpSamples: vector of control point values
#   * inpRange: n-by-2 array with bounds for each of the n signals
#   * noControlPoints: user defined number of control points (integer number)
#   * simulationTime, step: user defined values that construct the time interval starting from 0

# OUTPUTS:
#   * inputSignals: generated input signals


def ComputeInputSignal(interpolationType, cpSamples, inpRange, noControlPoints, simulationTime, step):
    start = 0
    end = simulationTime
    steptime = np.arange(start, end+step, step)
    timeVector = np.arange(0, simulationTime + 1, (simulationTime / (noControlPoints - 1)))

    cs = []
    if interpolationType == "CubicSpline":
        cs = CubicSpline(cpSamples, steptime)
    elif interpolationType == "pchip":
        cs = PchipInterpolator(timeVector, cpSamples)
    else:
        cs = PchipInterpolator(timeVector, cpSamples)
    # steptime = query points
    '''
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(timeVector, cpSamples, label='init')
    ax.plot(steptime, cs(steptime), label="S")
    ax.legend(loc='lower left', ncol=2)
    plt.show()
'''
    return cs(steptime)
