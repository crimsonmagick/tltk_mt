from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
import numpy as np

# Compute Input Signal based on control points and interpolation type
# INPUTS:
#   * interpolationType: user defined interpolation function (string name)
#   * cpSamples: vector of control point values
#   * inpRange: n-by-2 array with bounds for each of the n signals
#   * noControlPoints: user defined number of control points (integer number)
#   * simulationTime, step: user defined values that construct the time interval starting from 0

# OUTPUTS:
#   * inputSignals: generated input signals


def compute_input_signal(interpolation_type, cp_samples, inp_range, no_control_points, simulation_time, step):
    start = 0
    end = simulation_time
    steptime = np.arange(start, end+step, step)
    time_vector = np.arange(0, simulation_time + 1, (simulation_time / (no_control_points - 1)))

    cs = []
    if interpolation_type == "CubicSpline":
        cs = CubicSpline(cp_samples, steptime)
    elif interpolation_type == "pchip":
        cs = PchipInterpolator(time_vector, cp_samples)
    else:
        cs = PchipInterpolator(time_vector, cp_samples)
    # steptime = query points
    '''
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(time_vector, cpSamples, label='init')
    ax.plot(steptime, cs(steptime), label="S")
    ax.legend(loc='lower left', ncol=2)
    plt.show()
'''
    return cs(steptime)
