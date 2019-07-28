from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
import numpy as np

#
# Compute Input Signal based on control points and interpolation type
#
# INPUTS:
#   * interpolation_type: user defined interpolation function (string name)
#   * cp_samples: vector of control point values
#   * inp_range: n-by-2 array with bounds for each of the n signals
#   * no_control_points: user defined number of control points (integer number)
#   * simulation_time, step: user defined values that construct the time interval starting from 0

# OUTPUTS:
#   * generated input signals


def compute_input_signal(interpolation_type, cp_samples, inp_range, no_control_points, simulation_time, step):
    start = 0
    end = simulation_time
    step_time = np.arange(start, end+step, step)
    time_vector = np.arange(0, simulation_time + 1, (simulation_time / (no_control_points - 1)))

    if interpolation_type == "CubicSpline":
        cs = CubicSpline(cp_samples, step_time)
    elif interpolation_type == "pchip":
        cs = PchipInterpolator(time_vector, cp_samples)
    else:
        cs = PchipInterpolator(time_vector, cp_samples)

    # step_time = query points
    return cs(step_time)
