import numpy as np
import matlab.engine


# Simulate system function
# INPUTS:
#   * eng: matlab engine instance
#   * model: simulink (for the time being) model (string name)
#   * simulationTime: user defined simulation time
#   * inpSignal: input signal to the model
# OUTPUTS:
#   * timeStamps: timestamps of simulated trajectory
#   * internalStates: internal states at the timestamps
#   * outputs: outputs at the timestamps

def simulate_system(eng, model, simulation_time, inp_signal):
    start = 0
    end = 30
    step = 0.05
    steptime = (np.arange(start, end + step, step)).tolist()
    simopt = eng.simget(model)
    sim_time_array = matlab.double([0, simulation_time])
    signal_array = []
    for i in range(len(steptime)):
        signal_array.append([steptime[i], inp_signal[i]])

    signal_array = matlab.double(signal_array)

    # Commence simulink model simulation

    time_stamps, internal_states, outputs = eng.sim(model, sim_time_array, [], signal_array, nargout=3)

    return time_stamps, internal_states, outputs


def init_engine():
    eng = matlab.engine.start_matlab()
    return eng
