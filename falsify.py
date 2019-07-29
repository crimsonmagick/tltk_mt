from auxilliary import systemSimulator
from auxilliary import computeInputSignal
import stochasticOptimization
from scipy.optimize import minimize
import numpy as np


def sim_and_return_rob(z, *params):
    cur_sample = z
    model, step, inp_range, simulation_time, interpolation, pred_tags, rt, eng = params

    for i in range(0, len(cur_sample)):
        cur_sample[i] = min(max(inp_range[0], cur_sample[i]), inp_range[1])

    signal = computeInputSignal.compute_input_signal(interpolation, cur_sample,
                                                     inp_range, len(cur_sample), simulation_time, step)

    # Simulate the model
    time_stamps, internal_states, output = systemSimulator.simulate_system(eng, model, simulation_time, step, signal)
    outputs = [row[0] for row in output]
    speed_data = np.transpose(outputs)  # signal 1
    outputs = [row[1] for row in output]
    rpm_data = np.transpose(outputs)  # signal 2
    time_stamps = np.ravel(time_stamps)
    time_data = np.transpose(time_stamps)
    traces = {pred_tags[0]: speed_data, pred_tags[1]: rpm_data}

    rt.eval_interval(traces, time_data)

    print('---', rt.robustness, ' cps: ', z)

    return rt.robustness


def falsify(model, step, inp_range, simulation_time, interpolation, pred_tags, root):
    # Initialize matlab engine
    engine = systemSimulator.init_engine()
    params = (model, step, inp_range, simulation_time, interpolation, pred_tags, root, engine)

    cp_samples = np.random.uniform(low=inp_range[0], high=inp_range[1], size=(2,))

    myopt = {'maxiter': 100, 'disp': True}
    res2 = minimize(sim_and_return_rob, cp_samples, args=params, method='Nelder-Mead', options=myopt)
    print(res2)

    # Uncomment the section below to use stochasticOptimization functions
    '''
    samples_per_round = 50
    n = 25
    my_opt = [samples_per_round, n]
    res1 = stochasticOptimization.stochastic_optimizer(sim_and_return_rob, cp_samples, *params, options=my_opt)
    print(res1)
    '''


