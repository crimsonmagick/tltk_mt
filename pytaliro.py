import numpy as np
from scipy.optimize import minimize

import auxilliary.computeInputSignal as computeInputSignal
import auxilliary.systemSimulator as systemSimulator


def sim_and_return_rob(z, *params):
    cur_sample = z
    model, opt, interpolation, predicates, rt, eng = params
    step = opt[1]
    inp_range = opt[2]
    simulation_time = opt[3]
    predicate_tags = predicates[0]
    for i in range(0, len(cur_sample)):
        cur_sample[i] = min(max(inp_range[0], cur_sample[i]), inp_range[1])

    signal = computeInputSignal.compute_input_signal(interpolation, cur_sample,
                                                     inp_range, len(cur_sample), simulation_time, step)

    # Simulate the model
    if opt[0] == 'simulink':
        time_stamps, internal_states, output = systemSimulator.simulate_system(eng, model, simulation_time,
                                                                               step, signal)

    # How many predicates?
    no_predicates = len(predicates) - 1

    # Initialize traces dictionary and fill from simulation output
    traces = {}
    for i in range(no_predicates):
        outputs = [row[i] for row in output]
        predicate_data = np.transpose(outputs)
        traces[predicate_tags[i]] = predicate_data

    # Get time stamps from simulation output
    time_stamps = np.ravel(time_stamps)
    time_data = np.transpose(time_stamps)

    # Calculate robustness
    rt.eval_interval(traces, time_data)

    print('---', rt.robustness, ' cps: ', z)

    return rt.robustness


def falsify(model, interpolation, cp_samples, predicates, root, opt):

    # Initialize MATLAB engine
    engine = systemSimulator.init_engine()
    params = (model, opt, interpolation, predicates, root, engine)

    my_opt = {'maxiter': 100, 'disp': True}
    res2 = minimize(sim_and_return_rob, cp_samples, args=params, method='Nelder-Mead', options=my_opt)
    print(res2)

    # Uncomment the section below to use stochasticOptimization functions
    '''
    samples_per_round = 50
    n = 25
    my_opt = [samples_per_round, n]
    res1 = stochasticOptimization.stochastic_optimizer(sim_and_return_rob, cp_samples, *params, options=my_opt)
    print(res1)
    '''

