import matlab.engine
import numpy as np
from scipy.optimize import minimize
from numpy import genfromtxt
import auxilliary.computeInputSignal as computeInputSignal
import auxilliary.systemSimulator as systemSimulator

def sim_and_return_rob(z, *params):
    cur_sample = z
    model, opt, interpolation, predicates, rt, eng = params
    step = opt[1]
    inp_range = opt[2]
    simulation_time = opt[3]
    
    for i in range(0, len(cur_sample)):
        cur_sample[i] = min(max(inp_range[0], cur_sample[i]), inp_range[1])

    signal = computeInputSignal.compute_input_signal(interpolation, cur_sample,
                                                     inp_range, len(cur_sample), simulation_time, step)

    # Simulate the model
    if opt[0] == 'simulink':
        time_stamps, internal_states, output = systemSimulator.simulate_system(eng, model, simulation_time,
                                                                               step, signal)
    elif opt[0] == 'function':
        # User can add any desired function here
        # np.sin() is used as a placeholder
        time_stamps = np.linspace(-np.pi, np.pi, 10)
        output = time_stamps + 0.5*np.sin(2*time_stamps)
    else:
        time_stamps = np.array(genfromtxt('data/eqT2.csv'))
        output = np.array(genfromtxt('data/seqS2.csv', delimiter=',')).tolist()

    # How many predicates?
    # no_predicates = len(predicates) - 1

    # Initialize traces dictionary and fill from simulation output
    traces = {}
    tempArray = np.array(output) 
    tempArray = np.stack(tempArray, axis = 1)
    for i in range(0,len(predicates)):
        traces[predicates[i].variable_name] = np.float32(tempArray[i])
        
        
        
    # Get time stamps from simulation output
    time_stamps = np.ravel(time_stamps)
    time_data = np.float32(np.transpose(time_stamps))
    # print(traces)
    # print(time_data)    
    # Calculate robustness
    
    rt.eval_interval(traces, time_data)

    print('---', rt.robustness, ' cps: ', z)

    return rt.robustness


def falsify(model, interpolation, cp_samples, predicates, root, opt):

    # Initialize MATLAB engine
    if opt[0] == 'simulink':
        engine = systemSimulator.init_engine()
    else:
        engine = []
    params = (model, opt, interpolation, predicates, root, engine)

    my_opt = {'maxiter': 300, 'disp': True}
    res2 = minimize(sim_and_return_rob, cp_samples, args=params, method='Powell', options=my_opt)
    print(res2)

    # Uncomment the section below to use stochasticOptimization functions
    '''
    samples_per_round = 50
    n = 25
    my_opt = [samples_per_round, n]
    res1 = stochasticOptimization.stochastic_optimizer(sim_and_return_rob, cp_samples, *params, options=my_opt)
    print(res1)
    '''

