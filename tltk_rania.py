import torch
from numpy import random
from scipy.optimize import minimize
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

import auxilliary.computeInputSignal as computeInputSignal

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder("/data/imagenet", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval().cuda()

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in tqdm(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy on Tiny ImageNet train subset: {100 * correct / total:.2f}%")



def sim_and_return_rob(z, *params):
    cur_sample = z
    model, opt, interpolation, predicates, rt, iterations = params
    step = opt[1]
    inp_range = opt[2]
    simulation_time = opt[3]

    for i in range(0, len(cur_sample)):
        cur_sample[i] = min(max(inp_range[0], cur_sample[i]), inp_range[1])

    signal = computeInputSignal.compute_input_signal(interpolation, cur_sample,
                                                     inp_range, len(cur_sample), simulation_time, step)
    # print("SIGNAL:", signal.tolist())
    # print(len(signal))
    # print(inp_range)
    # x = input()
    # -------------------------------
    # In this part, the interpolated signal is a DNN configuration.
    # The DNN configuration should be reflected through tools/distiller/etc to the target DNN.
    # Then, inference is run and we acquire the accuracy trajectory in the data/data.csv.
    # For the time being, placeholder values are used for a 100x100 inference example.
    # -------------------------------
    # Get accuracy from file
    output = random.randint(50, 70, size=(100))  # pd.read_csv('data/data.csv', header=None)[1]
    # Generate time stamps
    time_stamps = np.linspace(0, iterations, len(output))

    # Initialize traces dictionary and fill from simulation output
    traces = {}
    traces['accuracy'] = np.array(output)

    # Get time stamps from simulation output
    time_stamps = np.ravel(time_stamps)
    time_data = np.float32(np.transpose(time_stamps))

    # Calculate robustness
    rt.reset()
    rt.eval_interval(traces, time_data)

    print('Rob:', rt.robustness, 'CPs: ', z, "Average accuracy: ", np.average(output), " MIN: ", output.min(), " MAX: ",
          output.max())

    return rt.robustness


def falsify(model, interpolation, cp_samples, predicates, root, opt):
    iterations = opt[-1]
    params = (model, opt, interpolation, predicates, root, iterations)

    my_opt = {'maxiter': 1000, 'disp': True}
    result = minimize(sim_and_return_rob, np.ravel(cp_samples), args=params, method='Nelder-Mead', options=my_opt)

    return result.x




import sys
sys.path.insert(1, '../../')
import numpy as np
import tltk_mtl as MTL


# Define number of convolution DNN layers
layers = 5

# Define output trajectory granularity
iterations = 100

model = 'blackbox'
model_outputs = ['accuracy']
step = 1
simulation_time = iterations

# Optimization vector configuration
pruning_ratio_LB = 0
pruning_ratio_UB = 9
pruning_ratio = [[pruning_ratio_LB,pruning_ratio_UB] for i in range(layers)]
weight_quant = [[5,8] for i in range(layers)]
activation_quant = [[5,8] for i in range(layers)]

# Flatten final vector and define control point bounds
inp_range = [item for sublist in pruning_ratio + weight_quant + activation_quant for item in sublist]
cp_bounds = [(pruning_ratio_LB,pruning_ratio_UB) for i in range(layers)] + [(5,8) for i in range(layers)] + [(5,8) for i in range(layers)]

opt = ['data', step, inp_range, simulation_time, model_outputs, iterations]

interpolation = 'pconst'

mode = 'cpu_threaded'

# STL formula definition in the form of Ax <= b (good set)
r1 = MTL.Predicate('accuracy',-1,-75)
root = MTL.Global(0,float('inf'),r1)

predicates = [r1]

results = falsify(model, interpolation, cp_bounds, predicates, root, opt)

# print(results)


