#!/bin/python3
#Please do not use this for large amounts of data. It is probably really slow 
import csv
import sys
sys.path.insert(1, '../robustness')
import MTL as MTL 
import numpy as np
import importlib
import time

skip = 0
position = 3
#csv_filename = "example_data.csv"
higher_dim_mapping = {}
verbose = False

if len(sys.argv) == 3:
    csv_filename = sys.argv[1]
    formula_filename = sys.argv[2]
elif len(sys.argv) > 3:
    csv_filename = sys.argv[1]
    formula_filename = sys.argv[2]
    for arg in sys.argv[3:]:
        if skip != 0:
            skip -= 1
            continue
        if arg == '-h':
            pred_name = sys.argv[position + 1]
            higher_dim_mapping[pred_name] = []
            for row_number in sys.argv[position + 2].split(','):
                higher_dim_mapping[pred_name].append(int(row_number))
            skip += 2
        if arg == "-v":    
            verbose  = True
            
        position+=1
            

traces = {}
time_stamps = []

with open(csv_filename, newline='') as csvfile:
    data_file_descripter = list(csv.reader(csvfile, delimiter=',', quotechar='"'))
    for name in higher_dim_mapping.keys():
        iterts = []
        for row_number in higher_dim_mapping[name]:
            iterts.append(data_file_descripter[row_number][1:])
        
        data_file_descripter = [i for j, i in enumerate(data_file_descripter) if j not in higher_dim_mapping[name]]
        traces[name] = np.array(list(zip(*iterts)),dtype=np.float64)

    for row in data_file_descripter:
            if row[0] != "time":
                traces[row[0]] = np.array(row[1:],dtype=np.float32)
            else:
                time_stamps = np.array(row[1:],dtype=np.float32)
      
formula = importlib.import_module(formula_filename[:-3])


t0 = time.time()
formula.root.eval_interval(traces, time_stamps)
t1 = time.time()
if not verbose:
    print(formula.root.robustness)
else:
    print("TLTk" ,'\t| Samples:', "{:,}".format(len(time_stamps)), '\t| Time: ', '%.4f'%(t1 - t0), '    \t| robustness', formula.root.robustness)
