#!/bin/python3

import csv
import sys
sys.path.insert(1, '../robustness')
import MTL as MTL 
import numpy as np
import importlib
import time

#csv_filename = "example_data.csv"
if len(sys.argv) == 3:
    csv_filename = sys.argv[1]
    formula_filename = sys.argv[2]
else:
    print("Wrong number of args", file = sys.stderr)
    sys.exit(1)

traces = {}
time_stamps = []

with open(csv_filename, newline='') as csvfile:
    data_file_descripter = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in data_file_descripter:
        if row[0] != "time":
            traces[row[0]] = np.array(row[1:],dtype=np.float32)
        else:
            time_stamps = np.array(row[1:],dtype=np.float32)
        
formula = importlib.import_module("test_formula.py"[:-3])


t0 = time.time()
formula.root.eval_interval(traces, time_stamps)
t1 = time.time()

print("TLTk" ,'\t| Samples:', "{:,}".format(len(time_stamps)), '\t| Time: ', '%.4f'%(t1 - t0), '    \t| robustness', formula.root.robustness)
