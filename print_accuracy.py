from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np
from datetime import timedelta
import os
from tabulate import tabulate
import csv

path_dir = '/mnt/sting/sorn111930/kd/all_file/experiments'
# tinyimagenet = [["date", "from_K_block", "numResNetblock", "model", "PSKD", "DML", "BYOT", 'training_acc','training_time','val_acc_top1','val_time']]
cifar100 = [["from_K_block", "numResNetblock", "model", "PSKD", "DML", "BYOT", 'training_acc', 'training_time', 'val_acc_top1']]
resls = []
for exp in os.listdir(path_dir):
    components = exp.split("_")
    print(components)
    current_path = os.path.join(path_dir, exp, 'tensorboard')
    if not os.path.isdir(current_path):
        print("skipping : {}".format(current_path))
        continue
    
    event_acc = EventAccumulator(current_path)
    event_acc.Reload()
    # Show all tags in the log file
    event_acc.Reload()
    # print(event_acc.Scalars.keys())
    # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
    if len(event_acc.Tags()['scalars']) == 0:
        continue
    
    df_training_acc = pd.DataFrame(event_acc.Scalars('train/acc'))
    a1 = np.array(df_training_acc['value']) 
    a2 = np.array(df_training_acc['wall_time']) 
    training_acc = round(100*float(a1[-1]),5)
    # training_time = str(timedelta(seconds= a2[-1] - a2[0])).split(".")[0]
    try:
        print(a1)
        training_time = ((a2[1] - a2[0]) + (a2[2] - a2[1]))/2.0
        
    except:
        training_time = None
    # print(a2[1], a2[0], a2[1] - a2[0], a2)
    
    
    df_val_acc = pd.DataFrame(event_acc.Scalars('val/val_top1'))
    a1 = np.array(df_val_acc['value']) 
    a2 = np.array(df_val_acc['wall_time']) 
    val_acc = round(float(a1[-1]),5)
    # val_time = str(timedelta(seconds= a2[-1] - a2[0])).split(".")[0]
    
    
    
    fromkblock = int(components[6])
    numresnetblock = int(components[10])
    model = components[12]
    PSKD = components[-7]
    DML = components[-5]
    BYOT = components[-3]
    DMLOutput = components[-1]
    
    # print(date, fromkblock, numresnetblock, model, PSKD, DML, BYOT)
    if 'cifar100' in current_path and '50' in model:
        resls.append([fromkblock, numresnetblock, model, PSKD, DML, BYOT,training_acc, training_time, val_acc])
    # else:
    #     tinyimagenet.append([date, fromkblock, numresnetblock, model, PSKD, DML, BYOT,training_acc, training_time, val_acc, val_time])
# print(training_time['value'].max()-training_time['value'].min())
# print(training_time['wall_time'][-1]-training_time['wall_time'][-1])

# resls.sort(key = lambda x : (x[2],x[3],x[4],x[5]))
cifar100 += resls

print(tabulate(cifar100, headers='firstrow', tablefmt='fancy_grid'))
#print(tabulate(tinyimagenet, headers='firstrow', tablefmt='fancy_grid'))

# with open('tinyimagenet.csv','w') as f:
#     for l in tinyimagenet:
#         wr = csv.writer(f, quoting=csv.QUOTE_ALL)
#         wr.writerow(l)

with open('cifar100.csv','w') as f:
    for l in cifar100:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(l)