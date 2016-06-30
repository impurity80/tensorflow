
import numpy as np
import csv

train_xy = []
test_xy = []
xy = []

with open('creep_LMP.csv','r') as f:
    f.readline()
    reader = csv.reader(f)
    for r in reader:
        xy.append(r)
        if np.random.random_sample() > 0.3:
            train_xy.append(r)
        else:
            test_xy.append(r)
    f.close()

with open('train.csv','w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='|')
    for x in train_xy:
        writer.writerow(x)
    f.close()

with open('test.csv','w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='|')
    for x in test_xy:
        writer.writerow(x)
    f.close()

npxy = np.array(xy)
minlist =  npxy.argmin(axis=0)
maxlist = npxy.argmax(axis=0)
print np
print npxy
#print minlist, minlist[0]
#print xy[minlist[2]][2], xy[maxlist[2]][2]
#print npxy.argmax(axis=0)
# print np.amin(x, axis=0)
