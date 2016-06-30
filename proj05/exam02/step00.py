import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

xy = np.genfromtxt('creep_LMP.csv',delimiter=',', dtype='float32')

xy = xy[1:,0:-1]

with open('MINMAX.csv','w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='|')
    for y in np.transpose(xy):
        writer.writerow([y.min(), y.max(), y.argmin(), y.argmax(), y.mean(), y.std()])
    f.close()

train_xy = []
test_xy = []

for x in xy:
    if np.random.random_sample() > 0.3:
        train_xy.append(x)
    else:
        test_xy.append(x)

np.savetxt('train.csv', train_xy, delimiter=',', fmt='%s')
np.savetxt('test.csv', test_xy, delimiter=',', fmt='%s')


# np.tofile('train.csv', sep=",")

# print xy[:,0].min(), xy[:,0].max(), xy[:,0].argmin(), xy[:,0].argmax(), xy[:,0].mean(), xy[:,0].std()
