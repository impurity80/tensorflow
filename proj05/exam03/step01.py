import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

input = np.transpose(np.loadtxt('in.txt', unpack=True, dtype='float32'))

# input = np.reshape(input, (1,23))
output = []

for i in range(101):
    for j in range(101):
        value = input.copy()
        value[14] = 0.4/100*i
        value[15] = 0.08/100*j
        output.append(value)

np.savetxt('out.txt', output, delimiter=" ", fmt="%s")