import math

import numpy
import numpy as np
import matplotlib.pyplot as plt

def readDataSetFile(filename):
  dataset = []
  with open(filename, "r") as file:
    for line in file:
      dataset.append([int(i) for i in line.strip().split()])
      #print(len([int(i) for i in line.strip().split()]))
  return dataset


def normalizeDataSet(s_dataset):
  min = 0
  max = 255
  a = 0
  b = 1
  new_dataset = []
  for data in s_dataset:
    ndata = []
    sumx = 0
    for i in range(0, len(data) - 1):
      sumx += data[i]**2
    sqtsumx = math.sqrt(sumx)
    for i in range(0, len(data) - 1):
      xs = ((data[i] - min) / (max - min))
      #xs = data[i] / sqtsumx
      ndata.append(round(xs, 5))
    ndata.append(data[len(data) - 1])
    new_dataset.append(ndata)
  return new_dataset


def decreaseDataSet(s_dataset):
  n_dataset = []
  for data in s_dataset:
    l_data = data[:27]
    r_data = data[36:]
    r_data[0] -= 1
    n_data = l_data + r_data
    n_dataset.append(n_data)
  return n_dataset


def e_distance(x, y):
  sum = 0
  for xi, yi in zip(x, y):
    #print(xi,yi)
    sum += (xi - yi)**2
  de = math.sqrt(sum)
  return de


#for data in normalizeDataSet(readDataSetFile("sat.trn")):


def r_vector(x, w):
  r = []
  for i in range(0, 7):
    r.append(e_distance(x, w[:, i]))
  return np.array(r)


data = readDataSetFile("sat.trn")
data = normalizeDataSet(data)
data = decreaseDataSet(data)
testing_data = readDataSetFile("sat.tst")
testing_data = normalizeDataSet(testing_data)
testing_data = decreaseDataSet(testing_data)
#print(e_distance(data[0],data[1010]))

#for data in decreaseDataSet(normalizeDataSet(data)):
#  print(f" {data} {len(data)}\n")

input_count = 27  #3x3 rgb colors
kohonenL_count = 7  #count clasters
#output_count = #
learning_rate = 0.1
weights = np.random.uniform(low=(0.5 - (1 / math.sqrt(input_count))),
                            high=(0.5 + (1 / math.sqrt(input_count))),
                            size=(input_count, kohonenL_count))
print(weights)

x_analyze = []
y_analyze = []

epochs = int(input('Введите количество итераций(эпох):'))
for epoch in range(epochs):
  print('epoch:',epoch)
  for set in data:
    i = np.random.randint(0, len(data))
    input_vector = np.array(data[i][:input_count])
    #print('x',input_vector)
    #print('w',weights)
    #print(e_distance(input_vector,weights[:,0]))

    r = r_vector(input_vector, weights)
    wta = data[i][27:]
    for ii in range(0, 27):
      weights[ii][wta] = weights[ii][wta] + learning_rate * (input_vector[ii] -
                                                            weights[ii][wta])

  t_correct = 0
  for t_set in testing_data[:100]:
     r = r_vector(t_set,weights)
     wta = np.argmin(r)
     t_correct += int(wta==t_set[27])
  x_analyze.append(epoch+1)
  y_analyze.append(round((t_correct/2000)*100, 2))

fig1, ax1 = plt.subplots()
ax1.plot(x_analyze,y_analyze)
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy %')
plt.savefig('accuracy_epoch.png')

while True:
  i = int(input('tst i:'))
  x = testing_data[i][:input_count]
  print('x:', testing_data[i])
  r = r_vector(x, weights)
  wta = np.argmin(r)
  print('Cluster:', wta)
