from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy, math

numpy.random.seed(3) # seed random values

dataset = numpy.loadtxt('all_songs.csv', delimiter=',', skiprows=1) # load in data
X = dataset[:,0:11][0:math.floor(len(dataset[:,0:11]) * 0.7)] # take in 70% of data for training
Y = dataset[:,11][0:math.floor(len(dataset[:,11]) * 0.7)]

X_test = dataset[:,0:11][math.floor(len(dataset[:,0:11]) * 0.7):] # take in 30% of data for training
Y_test = dataset[:,11][math.floor(len(dataset[:,0:11]) * 0.7):]

model = Sequential() # sequential model
model.add(Dense(12, input_dim=11, activation='relu')) # input layer
model.add(Dense(20, activation='relu')) # 20 neuron hidden layer
model.add(Dense(1, activation='sigmoid')) # 1 output

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # compile model

model.fit(X, Y, epochs=50, batch_size=10, class_weight={-1: 10, 1: 1, 0: 10}) # train model with class weights to account for unbalanced data

scores = model.evaluate(X_test, Y_test) # test data
print('{}: {}'.format(model.metrics_names[1], scores[1] * 100)) # print data
