
import numpy as np
import time
survey = np.array([
 # sword fights, romantic drama, scifi, good and evil, hate watching long movies, output
 [1, 0, 0, 0, 0, 1],
 [1, 0, 0, 0, 0, 1],
 [1, 0, 0, 0, 0, 1],
 [1, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 0, 0],
 [1, 0, 0, 0, 0, 1],
 [1, 0, 0, 0, 0, 1],

])

features_train = survey[:, 0:5]
labels_train = survey[:, 5]

# Keeping four surveys as our test set
test_survey = np.array([
 [1, 0, 0, 0, 0, 1],
 [1, 0, 0, 0, 0, 1],
 [0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0]

])

features_test = test_survey[:, 0:5]
labels_test = test_survey[:, 5]

from sklearn.neural_network import MLPClassifier
from warnings import filterwarnings
filterwarnings('ignore')

# Define the model
mlp = MLPClassifier(hidden_layer_sizes=(5,),
                   activation='tanh',
                   max_iter=3000,
                   random_state=1
                  )

#Each element in the tuple represents the number of nodes at the ith position where i is the index of the tuple.
#Thus the length of tuple denotes the total number of hidden layers in the network.

# Train the model
mlp.fit(features_train, labels_train)

print("Training set score: %f" % mlp.score(features_train, labels_train))
print("Testing set score: %f" % mlp.score(features_test, labels_test))
print("Type yes or no.")

sword = input("Do you like sword fights?: ")
sword.lower()
if sword == "yes":
   szero = 1
else:
   szero = 0

drama = input("Do you like romantic drama?: ")
drama.lower()
if drama == "yes":
   dzero = 1
else:
   dzero = 0

scifi = input("Is science fiction cool?: ")
scifi.lower()
if scifi == "yes":
   sfzero = 1
else:
   sfzero = 0

good = input("Do you like talking about good and evil?: ")
good.lower()
if good == "yes":
   gzero = 1
else:
   gzero = 0

long = input("Do you hate watching long movies?: ")
long.lower()
if long == "yes":
   lzero = 1
else:
   lzero = 0

features = [[szero, dzero, sfzero, gzero, lzero]]
print(features)
prediction = mlp.predict(features)
print(prediction)
if prediction == 1:
   print("You like Star Wars!")
else:
   print("You don't like Star Wars!")
