
import numpy as np
from keras.models import load_model
import pickle

model = load_model('full_CNN_model.h5')

X = pickle.load(open("X.p"))
Y = pickle.load(open("Y.p"))


X1 = X[0]

Y1_hat = model.predict(X1)

print("result = ", np.array_equal(Y1_hat, Y[0]))

