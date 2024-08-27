# code snippet to create a plot that illustrates the loss of classifying a single training example for different values of sigmoid of z.
import numpy as np
import matplotlib.pyplot as plt


# Sigmoid function gives the output ranges from 0 to 1
def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def loss_1(z):
	return -np.log(sigmoid(z))

def loss_0(z):
	return -np.log(1-sigmoid(z))

z = np.arange(-10,10,0.1)
sigma_z = sigmoid(z)

c1 = [loss_1(x) for x in z]
plt.plot(sigma_z, c1, label="L(w,b) if y=1")
c0 = [loss_0(x) for x in z]
plt.plot(sigma_z, c0, label="L(w,b) if y=0", linestyle="--")
plt.ylim(0.0,5.5)
plt.xlabel("$\sigma (z) $")
plt.ylabel("L(W,b)")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Loss_in_sigmoid.png")