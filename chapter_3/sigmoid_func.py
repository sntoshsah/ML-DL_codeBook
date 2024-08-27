import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))
def main():

	# Create a random value of range -7 to 7
	z = np.arange(-7,7,0.2)
	sigma_z = sigmoid(z)

	# plotting the sigmoid function
	plt.plot(z, sigma_z)
	plt.axvline(0.0, color="k")
	plt.ylim(-0.5, 1.5)
	plt.xlabel("z")
	plt.ylabel("$\sigma  (z)$")

	# y-axis tick and gridline
	plt.yticks([0.0, 0.5, 1.0, 1.5])
	ax = plt.gca()
	ax.yaxis.grid(True)
	plt.tight_layout()
	plt.savefig("sigmoid.png")


if __name__ == "__main__":
	main()