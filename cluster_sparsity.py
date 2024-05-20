import matplotlib.pyplot as plt
import pickle

with open("/home/hzlin/Documents/LibMultiLabel/eurlex_children_representation_elkan.pkl", "rb") as f:
    children_representation = pickle.load(f)
    print(children_representation)

X = []
Y = []
for i, element in enumerate(children_representation):
    X += [i+1] * len(element)
    Y += list(element)

plt.scatter(X, Y, s=0.3)
plt.yscale("log")
plt.show()