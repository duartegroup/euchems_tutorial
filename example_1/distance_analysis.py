import numpy as np
import matplotlib.pyplot as plt
import ase.io as aio

data = aio.read("mg_aqua_al.xyz", index=":")

distances = []

for structure in data:
    distances.append(structure.get_distances(0, np.arange(1, 19, 3)))


plt.hist(np.concatenate(distances), bins=30, color="skyblue", edgecolor="black")


plt.xlabel("Distance (\AA)")
plt.ylabel("Frequency")
plt.title("Mg-O distance")

plt.savefig("mg_o_dataset.pdf", bbox_inches="tight")
