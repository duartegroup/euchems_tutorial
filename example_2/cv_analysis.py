import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import ase.io as aio

rc("text", usetex=True)

data = aio.read("endo_ace_al.xyz", index=":")


collective_variable = []

for structure in data:
    r1 = structure.get_distance(1, 12)
    r2 = structure.get_distance(6, 11)
    collective_variable.append(0.5 * (r1 + r2))


x = np.arange(0, len(collective_variable))

clas = np.where(
    np.array(collective_variable) < 1.8,
    "PS",
    np.where(np.array(collective_variable) > 2.8, "RS", "TS"),
)

cdict = {"RS": "red", "TS": "blue", "PS": "black"}

fig, ax = plt.subplots()
for c in np.unique(clas):
    ix = np.where(clas == c)
    ax.scatter(x[ix], np.array(collective_variable)[ix], c=cdict[c], label=c, s=10)

plt.xlabel("Index")
plt.ylabel(r"$\frac{(r_1+r_2)}{2}$")
plt.title("Data points")
plt.legend()

plt.savefig("r12_dataset.pdf", bbox_inches="tight")
