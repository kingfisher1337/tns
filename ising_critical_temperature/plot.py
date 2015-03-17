import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = "\usepackage{bm}"

T = []
S = []

with open("output/entropy_bulk_T.dat", "r") as f:
    for line in f:
        fields = line.split(" ")
        T.append(float(fields[0]))
        S.append(float(fields[1]))

plt.plot(T, S, marker="x")
plt.title("$H = 0$, $L_y = 6$")
plt.xlabel("$T$ $[J]$")
plt.ylabel("$\\lim_{L_x \\rightarrow \\infty} S$")
plt.grid(True)
plt.savefig("plots/entropy_bulk_T.png", dpi=300)
plt.close()


T = []
S = []

with open("output/entropy_Lx=2_T.dat", "r") as f:
    for line in f:
        fields = line.split(" ")
        T.append(float(fields[0]))
        S.append(float(fields[1]))

plt.plot(T, S, marker="x")
plt.title("$H = 0$, $L_y = 6$")
plt.xlabel("$T$ $[J]$")
plt.ylabel("$S(L_x=2)$")
plt.grid(True)
plt.savefig("plots/entropy_Lx=2_T.png", dpi=300)
plt.close()

