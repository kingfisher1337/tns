import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = "\\usepackage{bm}"

data = dict()

with open("output/J_h_T_E.dat", "r") as f:
    for line in f:
        fields = line.split(" ")
        h = float(fields[1])
        if not data.has_key(h):
            data[h] = ([], [])
        data[h][0].append(float(fields[2])) # T
        data[h][1].append(-float(fields[3])) # E

for h in sorted(data):
    (T,E) = data[h]
    plt.plot(T, E, marker="x", label="$h = {:.1f}$".format(-h))

plt.legend(loc=2)
plt.title("$\\mathcal{H} = - \\sum\\limits_{\\langle j,k \\rangle} \\sigma_z^{(j)} \\sigma_z^{(k)} - h \\sum\\limits_j \\sigma_x^{(j)}$, $\\langle A \\rangle = \\langle \\psi \\vert A \\vert \\psi \\rangle$, $\\vert \\psi \\rangle = \\sum\\limits_{\\bm{\\sigma}} e^{-\\beta \\mathcal{H}_{\\mathrm{Ising}}^{\\mathrm{class.}}(\\bm{\\sigma})} \\vert \\bm{\\sigma} \\rangle$")
plt.xlabel("$T$")
plt.ylabel("$- \\frac{\\langle \\mathcal{H} \\rangle}{N} = 2 \\langle \\sigma_z \\sigma_z \\rangle + h \\langle \\sigma_x \\rangle$")
plt.yscale("log")
plt.ylim(1, 10)
plt.grid(True)
plt.savefig("plots/ising_tfiansatz.png", dpi=300)
plt.close()

for h in sorted(data):
    (T,E) = data[h]
    plt.plot(T, E, marker="x", label="$h = {:.1f}$".format(-h))

plt.legend(loc=2)
plt.title("$\\mathcal{H} = - \\sum\\limits_{\\langle j,k \\rangle} \\sigma_z^{(j)} \\sigma_z^{(k)} - h \\sum\\limits_j \\sigma_x^{(j)}$, $\\langle A \\rangle = \\langle \\psi \\vert A \\vert \\psi \\rangle$, $\\vert \\psi \\rangle = \\sum\\limits_{\\bm{\\sigma}} e^{-\\beta \\mathcal{H}_{\\mathrm{Ising}}^{\\mathrm{class.}}(\\bm{\\sigma})} \\vert \\bm{\\sigma} \\rangle$")
plt.xlabel("$T$")
plt.ylabel("$- \\frac{\\langle \\mathcal{H} \\rangle}{N} = 2 \\langle \\sigma_z \\sigma_z \\rangle + h \\langle \\sigma_x \\rangle$")
#plt.yscale("log")
plt.xlim(0,2.5)
plt.ylim(1.8, 2.5)
plt.grid(True)
plt.savefig("plots/ising_tfiansatz_detail.png", dpi=300)
plt.close()

