import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = "\usepackage{bm}"

for filename in sorted(os.listdir("output")):
    if filename.startswith("sigmax_sigmaz_T_"):
        print filename
        T = []
        mx = []
        mz = []
        err = []
        with open("output/" + filename, "r") as f:
            for line in f:
                fields = line.split(" ")
                T.append(float(fields[0]))
                mx.append(float(fields[1]))
                mz.append(float(fields[2]))
                if len(fields) > 3:
                    err.append(float(fields[3]))
        if len(err) > 0:
            plt.errorbar(T, mx, marker="x", label="$\\langle \\sigma_x \\rangle_{\\infty}$", yerr=err)
            plt.errorbar(T, mz, marker="x", label="$\\langle \\sigma_z \\rangle_{\\infty}$", yerr=err)
        else:
            plt.plot(T, mx, marker="x", label="$\\langle \\sigma_x \\rangle_{\\infty}$")
            plt.plot(T, mz, marker="x", label="$\\langle \\sigma_z \\rangle_{\\infty}$")
        plt.legend(loc=1)
        plt.xlabel("$T$")
        plt.ylim(0,1)
        plt.grid(True)
        plt.savefig("plots/" + filename.split(".dat")[0] + ".png")
        plt.close()
    elif filename.startswith("sigmazsigmaz_T_"):
        print filename
        T = []
        zz = []
        err = []
        with open("output/" + filename, "r") as f:
            for line in f:
                fields = line.split(" ")
                T.append(float(fields[0]))
                zz.append(float(fields[1]))
                if len(fields) > 2:
                    err.append(float(fields[2]))
        if len(err) > 0:
            plt.errorbar(T, zz, marker="x", yerr=err)
        else:
            plt.plot(T, zz, marker="x")
        plt.xlabel("$T$")
        plt.ylabel("$\\langle \\sigma_z^{(j)} \\sigma_z^{(j+1)} \\rangle$")
        plt.grid(True)
        plt.savefig("plots/" + filename.split(".dat")[0] + ".png")
        plt.close()

