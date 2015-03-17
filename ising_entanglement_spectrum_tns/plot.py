import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = "\usepackage{bm}"

Ly = 6
ymin = -2
ymax = 40
entropies = dict()

for filename in sorted(os.listdir("output")):
    if filename.startswith("entropy") and filename.endswith(".dat"):
        T = filter(lambda s: s.find("T=") != -1, filename[:-4].split("_"))[0].split("=")[-1]
        l = []
        S = []
        with open("output/" + filename, "r") as f:
            for line in f:
                fields = line.split(" ")
                l.append(int(fields[0]))
                S.append(float(fields[1]))
        plt.plot(l, S, marker="x", label="$T = " + T + "$")
        entropies[T] = S

plt.legend(loc=1)
plt.xlabel("$l = \\frac{L_x}{2}$")
plt.ylabel("$S = \\sum\\limits_j \\xi_j \\ln(\\xi_j)$")
plt.grid(True)
plt.savefig("plots/entropy_l_T_Ly=" + str(Ly) + ".png")
plt.close()

for filename in sorted(os.listdir("output")):
    if filename.startswith("entanglement_spectrum") and filename.endswith(".dat"):
        T = filter(lambda s: s.find("T=") != -1, filename[:-4].split("_"))[0].split("=")[-1]
        Lx = filter(lambda s: s.find("Lx=") != -1, filename[:-4].split("_"))[0].split("=")[-1]
        with open("output/" + filename, "r") as f:
            for line in f:
                fields = line.split(" ")
                if len(fields) == 3:
                    symbol = "r+" if np.abs(float(fields[1])) < 1e-10 else "bx"
                    plt.plot(float(fields[0]), -2.0*np.log(float(fields[2])), symbol)
            plt.title("$T = " + str(T) + "$, $H = 0$,\n$L_y = " + str(Ly) + "$, $L_x = " + Lx + "$, $\\ell = " + str(int(Lx)/2) + "$,\n$S = " + str(entropies[T][int(Lx)/2-1]) + "$")
            plt.subplots_adjust(top=0.85)
            plt.axes().set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            plt.axes().set_xticklabels(["$-\\pi$", "$-\\frac{\\pi}{2}$", "$0$", "$\\frac{\\pi}{2}$", "$\\pi$"])
            plt.xlim(-np.pi-0.2, np.pi+0.2)
            if not ymin is None:
                plt.ylim(ymin, ymax)
            plt.xlabel("$k$")
            plt.ylabel("$-2 \\ln(\\xi) \\quad$ (note: $\\xi^2 \\in \\sigma(\\rho_A)$, even $\\hat{=}$ red, odd $\\hat{=}$ blue)")
            plt.grid(True)
            ylim = plt.axes().get_ylim()
            aspectRatio = 3.0 * (2.0*np.pi+0.4) / (ylim[1] - ylim[0])
            plt.axes().set_aspect(aspectRatio)
            plt.gcf().set_size_inches(4.0, 6.0)
            plt.savefig("plots/entanglement_spectrum_T=" + T + "_Ly=" + str(Ly) + "_Lx=" + Lx + ".png", dpi=300)
            plt.close()

