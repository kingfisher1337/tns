import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from hashlib import md5

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = "\usepackage{bm}"

def read_hash(path):
    if not os.path.isfile(path):
        return ""
    with open(path, "rb") as f:
        md = f.read()
    return md
def write_hash(path, h):
    f = open(path, "wb")
    f.write(h)
    f.close()

data = dict()
err = dict()
drawnAnything = False

for filename in sorted(os.listdir("output")):
    if filename.startswith("h_sigmax_sigmaz_") and filename.find("svalerr") != -1:
        print filename
        h = []
        mx = []
        mz = []
        mxerr = []
        mzerr = []
        xierr = []
        md = md5()
        with open("output/" + filename, "r") as f:
            chi = filter(lambda s: s.find("chi=") != -1, filename[:-4].split("_"))[0].split("=")[-1]
            svalerr = float(filter(lambda s: s.find("svalerr=") != -1, filename[:-4].split("_"))[0].split("=")[-1])
            for line in f:
                fields = line.split(" ")
                h.append(-float(fields[0]))
                mx.append(float(fields[1]))
                mz.append(float(fields[2]))
                mxerr.append(float(fields[3]))
                mzerr.append(float(fields[4]))
                xierr.append(float(fields[5]))
                md.update(line)
        
        #svalerr = np.max(xierr[10:])
        
        if md.hexdigest() != read_hash("plots/" + filename.split(".dat")[0] + ".md5"):
            #if len(mzerr) > 0:
            #    plt.errorbar(h, mz, marker="x", label="$\\langle \\sigma_z \\rangle_{0,\\infty}$", yerr=mzerr)
            #    plt.errorbar(h, mx, marker="x", label="$\\langle \\sigma_x \\rangle_{0,\\infty}$", yerr=mxerr)
            #else:
            plt.plot(h, mz, marker="x", label="$\\langle \\sigma_z \\rangle_{0,\\infty}$")
            plt.plot(h, mx, marker="x", label="$\\langle \\sigma_x \\rangle_{0,\\infty}$")
            
            plt.title("1D-TFI ground state via 2D-CTMRG ($\\chi = " + chi + "$, $\\Delta \\xi = 10^{" + "{:.0f}".format(np.log10(svalerr)) + "}$), $\\mathcal{H} = -\\sum\\limits_{\\langle j,k \\rangle} \\sigma_z^{(j)} \\sigma_z^{(k)} + h \\sum\\limits_{j} \\sigma_x^{(j)}$")
            plt.legend(loc=5)
            plt.xlabel("$-h$")
            plt.ylim(0,1)
            plt.grid(True)
            plt.savefig("plots/" + filename.split(".dat")[0] + ".png", dpi=300)
            plt.close()
            write_hash("plots/" + filename.split(".dat")[0] + ".md5", md.hexdigest())
            drawnAnything = True
        
        if not data.has_key(chi) or svalerr < err[chi]:
            data[chi] = (h,mx,mz,mxerr,mzerr)
            err[chi] = svalerr

if drawnAnything:
    h = []
    mx = []
    mz = []
    with open("output_valentin/chi=48.dat", "r") as f:
        firstLine = True
        for line in f:
            if firstLine:
                firstLine = False
            else:
                fields = line.split("\t")
                if len(fields) == 12:
                    h.append(fields[0])
                    mx.append(fields[6])
                    mz.append(fields[11])
                else:
                    print "ignoring illegal line \"{0}\"".format(line)
    plt.plot(h, mz, marker="x", label="Valentin iTEBD, $\\chi = 48$")
    for chi in sorted(data):
        if int(chi) % 8 == 0:
            plt.plot(data[chi][0], data[chi][2], marker="x", label="CTMRG, $\\chi = " + chi + "$, $\\Delta \\xi = 10^{" + "{:.0f}".format(np.log10(err[chi])) + "}$")
            #plt.plot(data[chi][0], data[chi][2], marker="x", label="CTMRG, $\\chi = " + chi + "$, $\\Delta \\xi = 10^{" + "{:.2f}".format(np.log10(err[chi])) + "}$")
            #e = np.log10(np.max(data[chi][3]))
            #plt.plot(data[chi][0], data[chi][2], marker="x", label="CTMRG, $\\chi = " + chi + "$, $\\Delta \\langle \\sigma_z \\rangle = 10^{" + "{:.1f}".format(e) + "}$")
            #plt.plot(data[chi][0], data[chi][2], marker="x", label="CTMRG, $\\chi = " + chi + "$, $\\Delta \\langle \\sigma_z \\rangle = 10^{" + "{:.0f}".format(np.log10(err[chi])) + "}$")
            #plt.errorbar(data[chi][0], data[chi][2], marker="x", label="CTMRG, $\\chi = " + chi + "$, $\\Delta \\langle \\sigma_z \\rangle = 10^{" + "{:.0f}".format(np.log10(err[chi])) + "}$", yerr=[data[chi][2]-np.maximum(1e-10,data[chi][4]),data[chi][4]])
    plt.legend(loc=3)
    plt.xlabel("$-h$")
    plt.ylabel("$\\langle \\sigma_z \\rangle_{0,\\infty}$")
    plt.xlim(0.92, 1.06)
    #plt.ylim(1e-10,1)
    plt.yscale("log")
    plt.grid(True)
    plt.savefig("plots/h_sigmaz_cmpvalentin.png", dpi=300)
    plt.close()

