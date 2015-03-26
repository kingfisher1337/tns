import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from hashlib import md5
from fractions import Fraction

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

for filename in sorted(os.listdir("output")):
    if filename.startswith("h_mz_"):
        h = []
        mz = []
        md = md5()
        with open("output/" + filename, "r") as f:
            chi = filter(lambda s: s.find("chi=") != -1, filename[:-4].split("_"))[0].split("=")[-1]
            svalerr = float(filter(lambda s: s.find("svalerr=") != -1, filename[:-4].split("_"))[0].split("=")[-1])
            q = int(filter(lambda s: s.find("q=") != -1, filename[:-4].split("_"))[0].split("=")[-1])
            for line in f:
                fields = line.split(" ")
                h.append(-float(fields[0]))
                mz.append(map(float, fields[1:1+q]))
                md.update(line)
        
        markers = ["x", "+", "o", "D", "v", "^", "<", ">"]
        colours = ["b", "g", "r", "c", "m", "y", "k", "b"]
        if md.hexdigest() != read_hash("plots/" + filename.split(".dat")[0] + ".md5"):
            print filename
            for j in xrange(q):
                s = "| {:d} \\rangle \\langle {:d} |".format(j,j)
                plt.plot(h, map(lambda x: x[j], mz), markers[j], mfc="none", mec=colours[j], label="$\\langle " + s + " \\rangle_{0,\\infty}$")
            #plt.plot(h, mx, marker="x", label="$\\langle \\sigma_x \\rangle_{0,\\infty}$")
            plt.title("1D-QPotts ground state via 2D-CTMRG ($q = " + str(q) + "$, $\\chi = " + chi + "$, $\\Delta \\xi = 10^{" + "{:.0f}".format(np.log10(svalerr)) + "}$)")
            plt.legend(loc="best")
            plt.xlabel("$-h$")
            #plt.ylim(0,1)
            plt.grid(True)
            plt.savefig("plots/" + filename.split(".dat")[0] + ".png", dpi=300)
            plt.close()
            write_hash("plots/" + filename.split(".dat")[0] + ".md5", md.hexdigest())

primFieldsFM = [
    None, None,
    [Fraction(1,16)],
    [Fraction(1,8),Fraction(13,8)]
]
primFieldsPM = [
    None, None,
    [Fraction(0), Fraction(1,2)],
    [Fraction(0),Fraction(2,3)]
]

def scale_svals_for_fields(f, xi):
    xi = np.sort(-np.log(xi))
    f = sorted(f)[:2]
    f0 = f[0]
    if len(f) == 1:
        f1 = f0 + 1
    else:
        f1 = min([f[1],f0+1])
    xi = xi * float(f1-f0) / (xi[1]-xi[0])
    xi = xi - xi[0] + f0
    return xi

def scale_svals_fm(q, xi):
    return scale_svals_for_fields(primFieldsFM[q], xi)
def scale_svals_pm(q, xi):
    return scale_svals_for_fields(primFieldsPM[q], xi)

def get_yticks_for_fields(fields,ymin,ymax):
    t = []
    for f in fields:
        for j in range(int(np.ceil(ymin-f)), int(np.floor(ymax-h))+1):
            if not float(f+j) in t:
                t.append(float(f+j))
    return t
def get_yticklabels_for_fields(fields,ymin,ymax):
    t = []
    for f in fields:
        for j in range(int(np.ceil(ymin-f)), int(np.floor(ymax-h))+1):
            s = "0" if f == 0 else "\\frac{" + str(f.numerator) + "}{" + str(f.denominator) + "}"
            if j > 0:
                s += "+" + str(j)
            s = "$" + s + "$"
            if not s in t:
                t.append(s)
    return t

def get_fm_yticks(q,ymin,ymax):
    return get_yticks_for_fields(primFieldsFM[q],ymin,ymax)
def get_pm_yticks(q,ymin,ymax):
    return get_yticks_for_fields(primFieldsPM[q],ymin,ymax)
def get_fm_yticklabels(q,ymin,ymax):
    return get_yticklabels_for_fields(primFieldsFM[q],ymin,ymax)
def get_pm_yticklabels(q,ymin,ymax):
    return get_yticklabels_for_fields(primFieldsPM[q],ymin,ymax)

for filename in sorted(os.listdir("output")):
    if filename.startswith("ctmrgsvals_detail_"):
        h_xi = dict()
        md = md5()
        with open("output/" + filename, "r") as f:
            chi = filter(lambda s: s.find("chi=") != -1, filename[:-4].split("_"))[0].split("=")[-1]
            q = int(filter(lambda s: s.find("q=") != -1, filename[:-4].split("_"))[0].split("=")[-1])
            svalerr = float(filter(lambda s: s.find("svalerr=") != -1, filename[:-4].split("_"))[0].split("=")[-1])
            for line in f:
                fields = line.split(" ")
                h = -float(fields[0])
                xi = float(fields[1])
                if not h_xi.has_key(h):
                    h_xi[h] = []
                if xi > 0:
                    h_xi[h].append(xi)
                md.update(line)
        
        #md.update("foo")
        if md.hexdigest() != read_hash("plots/" + filename.split(".dat")[0] + ".md5"):
            print filename
            
            for h in h_xi:
                xi = h_xi[h]
                if h < 1 and len(xi) > 1:
                    xi = scale_svals_fm(q, xi)
                    plt.plot([h]*len(xi), xi, "b+")
            plt.ylabel("$-a \\log(\\xi) + b$")
            ymin,ymax = plt.axes().get_ylim()
            plt.axes().set_yticks(get_fm_yticks(q, ymin, ymax))
            plt.axes().set_yticklabels(get_fm_yticklabels(q, ymin, ymax))
            plt.grid(True)
            plt.title("1D-QPotts ground state via 2D-CTMRG ($q = " + str(q) + "$, $\\chi = " + chi + "$, $\\Delta \\xi = 10^{" + "{:.0f}".format(np.log10(svalerr)) + "}$)")
            plt.xlabel("$-h$")
            s = filename.split(".dat")[0]
            s = s.split("_detail_")
            s = s[0] + "_detail_fm_" + s[1]
            plt.savefig("plots/" + s + ".png", dpi=300)
            plt.close()
            
            for h in h_xi:
                xi = h_xi[h]
                if h > 1 and len(xi) > 1:
                    xi = scale_svals_pm(q, xi)
                    plt.plot([h]*len(xi), xi, "b+")
            plt.ylabel("$-a \\log(\\xi) + b$")
            ymin,ymax = plt.axes().get_ylim()
            plt.axes().set_yticks(get_pm_yticks(q, ymin, ymax))
            plt.axes().set_yticklabels(get_pm_yticklabels(q, ymin, ymax))
            plt.grid(True)
            plt.title("1D-QPotts ground state via 2D-CTMRG ($q = " + str(q) + "$, $\\chi = " + chi + "$, $\\Delta \\xi = 10^{" + "{:.0f}".format(np.log10(svalerr)) + "}$)")
            plt.xlabel("$-h$")
            s = filename.split(".dat")[0]
            s = s.split("_detail_")
            s = s[0] + "_detail_pm_" + s[1]
            plt.savefig("plots/" + s + ".png", dpi=300)
            plt.close()
            
            write_hash("plots/" + filename.split(".dat")[0] + ".md5", md.hexdigest())
                        

"""

hDegeneracyLabel1 = 0.5
hDegeneracyLabel2 = 1.5
primaryField1PM = [ Fraction(0), Fraction(0), Fraction(0),     Fraction(0),    Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0) ]
primaryField2PM = [ Fraction(1), Fraction(1), Fraction(1,2),   Fraction(2,3),  Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(1) ]
primaryField1FM = [ Fraction(0), Fraction(0), Fraction(1,16),  Fraction(1,8),  Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(0) ]
primaryField2FM = [ Fraction(1), Fraction(1), Fraction(17,16), Fraction(13,8), Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(1) ]

yTicks1 = []
yTickLabels1 = []
yTicks2 = []
yTickLabels2 = []
for q in xrange(9):
    f = [primaryField1FM[q], primaryField2FM[q]]
    t = list()
    l = list()
    for k in xrange(len(f)):
        for j in xrange(10):
            x = f[k] + j
            if not float(x) in t:
                t.append(float(x))
                s = "$0" if f[k].numerator == 0 else "$\\frac{" + str(f[k].numerator) + "}{" + str(f[k].denominator) + "}"
                if j > 0:
                    s += "+" + str(j)
                s += "$"
                l.append(s)
    yTicks1.append(t)
    yTickLabels1.append(l)
    
    f = [primaryField1PM[q], primaryField2PM[q]]
    t = list()
    l = list()
    for k in xrange(len(f)):
        for j in xrange(10):
            x = f[k] + j
            if not float(x) in t:
                t.append(float(x))
                s = "$0" if f[k].numerator == 0 else "$\\frac{" + str(f[k].numerator) + "}{" + str(f[k].denominator) + "}"
                if j > 0:
                    s += "+" + str(j)
                s += "$"
                l.append(s)
    yTicks2.append(t)
    yTickLabels2.append(l)


for filename in sorted(os.listdir("output")):
    if filename.startswith("ctmrgsvals_"):
        h_xi = dict()
        md = md5()
        #svalcntPM = dict()
        #svalcntFM = dict()
        with open("output/" + filename, "r") as f:
            chi = filter(lambda s: s.find("chi=") != -1, filename[:-4].split("_"))[0].split("=")[-1]
            q = int(filter(lambda s: s.find("q=") != -1, filename[:-4].split("_"))[0].split("=")[-1])
            svalerr = float(filter(lambda s: s.find("svalerr=") != -1, filename[:-4].split("_"))[0].split("=")[-1])
            for line in f:
                fields = line.split(" ")
                h = -float(fields[0])
                xi = float(fields[1])
                
                if not h_xi.has_key(h):
                    h_xi[h] = []
                if xi > 0:
                    h_xi[h].append(xi)
                
                md.update(line)
                
        md.update(str(np.random.rand(10)))
        
        if md.hexdigest() != read_hash("plots/" + filename.split(".dat")[0] + ".md5"):
            print filename
            
            plt.ylabel("$-a \\log(\\xi) + b$")
            
            for h in h_xi:
                xi = h_xi[h]
                xi = np.sort(-np.log(xi))
                if len(xi) > 1:
                    if h > 1:
                        if primaryField2PM[q] < primaryField1PM[q] + 1:
                            xi = xi * (primaryField2PM[q]-primaryField1PM[q]) / (xi[1]-xi[0])
                        else:
                            xi = xi / (xi[1]-xi[0])
                        xi = xi - xi[0] + primaryField1PM[q]
                    else:
                        if primaryField2FM[q] < primaryField1FM[q] + 1:
                            xi = xi * (primaryField2FM[q]-primaryField1FM[q]) / (xi[1]-xi[0])
                        else:
                            xi = xi / (xi[1]-xi[0])
                        xi = xi - xi[0] + primaryField1FM[q]
                plt.plot([h]*len(xi), xi, "b+")
                
                if h == hDegeneracyLabel1 or h == hDegeneracyLabel2:
                    svalcnt = dict()
                    for x in xi:
                        minDist = 100
                        minDistSval = None
                        for y in svalcnt:
                            if np.abs(x-y) < minDist:
                                minDist = np.abs(x-y)
                                minDistSval = y
                        if minDistSval is None or minDist > 1e-1:
                            svalcnt[x] = 1
                        else:
                            svalcnt[minDistSval] += 1
                    for x in svalcnt:
                        plt.annotate(s="$\\scriptstyle " + str(svalcnt[x]) + "$", xy=(h+0.015, x-0.05))
            
            #plt.axes().set_yticks(yTicks1[q])
            #plt.axes().set_yticklabels(yTickLabels1[q])
            #plt.grid(True)
            
            plt.axes().set_yticks(yTicks1[q] + yTicks2[q])
            plt.axes().set_yticklabels(yTickLabels1[q] + [""]*len(yTicks2[q]))
            plt.ylim(-0.5,5.5)
            #ylim = plt.axes().get_ylim()
            plt.grid(True)
            
            #ax2 = plt.twinx()
            #ax2.set_ylim(ylim)
            #ax2.set_ylim(-0.5,8)
            #ax2.set_yticks(yTicks2[q])
            #ax2.set_yticklabels(yTickLabels2[q])
            #ax2.grid(True)
            
            plt.title("1D-QPotts ground state via 2D-CTMRG ($q = " + str(q) + "$, $\\chi = " + chi + "$, $\\Delta \\xi = 10^{" + "{:.0f}".format(np.log10(svalerr)) + "}$)")
            plt.xlabel("$-h$")
            plt.savefig("plots/" + filename.split(".dat")[0] + ".png", dpi=300)
            plt.close()
            write_hash("plots/" + filename.split(".dat")[0] + ".md5", md.hexdigest())
            
            if q == 4:
                exit()
"""
