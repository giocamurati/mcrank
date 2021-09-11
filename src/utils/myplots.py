import matplotlib  # before importing pyplot

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}' 
matplotlib.rcParams['axes.grid'] = True

import math

from cycler import cycler
from matplotlib import pyplot as plt

SMALL_SIZE = 16 #14
MEDIUM_SIZE = 20 #16
BIGGER_SIZE = 24 #18


plt.style.use('seaborn-paper')
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=SMALL_SIZE) 
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#plt.rc('figure', figsize=(8,4.5))  # 16/9 aspect ratio

# line_styles = "- -- -. :".split()
        
# cycler1 = plt.rcParams['axes.prop_cycle']
# cycler2 = cycler('linestyle', line_styles)
# a, b = len(cycler1), len(cycler2)
# lcm = a * b // math.gcd(a,b)
# cycler1 = cycler1 * (lcm // a)
# cycler2 = cycler2 * (lcm // b)
# plt.rc('axes', prop_cycle=(cycler1 + cycler2))
# plt.rc('lines', linewidth=2)
# plt.rc('axes.spines', left=False, right=False, top=False, bottom=False)
        
