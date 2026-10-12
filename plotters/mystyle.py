import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
dt = 0.01

# plot style
# plt.style.use("./plotters/mystyle.mpltstyle")
style = 'seaborn-whitegrid'
mpl.style.use(style.lower())
mpl.rcParams.update({'font.size': 22})
mpl.rcParams['figure.figsize'] = (16, 9)
mpl.rcParams["axes.edgecolor"] = "0.15"
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams["axes.linewidth"] = 1.3
mpl.rcParams['axes.labelpad'] = 6
mpl.rcParams["axes.xmargin"] = 0
mpl.rcParams["axes.ymargin"] = 0.1
mpl.rcParams["axes.labelpad"] = 4.0
# mpl.rcParams['figure.titlesize'] = 20
# mpl.rcParams['axes.titlesize'] = 20
# mpl.rcParams['axes.labelsize'] = 20    # fontsize of the x and y labels
# mpl.rcParams['legend.fontsize'] = 20    # fontsize of the x and y labels

# mpl.rcParams["figure.autolayout"] = True
# grid
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = "C1C1C1"
mpl.rcParams["grid.linestyle"] = ":"
# plt.rcParams["xtick.major.size"] = 8
# plt.rcParams["xtick.major.width"] = 1.5
# plt.rcParams["xtick.direction"] = 'in'
# plt.rcParams["xtick.top"] = True
# print(plt.rcParams.keys())

# legend
mpl.rcParams['legend.frameon']  = True
mpl.rcParams["legend.framealpha"] = 1.0
mpl.rcParams["legend.edgecolor"] = "k"
mpl.rcParams["legend.fancybox"] = False
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True

# saving
mpl.rcParams["savefig.pad_inches"] = 0.003
mpl.rcParams['savefig.format']= 'png'
mpl.rcParams['savefig.dpi']= 300
mpl.rcParams["savefig.bbox"] = "tight"


# colours
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = sns.color_palette("Paired", 10)[::-1]
c_base       = "#0C2340"    # delft dark blue
color_serl50 = '#009B77'
color_serl10 = '#FFB81C' #'#EC6842' #colors[2]
color_td3 =  '#6F1D77' #colors[0]
c_ref = '#0C2340'
c_state =  '#A50034' #colors[4]
c_rate = colors[5]
c_command = '#0076C2'
c_alpha =  '#6CC24A' #colors[6]

# lines
linestyle_str = [
    ('solid', 'solid'),      # Same as (0, ()) or '-'
    ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
    ('dashed', 'dashed'),    # Same as style_ref
    ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
    ('loosely dotted',        (0, (1, 10))),
    ('dotted',                (0, (1, 1))),
    ('densely dotted',        (0, (1, 1))),
    ('long dash with offset', (5, (10, 3))),
    ('loosely dashed',        (0, (5, 10))),
    ('dashed',                (0, (5, 5))),
    ('densely dashed',        (0, (5, 1))),

    ('loosely dashdotted',    (0, (3, 10, 1, 10))),
    ('dashdotted',            (0, (3, 5, 1, 5))),
    ('densely dashdotted',    (0, (3, 1, 1, 1))),

    ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
style_ref = 'dashed'
constraint_style = (0, (3, 1, 1, 1, 1, 1))
