import numpy as np
import matplotlib.pyplot as plt
import scipy

# Math graph
def graph(x, y, xlabel=r'$x$', ylabel=r'$y$', save=False, filename='figure.eps'):
    # LaTeX font with size 9
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": 'serif',
        "font.size": 9})
    
    # plots y vs. x in black line with linesize 2 with the given axes
    fig = plt.figure(figsize=(6,4), dpi=500)
    ax = fig.add_subplot(111)
    
    # minimums and maximums of x and y 
    xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
    
    # reset minimum and maximum of y if y-range does not contain 0
    if 0 < ymin: ymin = -0.1*ymax
    if ymax < 0: ymax = -0.1*ymin
    
    # configures plot axes, labels and their positions with arrow axis tips
    if (xmin <= 0) and (0 <= xmax):
        ax.spines['left'].set_position(('data', 0)) 
        ax.set_ylabel(ylabel, rotation=0)
        ax.yaxis.set_label_coords(-xmin/(xmax - xmin), 1.02)
        ax.plot(0, 1, "^k", markersize=3, transform=ax.get_xaxis_transform(), clip_on=False)
    else:
        ax.spines['left'].set_visible(False)
        ax.set_ylabel(ylabel).set_visible(False)
    
    ax.spines['bottom'].set_position(('data', 0)) 
    ax.set_xlabel(xlabel)
    ax.xaxis.set_label_coords(1.02, -ymin/(ymax - ymin) + 0.02)
    ax.plot(1, 0, ">k", markersize=3, transform=ax.get_yaxis_transform(), clip_on=False)

    # plots y vs. x in black line with linesize 2 with the given axes
    plt.plot(x, y, 'k-', linewidth=.5)
    plt.axis([xmin, xmax, 1.1*ymin, 1.1*ymax])

    # change the spine linewidth
    plt.rcParams['axes.linewidth'] = 0.2

    # deletes top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # changes the size of ticks (both major and minor) to zero if ticks==False
    ax.tick_params(axis=u'both', which=u'both', length=0)

    # no tick labels
    plt.xticks([])
    plt.yticks([])

    # save the figure as eps vector image if save==True
    if (save == True): 
        plt.savefig(filename, format='eps', transparent=True)
    
    # show the plot
    plt.show()

