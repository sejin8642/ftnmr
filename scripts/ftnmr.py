# References
"""
Keeler pg.49 for general signal form
proton gyromagnetic ratio: https://physics.nist.gov/cgi-bin/cuu/Value?gammap
"""

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
        ax.yaxis.set_label_coords(-xmin/(xmax - xmin), 1.02)
        ax.set_ylabel(ylabel, rotation=0)
        ax.plot(0, 1, "^k", markersize=3, transform=ax.get_xaxis_transform(), clip_on=False)
    else:
        ax.spines['left'].set_visible(False)
        ax.set_ylabel(ylabel).set_visible(False)
    
    ax.spines['bottom'].set_position(('data', 0)) 
    ax.xaxis.set_label_coords(1.02, -ymin/(ymax - ymin) + 0.02)
    ax.set_xlabel(xlabel)
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

# Larmor angular frequency
def larmor(B=1.5, unit='MHz'):
    """ Returns Larmor angular frequency based on external B field """
    if unit=='MHz':
        return 267.52218744*B
    elif unit=='kHz':
        return 267.52218744*pow(10,3)*B
    else:
        raise ValueError("Frequency unit must be either MHz or KHz")

# free induction decay
class fid():
    """
    Free induction decay class.

    This class will create a FID signal with only one adjusted signal. All the class variables are in SI unit, but time for instance variable is either msec or micron.

    Attributes
    ----------
    B: float
        External magnetic field
    timeunit: str
        Unit for time variable, t
    shift: float
        Chemical shift
    shift_maximum: float
        Maximum chemical shift the spectrometer can observer
    T2: float
        Transverse relaxation time
    r: float
        Relaxivity for T2
    dt: float
        Sampling interval, also called timestep
    nsp: float
        Number of samples in one period
    t_cut: float
        The minimum sampling duration
    gamma: float
        Gyromagnetic ratio
    f0: float
        Adjusted signal frequency according to chemical shift
        (It is a detected frequency minus the reference frequency)
    w: float
        Adjusted signal angular frequency
    f_s: float
        Sampling frequency
    ns: integer
        Total number of samples
    t: list[float]
        A list of times at which the signal is sampled (default ~6*T2)
    signal: list[float]
        FID signal

    Methods
    -------
    sampling_frequency()
        Returns a sampling frequency
    signal_frequency()
        Returns an adjusted signal frequency
    time()
        Returns a list of sampling times and sampling rate f_s
    signal_output()
        Returns FID signal
    call()
        Returns the sampled FID signal

    """
    gamma = 267.52218744*pow(10,6)

    def __init__(
            self,
            B=1.5,
            timeunit='msec',
            shift=5.0,
            shift_maximum=100.0,
            T2=2000,
            t_cut=12000):
        """ 
        Constructor

        Parameters
        ----------
        B: float
            External magnetic field (default 1.5 Tesla) 
        timeunit: str
            Unit string for time variable. It is either msec or micron (default msec)
        shift: float
            Chemical shift (default 0.5)
        shift_maximum: float
            Maximum chemical shift to set the maximum frequency (default 10.0)
        T2: float
            Relaxation constant (default 2000.0)
        t_cut: float
            Cutoff time that the maximum t valule must exceed (default 12000.0)
        """

        # Constructor parameters
        self.B = B # Approximately 2000 msec is T2 for water/CSF at 1.5T
        self.timeunit = timeunit
        self.shift = shift
        self.shift_maximum = shift_maximum
        self.T2 = T2
        self.r = 1/T2
        self.f_s, self.frequency_unit = self.sampling_frequency(shift_maximum, B, timeunit)
        self.dt = 1/self.f_s
        self.ns, self.t = self.time(self.f_s, t_cut)
        self.f0 = self.signal_frequency(B, timeunit, shift)
        self.nsp = 1/(self.f0*self.dt)
        self.w = 2*np.pi*self.f0
        self.signal = self.signal_output()

    @classmethod
    def sampling_frequency(cls, shift_maximum, B, timeunit):
        """
        Returns sampling frequency based on the external B field and maximum chemical shift

        Parameters
        ----------
        shift_maximum: float
            Maximum chemical shift the spectrometer can observe
        B: float
            External magnetic field
        timeunit: str
            unit for time variable (either msec or micron)

        Returns
        -------
        f_s: float
            Sampling frequency or the maximum frequency of the spectrometer
        """
        if timeunit == 'msec':
            return 0.5*shift_maximum*cls.gamma*B*pow(10, -9)/np.pi, 'kHz'
        elif timeunit == 'micron':
            return 0.5*shift_maximum*cls.gamma*B*pow(10, -12)/np.pi, 'MHz'
        else:
            raise ValueError('Incorrect time unit is specified: use msec or micron')

    @classmethod
    def signal_frequency(cls, B, timeunit, shift):
        """
        Signal frequency adjusted according to chemical shift

        Parameters
        ----------
        B: float
            External magnetic field
        timeunit: str
            Unit for time variable, t
        shift: float
            Chemical shift

        Returns
        -------
        f0: float
            Signal frequency

        Raises
        ------
        ValueError
            If incorrect timeunit is specified (it is either msec or micron).
        """
        if timeunit == 'msec':
            return 0.5*pow(10, -9)*shift*B*cls.gamma/np.pi
        elif timeunit == 'micron':
            return 0.5*pow(10, -12)*shift*B*cls.gamma/np.pi
        else:
            raise ValueError('Incorrect time unit is specified: use msec or micron')

    @staticmethod
    def time(f_s, t_cut):
        """
        A list of times at which the signal is sampled with sampling interval and rate.

        Returns
        -------
        t: list[float]
            Sampled times
        f_s: float
            Sampling rate
        """
        p = int(np.log2(t_cut*f_s + 1)) + 1
        ns = pow(2, p) # total number of samples including t = 0 
        return ns, np.arange(0, ns)/f_s

    def signal_output(self):
        """
        Sampled FID signal

        Returns
        -------
        signal: list[float]
            FID signal
        """
        return np.exp(1j*self.w*self.t)*np.exp(-self.r*self.t)

    def __repr__(self):
        return "fid() [check the attributes if you wish to change the default variables]"

    def __call__(self):
        """ returns signal """ 
        return self.signal

# Lorentzian class
class lorentzian():
    """
    Absorption Lorentzian class

    This class will create an absorption Lorentzian profile with frequency domain

    Attributes
    ----------
    f: list[float]
        Domain frequencies
    r: float
        Relaxivity
    f0: float
        Larmor frequency
    lorentz: list[float]
        Lorentzian function output

    Methods
    -------
    lorz()
        Returns the lorentz attribute
    """
    def __init__(self, df, ns, r, f0):
        """
        Constructor

        Parameters
        ----------
        df: float
            Frequency interval (Sampling frequency over number of samples)
        ns: integer
            Total number of frequencies
        r: float
            Relaxivity
        f0:
            Frequency shift
        """
        self.f = np.arange(0, ns)*df
        self.r = r
        self.f0 = f0
        self.lorentz = self.lorz()

    def lorz(self):
        """
        Lorentzian function

        Returns
        -------
        lorentz: list[float]
            Lorentzian output
        """
        A = 2*np.pi*(self.f0 - self.f)
        B = pow(self.r, 2) + 4*pow(np.pi, 2)*pow((self.f - self.f0), 2)

        return self.r/B + 1j*A/B

    def __call__(self):
        """ returns lorentz """
        return self.lorentz

# Baseline 
