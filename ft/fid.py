# simple free induction decay code
# References
"""
Keeler pg.49 for general signal form
proton gyromagnetic ratio: https://physics.nist.gov/cgi-bin/cuu/Value?gammap
"""
import numpy as np

class fid():
    """
    Free induction decay class.

    This class will create a FID signal with only one adjusted signal.

    Attributes
    ----------
    B: float
        External magnetic field
    timeunit: str
        Unit for time variable, t
    shift: float
        Chemical shift
    T2: float
        Transverse relaxation time
    nsp: int
        Number of samples in one period
    t_cut: float
        The sampling duration
    gamma: float
        Gyromagnetic ratio
    w: float
        Adjusted signal frequency according to chemical shift
        (It is an observed frequency minus the reference frequency)
    t: list[float]
        A list of times at which the signal is sampled (~6*T2)
    dt: float
        Sampling interval, also called timestep
    f_s: float
        Sampling frequency
    signal: list[float]
        FID signal

    Methods
    -------
    sftq()
        Returns an adjusted signal frequency
    time()
        Returns a list of sampling times and sampling rate f_s
    sgnl()
        Returns FID signal
    call()
        Returns the sampled FID signal

    """
    def __init__(
            self,
            B=1.5,
            timeunit='msec',
            shift=0,
            T2=2000,
            nsp=40,
            t_cut=12000):
        """ 
        Constructor

        """
        self.B = B # Approximately 2000 msec is T2 for water/CSF at 1.5T
        self.timeunit = timeunit
        self.shift = shift
        self.T2 = T2
        self.nsp = nsp
        self.t_cut = t_cut
        self.gamma = 267.52218744*pow(10,6)
        self.w = self.sfrq()
        self.t, self.dt, self.f_s = self.time()
        self.signal = self.sgnl()

    def sfrq(self):
        """
        Signal frequency adjusted according to chemical shift

        Returns
        -------
        w: float
            Signal frequency

        Raises
        ------
        ValueError
            If incorrect timeunit is specified (it is either msec or micron).
        """
        if self.timeunit == 'msec':
            return pow(10, -9)*self.shift*self.B*self.gamma
        elif self.timeunit == 'micron':
            return pow(10, -12)*self.shift*self.B*self.gamma
        else:
            raise ValueError('Incorrect time unit is specified: use msec or micron')

    def time(self):
        """
        A list of times at which the signal is sampled with sampling interval and rate.

        Returns
        -------
        t: list[float]
            Sampled times
        dt: float
            Sampling interval, also called timestep
        f_s: float
            Sampling rate
        """
        if self.shift == 0: # 1024 samples total when there is no oscillation
            dt = self.t_cut/1023
            return np.arange(0, 1024)*dt, dt, 1/dt
        else:
            f_s = 0.5*self.nsp*self.w/np.pi
            dt = 1/f_s
            p = int(np.log2(self.t_cut*f_s + 1)) + 1
            ns = pow(2, p) # total number of samples including t = 0 
            return np.arange(0, ns)*dt, dt, f_s

    def sgnl(self):
        """
        Sampled FID signal

        Returns
        -------
        signal: list[float]
            FID signal
        """
        return np.exp(1j*self.w*self.t)*np.exp(-self.t/self.T2)

    def __repr__(self):
        return "fid() [check the attributes if you wish to change the default variables]"

    def __call__(self):
        """ returns signal """ 
        return self.signal
