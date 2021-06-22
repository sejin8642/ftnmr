# simple free induction decay code
# References
"""
Keeler pg.49 for general signal form
proton gyromagnetic ratio: https://physics.nist.gov/cgi-bin/cuu/Value?gammap
"""
import numpy as np

class fid():
    gamma = 267.52218744*pow(10,6)
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
    dt: float
        Sampling interval, also called timestep
    nsp: int
        Number of samples in one period
    t_cut: float
        The sampling duration
    gamma: float
        Gyromagnetic ratio
    f0: float
        Adjusted signal frequency according to chemical shift
        (It is an observed frequency minus the reference frequency)
    w: float
        signal angular frequency
    t: list[float]
        A list of times at which the signal is sampled (~6*T2)
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
    gamma = 267.52218744*pow(10,6)

    def __init__(
            self,
            B=1.5,
            timeunit='msec',
            shift=0.5,
            T2=2000,
            dt=40,
            t_cut=12000):
        """ 
        Constructor

        """
        self.B = B # Approximately 2000 msec is T2 for water/CSF at 1.5T
        self.timeunit = timeunit
        self.shift = shift
        self.T2 = T2
        self.dt = dt
        self.f0, self.nsp = self.sfrq(B, timeunit, dt, shift)
        self.w = 2*np.pi*self.f0
        self.f_s, self.ns, self.t = self.time(dt, t_cut)
        self.f = np.arange(0, self.ns)*self.f_s/self.ns
        self.signal = self.sgnl()
    
    @classmethod
    def sfrq(cls, B=1.5, timeunit='msec', dt=40, shift=0.5):
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
            f0 = 0.5*pow(10, -9)*shift*B*cls.gamma/np.pi
            nsp = 2*np.pi/(f0*dt)
            return f0, nsp
        elif timeunit == 'micron':
            f0 = 0.5*pow(10, -12)*shift*B*cls.gamma/np.pi
            nsp = 2*np.pi/(f0*dt)
            return f0, nsp
        else:
            raise ValueError('Incorrect time unit is specified: use msec or micron')

    @staticmethod
    def time(dt, t_cut):
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
        f_s = 1/dt
        p = int(np.log2(t_cut*f_s + 1)) + 1
        ns = pow(2, p) # total number of samples including t = 0 
        return f_s, ns, np.arange(0, ns)*dt,

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
