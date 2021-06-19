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
    total_t: float
        The sampling duration
    gamma: float
        Gyromagnetic ratio
    w_ref: float
        Reference angular frequency (lowest frequency)
    w: float
        Adjusted signal frequency according to chemical shift
        It is an observed frequency minus the reference frequency)
    t: list[float]
        A list of times at which the signal is sampled (~6*T2)
    signal: list[float]
        FID signal

    Methods
    -------
    sftq()
        Returns an adjusted signal frequency
    time()
        Returns a list of sampling times and sampling rate SR
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
            total_t=12000):
        """ 
        Constructor

        """
        self.B = B # Approximately 2000 msec is T2 for water/CSF at 1.5T
        self.timeunit = timeunit
        self.shift = shift
        self.T2 = T2
        self.nsp = nsp
        self.total_t = total_t
        self.gamma = 267.52218744*pow(10,6)
        self.w_ref = self.B*self.gamma
        self.w = self.sfrq()
        self.t, self.dt, self.SR = self.time()
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
            return pow(10, -9)*self.shift*self.w_ref
        elif self.timeunit == 'micron':
            return pow(10, -12)*self.shift*self.w_ref
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
        SR: float
            Sampling rate
        """
        if self.shift == 0: # 1001 samples total when there is no oscillation
            dt = self.total_t/1000
            return np.linspace(0, self.total_t, 1001), dt, 1/dt
        else:
            SR = 0.5*self.nsp*self.w/np.pi
            ns = int(self.total_t*SR) + 1 # total number of samples including t = 0 
            return np.linspace(0, (ns - 1)/SR, ns), 1/SR, SR

    def sgnl(self):
        """
        Sampled FID signal

        Returns
        -------
        signal: list[float]
            FID signal
        """
        return np.exp(-self.t/self.T2)*(np.cos(self.w*self.t) + 1j*np.sin(self.w*self.t))

    def __repr__(self):
        return "fid() [check its attributes if you wish to change the default variables]"

    def __call__(self):
        """ returns signal """ 
        return self.signal
