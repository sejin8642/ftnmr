# fid class (free induction decay for a single proton)
import numpy as np

class fid():
    """
    Free induction decay class.

    This class will create a FID signal with only one adjusted signal. 
    All the class variables are in SI unit, but time for instance variable is either msec or micron.

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
        (It is a real frequency minus the reference frequency)
    w: float
        Adjusted signal angular frequency
    f_s: float
        Sampling frequency
    f_l: float
        Ordinary Larmor frequency
    frequency_unit: str
        Unit for ordinary frequency used
    ns: integer
        Total number of samples
    p: integer
        Power of two that yields the number of samples
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

    # fid class attributes
    gamma = 267.52218744*pow(10,6)

    # fid constructor
    def __init__(
            self,
            B=1.5,
            timeunit='msec',
            shift_maximum=128.0,
            t_cut=600,
            shift=5.0,
            T2=100):
        """ 
        fid constructor

        Parameters
        ----------
        B: float
            External magnetic field (default 1.5 Tesla) 
        timeunit: str
            Unit string for time variable. It is either msec or micron (default msec)
        shift: float
            Chemical shift in units of ppm (default 0.5 ppm)
        shift_maximum: float
            Maximum chemical shift to set the maximum frequency (default 128.0 ppm)
        T2: float
            Relaxation constant (default 100.0)
        t_cut: float
            Cutoff time that the maximum t valule must exceed (default 12000.0)
        """

        # fid constructor attributes
        self.B = B # Approximately 2000 msec is T2 for water/CSF at 1.5T
        self.timeunit = timeunit
        self.shift = shift
        self.shift_maximum = shift_maximum
        self.T2 = T2
        self.r = 1/T2
        self.f_s, self.f_l, self.frequency_unit = self.sampling_frequency(shift_maximum, B, timeunit)
        self.dt = 1/self.f_s
        self.ns, self.p, self.t = self.time(self.f_s, t_cut)
        self.f0 = self.signal_frequency(B, timeunit, shift)
        try:
            self.nsp = 1/(self.f0*self.dt)
        except ZeroDivisionError:
            self.nsp = None
        self.w = 2*np.pi*self.f0
        self.signal = self.signal_output()

    @classmethod
    def sampling_frequency(cls, shift_maximum, b, timeunit):
        """
        returns sampling frequency based on the external b field and maximum chemical shift

        parameters
        ----------
        shift_maximum: float
            maximum chemical shift the spectrometer can observe
        b: float
            external magnetic field
        timeunit: str
            unit for time variable (either msec or micron)

        returns
        -------
        f_s: float
            sampling frequency or the maximum frequency of the spectrometer
        f_l: float
            ordinary larmor frequency
        """
        if timeunit == 'msec':
            f_s = 0.5*shift_maximum*cls.gamma*b*pow(10, -9)/np.pi
            f_l = 0.5*cls.gamma*b*pow(10, -3)/np.pi
            return f_s, f_l, 'khz'
        elif timeunit == 'micron':
            f_s = 0.5*shift_maximum*cls.gamma*b*pow(10, -12)/np.pi,
            f_l = 0.5*cls.gamma*b*pow(10, -6)/np.pi
            return f_s, f_l, 'mhz'
        else:
            raise valueerror('incorrect time unit is specified: use msec or micron')

    # fid signal_frequency method
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

    # fid time method
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
        return ns, p, np.arange(0, ns)/f_s

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
