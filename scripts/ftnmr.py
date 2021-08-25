# References
"""
Keeler pg.49 for general signal form
proton gyromagnetic ratio: https://physics.nist.gov/cgi-bin/cuu/Value?gammap
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import binom
from itertools import product

# Larmor angular frequency function
def larmor(B=1.5, unit='MHz'):
    """ Returns Larmor angular frequency based on external B field 

    Parameters
    ----------
    B: float
        Magnetic field in unit of Tesla
    unit: str
        Unit for ordinary frequency (default MHz)

    Returns
    -------
        Larmor angular frequency in either MHz or kHz
    """
    if unit=='MHz':
        return 267.52218744*B
    elif unit=='kHz':
        return 267.52218744*pow(10,3)*B
    else:
        raise ValueError("Frequency unit must be either MHz or kHz")

# molecule class 
class molecule(): 
    """
    Molecule class 

    This class contains hydrogen groups with the number and chemical shifts of each group.
    Based on J-coupling constants, spectral splits with their distribution is also created

    Attributes
    ----------
    hydrogens: dict[str]: (int, float)
        Dictionary of hydrogen groups with a number and a chemical shifts of the members
    couplings: list[(str, str, float)]
         List of J-couplings between two hydrogen groups. The last element of the tuple
         is the J-coupling, and the unit for it is Hz
    splits: 
    """

    # molecule constructor
    def __init__(
            self,
            hydrogens={'a':(1, 10.0)},
            couplings=[]):
        """ molecule constructor

        Parameters
        ----------
        hydrogens: dict[str]: (int, float)
            Dictionary of hydrogen groups with a number and a chemical shifts of the members
            (default a:(1, 10.0))
        couplings: list[(str, str, float)]
            List of J-couplings between two hydrogen groups. The last element of the tuple 
            is the J-coupling, and the unit for it is Hz (default None)
        splits: dict[str]: (array, array)
            Spectral splits for each hydrogen group and their probability distribution within each split
        """ 
        A = list(dict.fromkeys([k for b in couplings for k in b[:-1]]))
        B = {k:[ (b[2], hydrogens[ b[ 1-b.index(k) ] ][0]) for b in couplings if k in b] for k in A}

        C0 = {k:[ [n*b/2 for n in range(-d, d+1, 2) ] for b, d in B[k]] for k in A}
        C1 = {k:[ [binom(d, x)/pow(2,d) for x in range(0, d+1)] for b, d in B[k] ] for k in A}
        D0 = {k:[ sum(i) for i in product(*C0[k]) ] for k in A}
        D1 = {k:[ np.prod(i) for i in product(*C1[k]) ] for k in A}
        E0 = {k:[ D0[k][n] for n in np.argsort(D0[k]) ] for k in A}
        E1 = {k:[ D1[k][n] for n in np.argsort(D0[k]) ] for k in A}

        ind = lambda k: filter(lambda i: not np.isclose(E0[k][i-1], E0[k][i]), range(0, len(E0[k])))
        F0 = {k:[E0[k][i] for i in ind(k)] for k in A}
        F1 = {k:[E1[k][i] for i in ind(k)] for k in A}

        dup = lambda k: filter(lambda i: np.isclose(E0[k][i-1], E0[k][i]), range(0, len(E0[k])))
        for k in A:
            n = 0
            for i in dup(k):
                F1[k][i-1-n] += E1[k][i]
                n += 1
        
        # molecule constructor attributes
        self.hydrogens = hydrogens
        self.couplings = couplings
        self.splits = {k:(np.array(F0[k]), np.array(F1[k])) for k in A}

# sample class
class sample():
    """
    NMR sample class

    This class creates an NMR sample with molecules and solvent

    Attributes
    ----------
    molecules: dict[str]: (molecule, float)
        A dictionary of molecules with their relative abundance
    T2: float
        T2 of the sample
    r: float
        Relaxivity of the sample (1/T2)
    timeunit: str
        Unit for T2
    """

    # sample constructor
    def __init__(self, molecules, T2=100.0, timeunit='msec'):
        """ 
        sample constructor

        Parameters
        ----------
        molecules: dict[str]: (molecule, float)
            A dictionary of molecules with their relative abundance
        T2: float
            T2 of the sample
        timeunit: str
            Unit for T2
        """

        # sample constructor attributes
        self.molecules = molecules 
        self.T2 = T2
        self.r = 1/T2
        self.timeunit = timeunit

# spectrometer class
class spectrometer():
    """
    Spectrometer class

    Attributes
    ----------
    B: float
        External magnetic field
    timeunit: str
        Unit for time variable, t
    shift_maximum: float
        Maximum chemical shift the spectrometer can observer
    dt: float
        Sampling interval, also called timestep
    t_cut: float
        The minimum sampling duration
    gamma: float
        Gyromagnetic ratio
    f_s: float
        Sampling frequency
    f_l: float
        Ordinary Larmor frequency
    f_unit: str
        Unit for ordinary frequency used
    ns: integer
        Total number of samples
    p: integer
        Power of two that yields the number of samples
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

    # spectrometer class attributes
    gamma = 267.52218744*pow(10,6)

    # spectrometer constructor
    def __init__(
            self,
            B=10.0,
            timeunit='msec',
            shift_maximum=128.0,
            t_cut=600):
        """ spectrometer constructor

        Parameters
        ----------
        B: float
            External magnetic field (default 1.5 Tesla) 
        timeunit: str
            Unit string for time variable. It is either msec or micron (default msec)
        shift_maximum: float
            Maximum chemical shift to set the maximum frequency (default 128.0 ppm)
        t_cut: float
            Cutoff time that the maximum t value must exceed (default 12000.0)
        """

        # spectrometer constructor attributes
        self.B = B # Approximately 2000 msec is T2 for water/CSF at 1.5T
        self.timeunit = timeunit
        self.shift_maximum = shift_maximum
        self.f_unit, self.ep = self.unit(timeunit)
        self.f_s = 0.5*shift_maximum*self.gamma*B*pow(10, self.ep-6)/np.pi
        self.f_l = 0.5*self.gamma*B*pow(10, self.ep)/np.pi
        self.dt = 1/self.f_s
        self.ns, self.p, self.t = self.time(self.f_s, t_cut)

    # spectrometer unit method
    def unit(self, timeunit):
        """ Unit method

        Returns frequency unit and time exponent that turns seconds into miliseconds or microseconds
        
        Parameters
        ----------
        timeunit: str
            Unit for time. It is either msec or micron
        
        Returns
        -------
        f_unit: str
            Unit for frequency
        exp: int
            Exponent of 10 to convert seconds into miliseconds or microseconds
        """

        if timeunit == 'msec':
            return 'kHz', -3
        elif timeunit == 'micron':
            return 'MHz', -6
        else:
            raise ValueError('incorrect time unit is specified: use msec or micron')
        
    # spectrometer time method
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

    # spectrometer calibrate method
    def calibrate(
            self,
            B=10.0,
            timeunit='msec',
            shift_maximum=128.0,
            t_cut=600):
        """
        Spectrometer calibrate method

        This method will calibrate spectrometer settings to default if no inputs were provided.
        The parameters for this method is the same as the constructor
        """

        # spectrometer calibrate attributes
        self.B = B # Approximately 2000 msec is T2 for water/CSF at 1.5T
        self.timeunit = timeunit
        self.shift_maximum = shift_maximum
        self.f_unit, self.ep = self.unit(timeunit)
        self.f_s = 0.5*shift_maximum*self.gamma*B*pow(10, self.ep-6)/np.pi
        self.f_l = 0.5*self.gamma*B*pow(10, self.ep)/np.pi
        self.dt = 1/self.f_s
        self.ns, self.p, self.t = self.time(self.f_s, t_cut)

    # spectrometer measure method
    def measure(self, sample):
        """" 
        Measures FID signal from the sample

        """
        if sample.timeunit != self.timeunit:
            raise ValueError("Sample time unit does not match with spectrometer time unit")

        # spectrometer measure attributes
        self.w = pow(10, self.ep-6)*shift*self.gamma*self.B
        self.f0 = 0.5*self.w/np.pi
        self.nsp = 1/(self.f0*self.dt)
        self.signal = self.signal_output()

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

# fid class (free induction decay for a single proton)
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
            B=10.0,
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
        self.nsp = 1/(self.f0*self.dt)
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

# Lorentzian class
class lorentzian():
    """
    Absorption Lorentzian class

    This class will create an absorption Lorentzian profile with frequency domain

    Attributes
    ----------
    f: list[float]
    unit: str
        Unit string for ordinary frequency (default kHz)
    ns: integer
        Total number of frequencies (default pow(2, 15))
    r: float
        Relaxivity
    f0: float
        Ordinary Larmor frequency
    lorentz: list[float]
        Lorentzian function output

    Methods
    -------
    lorz()
        Returns the lorentz attribute
    """

    # Lorentzian constructor
    def __init__(
            self,
            unit='kHz',
            ns=pow(2,15),
            f_max=55.0,
            r=0.01,
            f0=4.25,
            f_l=425000,
            ob=object()):
        """
        Constructor

        Parameters
        ----------
        unit: str
            Unit string for ordinary frequency (default kHz)
        ns: integer
            Total number of frequencies (default pow(2, 15))
        f_max: float
            Maximum frequency for f-domain (default 55.0)
        r: float
            Relaxivity (default 0.01)
        f0: float
            Adjusted detected frequency (default 4.25)
        f_l: float
            Ordinary Larmor frequency (default 425000)
        ob: fid class
            fid object from which to extract its attributes (default object())
        """

        # Lorentzian object attributes
        if isinstance(ob, fid):
            self.unit = ob.frequency_unit
            self.ns = ob.ns
            self.p = ob.p
            self.f = np.arange(0, ob.ns)*ob.f_s/ob.ns # the last f excludes fmax
            self.cs = pow(10, 6)*self.f/ob.f_l
            self.r = ob.r
            self.f0 = ob.f0
            self.lorentz = self.lorz()
        else:
            self.unit = unit
            self.ns = ns
            self.p = np.log2(ns)
            self.f = np.arange(0, ns)*f_max/ns # the last f excludes f_max
            self.cs = pow(10, 6)*self.f/f_l
            self.r = r
            self.f0 = f0
            self.lorentz = self.lorz()

    # lorz method of Lorentzian
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

#
# Baseline 

