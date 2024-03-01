# References
"""
Keeler pg.49 for general signal form
proton gyromagnetic ratio: https://physics.nist.gov/cgi-bin/cuu/Value?gammap
"""

from itertools import product
import fid
from pathlib import Path
import h5py

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import binom
from scipy.stats import truncnorm
from scipy import interpolate

import tensorflow as tf

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

    This class contains hydrogen groups with the total number and chemical shift of each group.
    Based on J-coupling constants, spectral splits with their distribution is also created

    Attributes
    ----------
    hydrogens: dict[str]: (int, float, float)
        Dictionary of hydrogen groups with the total number, chemical shifts, T2 of 
        the members for each group (default TMS a:(12, 0.0, 150.0)). The default unit for
        T2 is msec
    couplings: list[(str, str, float)]
         List of J-couplings between two hydrogen groups. The last element of the tuple
         is the J-coupling, and the unit for it is Hz
    splits: dict[str]: (array, array)
        Spectral splits for each hydrogen group and their probability distribution within each split
    """

    # molecule constructor
    def __init__(
            self,
            hydrogens={'a':(12, 0.0, 150.0)},
            couplings=[]):
        """ molecule constructor

        Parameters
        ----------
        hydrogens: dict[str]: (int, float, float)
            Dictionary of hydrogen groups with the total number, chemical shifts, T2 of 
            the members for each group (default TMS a:(12, 0.0, 150.0)). The default unit for
            T2 is msec
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

# spectrometer Class
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
    p_l: int
        Power of two to narrow the frequency range of the processed signal
    shift_cutoff: float
        Highest shift of the frequency domain for the FFT
    f_unit: str
        Unit for ordinary frequency used
    ep: int
        Exponent of 10 to convert seconds into miliseconds or microseconds
    w_l: float
        angular Larmor frequency
    f_l: float
        regular Larmor frequency (w_l/2pi)
    f_s: float
        Sampling frequency for frequency-adjusted signal
    dt: float
        Sampling interval, also called timestep
    gamma: float
        Gyromagnetic ratio
    ns: integer
        Total number of collected signal samples (different from the number of FFT output)
    p: integer
        Power of two that yields the number of FFT-processed data points. 2^p is the number of
        data to DFT 
    t: numpy array[float]
        Times at which signal is sampled, sampling duration from zero to maximum t
    df: float
        Frequency resolution of FFT in <f_unit> unit
    nf: int
        Total number of FFT-processed signal output of the spectrometer. This number is different 
        from 2^p which is the total number of FFT input data. 2^p number of data is FFT-processed,
        and the output of FFT has the same size of 2^p. But the spectrometer will truncate the data
        if nf is smaller than 2^p
    w_max: float
        Maximum angular frequency you can observe on hydrogen in your final processed signal
    f: numpy array[float]
        Frequency domain for FFT output in <f_unit> unit. Even though the frequency range is much
        smaller than regular Larmor frequency, detected frequency falls within f as the detected
        frequency is chemically shifted regular Larmor frequency minus some reference frequency
    shift: numpy array[float]
        Chemical shift doamin for FFT output
    hr: float
        One over number of hydrogens of the reference molecule, usually TMS which has 12 hydrogens
    r: float
        Relaxivity for reference hydrogen
    std: float
        Standard deviation of signal noise
    dtype: str
        Data type for shift and spectra
    noise: numpy array[complex float]
        Signal noise
    ps: numpy array[float]
        Zero and first order phase parameters for phast shift (default 0 for both)
    smooth: int
        0 if target signal is noiseless, and 1 if target signal also has the noise
        The noise is the same as raw signal noise
    extra_target: bool
        If true, two target spectra will be returned instead of one (default False). The extra
        target has different noise than the first one, and it is used to compare performance
        between corrected spectrum and the different target spectrum
    target_signal: numpy array[complex float]
        NMR signal without any artifacts (it could be noiseless if smoothness == True)
    splits: list[tuple(float, float, float)]
        List of relative angular Larmor frequencies(detected angular Larmor frequency minus reference
        angular Larmor frequency),  relative abundances, and relaxivities for sample molecules. 
        splits is created once sampe molecules are measured
    signal: numpy array[compelx float]
        NMR sample signal
    target_signal: numpy array[compelx float]
        NMR sample signal without artifacts (it could be noiseless as well)
    FFT: numpy array[complex float]
        FFT output of the signal
    target_FFT: numpy array[complex float]
        FFT output of the target signal 
    spectra_artifact: numpy array[float]
        Spectra artifact 
    target: numpy array[float]
        NMR spectra target output without the artifact (real part of target_FFT)
    target2: numpy array[float]
        Second NMR spectra target output without the artifact (real part of target_FFT). This signal
        is useful for comparing performance between corrected NMR spectrum and second NMR spectrum.
        The second NMR spectra will have different noise than the first one (although they are the
        same type of noise with the same STD)
    spectra: numpy array[complex float]
        NMR spectra output with the artifact and noise (real part of FFT)
    measurement: bool
        False if no measurement is conducted yet (default False). Once measure method is invoked
        it becomes True

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
    ip2 = 0.5/np.pi

    # spectrometer constructor
    def __init__(
            self,
            B=10.0,
            timeunit='msec',
            shift_maximum=128.0,
            shift_minimum=15,
            t_cut=1500,
            f_min=0.2,
            RH=12,
            r=0.005,
            std=0.00005,
            dtype='float32'):
        """ spectrometer constructor

        Parameters
        ----------
        B: float
            External magnetic field (default 10.0 Tesla) 
            This affects angular Larmor frequency, and Larmor frequency together with shift_maximum
            is used to determine the sampling rate
        timeunit: str
            Unit string for time variable. It is either msec or micron (default msec)
        shift_maximum: float
            Maximum chemical shift to set the maximum frequency of frequency region of FFT calculation
            (default 128.0 ppm). However, the spectrometer FFT output is clipped at shift_cutoff 
            determined by shift_minimum because no molecules are expected to be detected after such
            frequency. In other words, shift_maximum together with angular Larmor frequency is used to 
            determine sampling rate as sampling rate sets the maximum frequency of FFT output region
        shift_minimum: float
            Minimum chemical shift that the frequency range of FFT must include (default 15.0)
            Remember that FFT calculation output is truncated to yield spectrometer FFT output such that 
            the spectrometer FFT output range is set to include shift_minimum and to be much smaller than
            shift_maximum
        t_cut: float
            Cutoff time period that signal sampling duration, t, must exceed (default 1500.0)
            This will determine how long the spectrometer will measure the signal. Together with f_min
            it determines the number of signal samples and FFT output
        f_min: float
            Minimum frequency resolution for high resolution NMR spectrum (default 0.2 Hz). Together
            with t_cut, it determines the number of signal samples and FFT output
        RH: integer
            Number of hydrogens the reference molecule contains (default 12)
        r: float
            Relaxivity for reference hydrogen (default 0.005 kHz)
        std: float
            Standard deviation of signal noise (default 0.0001)
        dtype: str
            Data type for shift and spectra
        """

        # spectrometer constructor attributes
        self.B = B # Approximately 2000 msec is T2 for water/CSF at 1.5T
        self.timeunit = timeunit
        self.shift_maximum = shift_maximum
        self.p_l = int(np.log2(shift_maximum/shift_minimum))
        self.shift_cutoff = shift_maximum*pow(2, -self.p_l)
        self.f_unit, self.ep = self.unit(timeunit)
        self.w_l = self.gamma*B*pow(10, self.ep)
        self.f_l = self.ip2*self.w_l
        self.f_s = self.ip2*shift_maximum*pow(10, -6)*self.w_l
        self.dt = 1/self.f_s
        self.ns, self.p, self.t = self.time(t_cut, f_min)
        self.df = self.f_s*pow(2, -self.p)
        self.nf = pow(2, self.p-self.p_l)
        self.w_max = 2*np.pi*self.nf*self.nf
        self.f = self.df*np.arange(0, self.nf)
        self.shift = ((self.shift_cutoff/self.nf)*np.arange(0, self.nf)).astype(dtype)
        self.hr = 1/RH
        self.r = r
        self.std = std
        self.dtype = dtype
        self.noise = np.zeros(self.ns, dtype=np.complex128)
        self.ps = np.zeros(2)
        self.smooth = 0
        self.spectra_artifact = np.zeros(self.nf)
        self.measurement = False

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
        ex: int
            Exponent of 10 to convert seconds into miliseconds or microseconds
        """

        if timeunit == 'msec':
            return 'kHz', -3
        elif timeunit == 'micron':
            return 'MHz', -6
        else:
            raise ValueError('incorrect time unit is specified: use msec or micron')
        
    # spectrometer time method
    def time(self, t_cut, f_min):
        """
        A list of times at which the signal is sampled with sampling interval and rate.

        Parameters
        ----------
        t_cut: float
            Cutoff time that the maximum t value must exceed (default 1500.0)
        f_min: float
            Minimum frequency resolution for high resolution NMR spectrum (default 0.2 Hz)

        Returns
        -------
        ns: integer
            Number of signal samples
        p: integer
            Power of two data points for sampled signal to be processed before clipping at
            cutoff chemical shift (usually more than ns, thus zero padding occurs)
        t: numpy array[float]
            Sampled times
        """
        p = int(np.log2(t_cut*self.f_s + 1)) + 1
        ns = pow(2, p) # total number of signal samples including t = 0 
        p += int(np.log2(self.f_s*pow(10, -self.ep)/(ns*f_min))) + 1
        return ns, p, np.arange(0, ns)/self.f_s

    # spectrometer calibrate method
    def calibrate(
            self,
            B=10.0,
            timeunit='msec',
            shift_maximum=128.0,
            shift_minimum=15,
            t_cut=1500,
            f_min=0.2,
            RH=12,
            std=0.00005,
            dtype='float32',
            baseline=False,
            phase_shift=False):
        """
        Spectrometer calibrate method

        This method will calibrate spectrometer settings to default if no inputs were provided.
        It is essentially __init__. Parameters for this method are the same as the constructor
        """
        self.__init__(
                B=B,
                timeunit=timeunit,
                shift_maximum=shift_maximum,
                shift_minimum=shift_minimum,
                t_cut=t_cut,
                f_min=f_min,
                RH=RH,
                std=std,
                dtype=dtype,
                baseline=baseline,
                phase_shift=phase_shift)

    # spectrometer artifact method
    def artifact(
            self,
            baseline=False,
            phase_shift=False,
            smoothness=False):
        """
        Artifact method

        For now, only baseline distortion artifact is implemented
        phase shift is next to be added here

        Parameters
        ----------
        baseline: Bool
            If true, baseline distortion artifact is created (default False)
        phase_shift: Bool
            If true, phase shift (zero and first order) is applied to hydrogens (default False)
        smoothness: Bool
            The target (as opposed to real) signal measurement output is noiseless if True 
            (default False). Noise is inherently present in the real data. For inspection, set 
            noise=False for measure method
        """
        # initialize spectra artifact
        self.spectra_artifact = np.zeros(self.nf)
        
        # add baseline artifact to the final spectra
        if baseline:
            n = np.random.randint(2, 25)
            sd = 0.15
            w = 0.3/sd
            upper_bound = truncnorm(-w, w, loc=0.3, scale=sd).rvs(1)[0]
            y = np.random.uniform(0.0, upper_bound, n+1)
            if (2 < n) and (n < 21):
                bin_size = self.shift_cutoff/n
                std = bin_size/10
                b = 0.5*bin_size/std
                x = np.array(
                        [0]+
                        [truncnorm(-b, b, loc=bin_size*mu, scale=std).rvs(1)[0] for mu in range(1, n)]+
                        [self.shift_cutoff])
                tck = interpolate.splrep(x, y, s=0)
                self.spectra_artifact += interpolate.splev(self.shift, tck, der=0)
            else: 
                self.spectra_artifact += ( (y[-1] - y[0])/self.shift_cutoff*self.shift + y[0] )

        # add phase shift to the raw signal
        if phase_shift:
            random_number = np.random.uniform(0, 1)
            if random_number < 0.25:
                # slope for first order phase shift
                self.ps[0] = np.random.uniform(0, 0.125*np.pi/self.w_max)             
                # y-intercept for zero order phase shift
                self.ps[1] = 0
            elif 0.75 < random_number:
                self.ps[0] = 0
                self.ps[1] = np.random.uniform(0, 0.125*np.pi) 
            else:
                self.ps[0] = np.random.uniform(0, 0.125*np.pi/self.w_max) 
                self.ps[1] = np.random.uniform(0, 0.125*np.pi) 

        # noiseless target signal if true
        if smoothness == True:
            self.smooth = 0
        else:
            self.smooth = 1

    # spectrometer measure method
    def measure(self, moles, noise=True, extra_target=False):
        """" 
        Measures FID signal from the sample
        
        Parameter
        ---------
        moles: dict[str]:(molecule, float)
            Dictionary that contains multiple molecule objects with their relative abundances.
        noise: bool
            If True, noise is introduced with std
        extra_target: bool
            If true, two target spectra will be returned instead of one (default False). The extra
            target has different noise than the first one, and it is used to compare performance
            between corrected spectrum and the different target spectrum
        """
        self.measurement = True # to indicate that at least one measurement is done
        self.extra_target=extra_target # to includate extra target with different noise
        t = self.t # measurement time period for notation readability

        # adding noise to the raw signal if noise is true
        if noise:
            real_noise = np.random.normal(0, self.std, self.ns)
            imag_noise = np.random.normal(0, self.std, self.ns)
            self.noise = real_noise + 1j*imag_noise 

        # relaxivities for corresponding hydrogen groups
        relaxivity = {x:{y: 1/moles[x][0].hydrogens[y][2] for y in moles[x][0].hydrogens} 
                     for x in moles}

        # Split frequencies and their relative abundance (relative to RH)
        A = [(
            moles[x][0].hydrogens[y][1]*pow(10, -6)*self.w_l,
            moles[x][1]*moles[x][0].hydrogens[y][0]*self.hr,
            relaxivity[x][y])
            for x in moles for y in moles[x][0].hydrogens if y not in moles[x][0].splits] \
        +   [(
            pow(10, -6)*moles[x][0].hydrogens[y][1]*self.w_l + 2*pow(10, self.ep)*np.pi*z,
            moles[x][1]*moles[x][0].hydrogens[y][0]*k*self.hr,
            relaxivity[x][y])
            for x in moles for y in moles[x][0].splits
            for z, k in zip(moles[x][0].splits[y][0], moles[x][0].splits[y][1])]
       
        self.splits = A
        
        S, T = self.ps  # Add zero or first order phase shift [0, pi/4) if phase_shift is true
        # Separate FID list is created. Target FID is what we aim to get from DNN
        separate_fid = [N*np.exp(1j*(w*t + S*w + T))*np.exp(-r*t) for w, N, r in A] 
        target_fid = [N*np.exp(1j*w*t)*np.exp(-r*t) for w, N, r in A] 
            
        # Final signal and its spectra (FFT of signal) from all hydrogen FID
        self.signal = self.dt*self.r*np.sum(separate_fid, axis=0) + self.noise
        self.target_signal = self.dt*self.r*np.sum(target_fid, axis=0) + self.smooth*self.noise

        ###### any modification to the measured signal is done here ######



        # extra target to compare performance
        if extra_target:
            real_noise = np.random.normal(0, self.std, self.ns)
            imag_noise = np.random.normal(0, self.std, self.ns)
            noise2 = real_noise + 1j*imag_noise 
            target_signal2 = self.dt*self.r*np.sum(target_fid, axis=0) + noise2
            target_FFT2 = np.fft.fft(target_signal2, n=pow(2, self.p))[:self.nf]
            self.target2 = target_FFT2.real.astype(self.dtype)
        else:
            self.target2 = None

        # DFT calculations
        self.FFT = np.fft.fft(self.signal, n=pow(2, self.p))[:self.nf]
        self.target_FFT = np.fft.fft(self.target_signal, n=pow(2, self.p))[:self.nf]
        self.spectra = self.FFT.real.astype(self.dtype) + self.spectra_artifact.astype(self.dtype)
        self.target = self.target_FFT.real.astype(self.dtype)

    def __repr__(self):
        return "Spectrometer class that measures a sample solution with organic molecules in it"

    def __call__(self):
        try:
            if self.extra_target:
                return self.spectra, self.target, self.target2
            else:
                return self.spectra, self.target
        except AttributeError:
            return None

# Lorentzian Class
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
    f: list[float]
        Vertical frequency axis values
    shift: list[float]
        Vertical frequency axis values in chemical shift
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
        if isinstance(ob, fid.fid):
            self.unit = ob.frequency_unit
            self.ns = ob.ns
            self.p = ob.p
            self.f = np.arange(0, ob.ns)*ob.f_s/ob.ns # the last f excludes fmax
            self.shift = pow(10, 6)*self.f/ob.f_l
            self.r = ob.r
            self.f0 = ob.f0
            self.lorentz = self.lorz()
        else:
            self.unit = unit
            self.ns = ns
            self.p = np.log2(ns)
            self.f = np.arange(0, ns)*f_max/ns # the last f excludes f_max
            self.shift = pow(10, 6)*self.f/f_l
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

def max_reduction(spec, data_length):
    """
    this function takes spectrometer object and returns the measured spectra and target with
    reshaped size using max method. Reduced chemical shift range is also returned (chemical
    shift range is still the same, only the number of data points are truncated)

    Parameters
    ----------
    spectrometer: spectrometer object
        spectrometer object with measured signal and spectra
    data_length: int
        size of resized spectrometer output. If the size is bigger than spectrometer.nf,
        then the original output (spectro and target) will be returned. Make sure this number
        is some power of two (eg, 2**10)

    Returns
    -------
    spectra: numpy array[float] 
        reshaped spectra of spectrometer
    target: numpy array[float] 
        reshaped target of spectrometer
    chemical shift: numpy array[float]
        chemical shift range, but fewer data points to match the data size
    """
    assert spec.measurement == True, "spectrometer has no measurement"
    rescale_ratio = int(spec.nf/data_length)

    if  spec.nf <= data_length:
        return spec()
    else:
        target = np.reshape(spec.target, (data_length, rescale_ratio))
        spectra = np.reshape(spec.spectra, (data_length, rescale_ratio))
        return np.max(spectra, axis=1), np.max(target, axis=1), spec.shift[::rescale_ratio]

def mean_reduction(spec, data_length):
    """
    this function takes spectrometer object and returns the measured spectra and target with
    reshaped size using mean method. Reduced chemical shift range is also returned (chemical
    shift range is still the same, only the number of data points are truncated)

    Parameters
    ----------
    spectrometer: spectrometer object
        spectrometer object with measured signal and spectra
    data_length: int
        size of resized spectrometer output. If the size is bigger than spectrometer.nf,
        then the original output (spectro and target) will be returned. Make sure this number
        is some power of two (eg, 2**10)

    Returns
    -------
    spectra: numpy array[float] 
        reshaped spectra of spectrometer
    target: numpy array[float] 
        reshaped target of spectrometer
    chemical shift: numpy array[float]
        chemical shift range, but fewer data points to match the data size
    """
    assert spec.measurement == True, "spectrometer has no measurement"
    rescale_ratio = int(spec.nf/data_length)

    if  spec.nf <= data_length:
        return spec()
    else:
        target = np.reshape(spec.target, (data_length, rescale_ratio))
        spectra = np.reshape(spec.spectra, (data_length, rescale_ratio))
        return np.mean(spectra, axis=1), np.mean(target, axis=1), spec.shift[::rescale_ratio]

def load_spec_data(
        directory, 
        batch_size=32,
        numpy_array=False,
        extra_target=False):
    """
    this function loads data from hdf5 files in a directory and returns datasets as either numpy arrays or TF datasets. The directory must contain hdf5 files, the total number of which must be 10*2**p where p is a positive integer. This ensures that datasets of train, valid, and test are 80%, 10%, and 10% each. 

    parameters
    ----------
    directory: PosixPath or str
        directory path in which HDF5 data files are stored and from which load_spec_data loads
        such files. The numbmer of HDF5 files must be 10*2**p where p is a positive integer
    batch_size: int
        batch size of data for model training
    numpy_array: bool
        If True, the function returns dataset as numpy arrays (default False). Otherwise the
        returned datasets are TF dataset
    extra_target: bool
        If True, there is one more target, namely target2, which is the same target as the first one,
        but with different noise

    return
    ------
    datasets of train, valid, test: TF datasets or numpy arrays
        TF datasets of train, valid, and test are returned unless numpy_array == True 
    """
    # get all hdf5 file paths and make sure that there are more than 10 hdf5 files
    data_dir = Path(directory)
    file_paths = data_dir.glob('*.hdf5')
    hdf5_files = [str(file) for file in file_paths]    
    assert 9 < len(hdf5_files), "there are fewer than 10 hdf5 files"

    # making sure that the number of hdf5 files are 10*2**p where p is a positive integer
    p = np.log2(len(hdf5_files)/10)
    assert p.is_integer(), "number of hdf5 files are not 10*2**p where p is a positive int"

    # create train, valid, and test hdf5 file path lists separately
    train_num = int(8*2**p)
    valid_num = int(2**p)
    train_data_files = hdf5_files[:train_num]
    valid_data_files = hdf5_files[train_num:train_num+valid_num]
    test_data_files = hdf5_files[-valid_num:]

    # obtain number of data in each hdf5 file ans its data type and length
    with h5py.File(hdf5_files[0], 'r') as f:
        dtype = f['data'].dtype
        num_samples = f['data'].shape[0]
        data_length = f['data'].shape[1]

    # set buffer size based on the number of data in each hdf5 file
    buffer_size = int(num_samples/32)

    # preallocate numpy arrays for data
    X_train = np.zeros((num_samples*train_num, data_length), dtype=dtype)
    y_train = np.zeros((num_samples*train_num, data_length), dtype=dtype)

    X_valid = np.zeros((num_samples*valid_num, data_length), dtype=dtype)
    y_valid = np.zeros((num_samples*valid_num, data_length), dtype=dtype)

    X_test = np.zeros((num_samples*valid_num, data_length), dtype=dtype)
    y_test = np.zeros((num_samples*valid_num, data_length), dtype=dtype)

    if extra_target:
        y_train2 = np.zeros((num_samples*train_num, data_length), dtype=dtype)
        y_valid2 = np.zeros((num_samples*valid_num, data_length), dtype=dtype)
        y_test2 = np.zeros((num_samples*valid_num, data_length), dtype=dtype)

    # load the data into numpy arrays
    for index, file_path in enumerate(train_data_files):
        start = num_samples*index
        with h5py.File(file_path, 'r') as f:
            X_train[start:start+num_samples] = f['data'][:]
            y_train[start:start+num_samples] = f['target'][:]
            if extra_target:
                y_train2[start:start+num_samples] = f['target2'][:]

    for index, file_path in enumerate(valid_data_files):
        start = num_samples*index
        with h5py.File(file_path, 'r') as f:
            X_valid[start:start+num_samples] = f['data'][:]
            y_valid[start:start+num_samples] = f['target'][:]
            if extra_target:
                y_valid2[start:start+num_samples] = f['target2'][:]

    for index, file_path in enumerate(test_data_files):
        start = num_samples*index
        with h5py.File(file_path, 'r') as f:
            X_test[start:start+num_samples] = f['data'][:]
            y_test[start:start+num_samples] = f['target'][:]
            if extra_target:
                y_test2[start:start+num_samples] = f['target2'][:]

    # sometimes numpy array dataset is needed
    if numpy_array == True:
        if extra_target:
            return X_train, y_train, y_train2,  X_valid, y_valid, y_valid2, X_test, y_test, y_test2 
        else:
            return X_train, y_train, X_valid, y_valid, X_test, y_test 

    # create TF dataset shuffled, batched and prefetched
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset_train = dataset_train.shuffle(buffer_size=buffer_size, seed=42)
    dataset_train = dataset_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    dataset_valid = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    dataset_valid = dataset_valid.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset_test = dataset_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset_train, dataset_valid, dataset_test

def load_shift(hdf5_path):
    """
    returns chemical shift range from hdf5 path

    parameter
    ---------
    hdf5_path: Path object or str
        file path of chemical shift

    return
    ------
    shift: numpy array
        numpy array of chemical shift
    """
    with h5py.File(hdf5_path, 'r') as f:
        shift = f['shift'][:]

    return shift

def HDF5_load(hdf5_path, batch_size=128, numpy_array=False, extra_target=False):
    """
    HDF5_load loads a single hdf5 file of NMR spectra and returns either TF dataset or Numpy array
    """
    with h5py.File(hdf5_path, 'r') as f:
        X = f['data'][:]
        y = f['target'][:]
        if extra_target:
            y2 = f['target2'][:]

    # sometimes numpy array dataset is needed
    if numpy_array == True:
        if extra_target:
            return X, y, y2
        else:
            return X, y
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def model_NMR(input_length, GRU_unit, first_filter_num, second_filter_num):
    """
    This function returns neural network model that first processes NMR spectral data using
    recurrent neural network (GRU and bidirectioinal), and then second processes with CNN layers. 
    Note that the very first CNN kernel size is 1 x 2N where N is the number of GRU units. Please
    inspect the below code to fully understand the model structure as they are more complicated 
    than described above

    parameters
    ----------
    input_length: int
        NMR data length which should be a power of 2
    GRU_unit: int
        number of GRU units which should be a power of 2
    first_filter_num: int
        number of first CNN filters which should be a power of 2
    second_filter_num: int
        number of second CNN filters which should be a power of 2
       
    return
    ------
    NMR model: Keras model
        Keras NN that processes NMR spectral data and returns modified spectra
    """
    # check your input arguments are powers of 2
    l2 = np.log2
    assert l2(GRU_unit).is_integer(), "GRU_unit is not a power of 2"
    assert l2(first_filter_num).is_integer(), "first_filter_num is not a power of 2"
    assert l2(second_filter_num).is_integer(), "second_filter_num is not a power of 2"

    seq_input = keras.layers.Input(shape=[input_length])

    expand_layer = keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))
    expand_output = expand_layer(seq_input)

    GRU_output = keras.layers.Bidirectional(
        keras.layers.GRU(GRU_unit, return_sequences=True))(expand_output)

    expand_output2 = expand_layer(GRU_output)

    cnn_layer1 = keras.layers.Conv2D(
        filters=first_filter_num,
        kernel_size=(1, 2*GRU_unit),
        activation='elu') # elu
    cnn_output1 = cnn_layer1(expand_output2)

    transpose_layer = keras.layers.Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 2]))
    transpose_output = transpose_layer(cnn_output1)

    cnn_layer2 = keras.layers.Conv2D(
        filters=second_filter_num,
        kernel_size=(1, first_filter_num),
        activation='selu') # selu
    cnn2_output = cnn_layer2(transpose_output)

    transpose2_output = transpose_layer(cnn2_output)

    cnn_layer3 = keras.layers.Conv2D(
        filters=1,
        kernel_size=(1, second_filter_num),
        activation='LeakyReLU') # selu
    cnn3_output = cnn_layer3(transpose2_output)

    flat_output = keras.layers.Flatten()(cnn3_output)

    model_output = keras.layers.Add()([seq_input, flat_output])
    return keras.Model(inputs=[seq_input], outputs=[model_output])

def sliced_spec_data(
        directory, 
        batch_size=32,
        slice_num=8,
        numpy_array=False):
    """
    this function loads hdf5 files of NMR spectra and returns TF dataset or numpy array of them. Each data sample is sliced into slice_num (default 8) to reduce the input data length, but train each sliced chunk as if it is an instance of NMR spectrum. This helps to reduce the VRAM requirement.

    parameters
    ----------
    directory: PosixPath or str
        directory path in which HDF5 data files are stored and from which load_spec_data loads
        such files. The numbmer of HDF5 files must be 10*2**p where p is a positive integer
    batch_size: int
        batch size of data for model training (default 32)
    slice_num: int
        number of slices from one NMR spectrum sample (default 8)
    numpy_array: bool
        If True, the function returns dataset as numpy arrays (default False). Otherwise the
        returned datasets are TF dataset

    return
    ------
    datasets of train, valid, test: TF datasets or numpy arrays
        TF datasets of train, valid, and test are returned unless numpy_array == True 
    """
    # get all hdf5 file paths and make sure that there are more than 10 hdf5 files
    data_dir = Path(directory)
    file_paths = data_dir.glob('*.hdf5')
    hdf5_files = [str(file) for file in file_paths]    
    
    # file number restrictions
    assert 9 < len(hdf5_files), "there are fewer than 10 hdf5 files"
    p = np.log2(len(hdf5_files)/10)
    assert p.is_integer(), "number of hdf5 files are not 10*2**p where p is a positive int"

    # create train, valid, and test hdf5 file path lists separately
    train_num = int(8*2**p)
    valid_num = int(2**p)
    train_data_files = hdf5_files[:train_num]
    valid_data_files = hdf5_files[train_num:train_num+valid_num]
    test_data_files = hdf5_files[-valid_num:]

    # obtain number of data in each hdf5 file ans its data type and length
    with h5py.File(hdf5_files[0], 'r') as f:
        dtype = f['data'].dtype
        num_samples = f['data'].shape[0]
        data_length = f['data'].shape[1]

    # set buffer size based on the number of data in each hdf5 file
    buffer_size = int(num_samples/32)

    # preallocate numpy arrays for data
    train_sample_num = num_samples*train_num
    X_train = np.zeros((train_sample_num, data_length), dtype=dtype)
    y_train = np.zeros((train_sample_num, data_length), dtype=dtype)

    valid_sample_num = num_samples*valid_num
    X_valid = np.zeros((valid_sample_num, data_length), dtype=dtype)
    y_valid = np.zeros((valid_sample_num, data_length), dtype=dtype)

    X_test = np.zeros((valid_sample_num, data_length), dtype=dtype)
    y_test = np.zeros((valid_sample_num, data_length), dtype=dtype)

    # load the data into numpy arrays
    for index, file_path in enumerate(train_data_files):
        start = num_samples*index
        with h5py.File(file_path, 'r') as f:
            X_train[start:start+num_samples] = f['data'][:]
            y_train[start:start+num_samples] = f['target'][:]

    for index, file_path in enumerate(valid_data_files):
        start = num_samples*index
        with h5py.File(file_path, 'r') as f:
            X_valid[start:start+num_samples] = f['data'][:]
            y_valid[start:start+num_samples] = f['target'][:]

    for index, file_path in enumerate(test_data_files):
        start = num_samples*index
        with h5py.File(file_path, 'r') as f:
            X_test[start:start+num_samples] = f['data'][:]
            y_test[start:start+num_samples] = f['target'][:]
               
    # smaller data length samples
    smaller_length = int(data_length/slice_num)
    X_train = X_train.reshape((train_sample_num, slice_num, smaller_length))
    X_train = X_train.reshape((train_sample_num*slice_num, smaller_length))
    y_train = y_train.reshape((train_sample_num, slice_num, smaller_length))
    y_train = y_train.reshape((train_sample_num*slice_num, smaller_length))

    X_valid = X_valid.reshape((valid_sample_num, slice_num, smaller_length))
    X_valid = X_valid.reshape((valid_sample_num*slice_num, smaller_length))
    y_valid = y_valid.reshape((valid_sample_num, slice_num, smaller_length))
    y_valid = y_valid.reshape((valid_sample_num*slice_num, smaller_length))

    X_test = X_test.reshape((valid_sample_num, slice_num, smaller_length))
    X_test = X_test.reshape((valid_sample_num*slice_num, smaller_length))
    y_test = y_test.reshape((valid_sample_num, slice_num, smaller_length))
    y_test = y_test.reshape((valid_sample_num*slice_num, smaller_length))

    # sometimes numpy array dataset is needed
    if numpy_array == True:
        return X_train, y_train, X_valid, y_valid, X_test, y_test 

    # create TF dataset shuffled, batched and prefetched
    dataset_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset_train = dataset_train.shuffle(buffer_size=buffer_size, seed=42)
    dataset_train = dataset_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    dataset_valid = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    dataset_valid = dataset_valid.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    dataset_test = dataset_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset_train, dataset_valid, dataset_test

def sliced_spec_data2(
        directory, 
        batch_size=32,
        slice_num=8,
        numpy_array=False):
    """
    same as sliced_spec_data, but fewer than 10 hdf5 files can be accepted and there is no segregation into train, valid, and test datasets

    parameters
    ----------
    directory: PosixPath or str
        directory path in which HDF5 data files are stored and from which load_spec_data loads
        such files. The numbmer of HDF5 files must be 10*2**p where p is a positive integer
    batch_size: int
        batch size of data for model training (default 32)
    slice_num: int
        number of slices from one NMR spectrum sample (default 8)
    numpy_array: bool
        If True, the function returns dataset as numpy arrays (default False). Otherwise the
        returned datasets are TF dataset

    return
    ------
    datasets of train, valid, test: TF datasets or numpy arrays
        TF datasets of train, valid, and test are returned unless numpy_array == True 
    """
    # get all hdf5 file paths and make sure that there are more than 10 hdf5 files
    data_dir = Path(directory)
    file_paths = data_dir.glob('*.hdf5')
    hdf5_files = [str(file) for file in file_paths]    
    num_files = len(hdf5_files)
    
    # obtain number of data in each hdf5 file ans its data type and length
    with h5py.File(hdf5_files[0], 'r') as f:
        dtype = f['data'].dtype
        num_samples = f['data'].shape[0]
        data_length = f['data'].shape[1]

    # preallocate numpy arrays for data
    train_sample_num = num_samples*num_files
    X = np.zeros((train_sample_num, data_length), dtype=dtype)
    y = np.zeros((train_sample_num, data_length), dtype=dtype)

    # load the data into numpy arrays
    for index, file_path in enumerate(hdf5_files):
        start = num_samples*index
        with h5py.File(file_path, 'r') as f:
            X[start:start+num_samples] = f['data'][:]
            y[start:start+num_samples] = f['target'][:]
               
    # reshape the data to increase number of samples with smaller data length
    smaller_length = int(data_length/slice_num)
    X = X.reshape((train_sample_num, slice_num, smaller_length))
    X = X.reshape((train_sample_num*slice_num, smaller_length))
    y = y.reshape((train_sample_num, slice_num, smaller_length))
    y = y.reshape((train_sample_num*slice_num, smaller_length))

    # sometimes numpy array dataset is needed
    if numpy_array == True:
        return X, y

    # create TF dataset shuffled, batched and prefetched
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

def save_history(history, filename):
    """
    Save a history dict as hdf5 file
    
    Parameters
    ----------
    history: dict[str:list]
        dictionary of history content ['loss', 'mse', 'val_loss', 'val_mse']. Each item is a list of floats 
    filename: str
        hdf5 file path string
    """
    with h5py.File(filename, "w") as f:
        for key, value in history.history.items():
            f.create_dataset(key, data=np.array(value))

