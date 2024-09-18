# References
"""
Keeler pg.49 for general signal form
proton gyromagnetic ratio: https://physics.nist.gov/cgi-bin/cuu/Value?gammap
"""

from itertools import product
from functools import partial
from copy import copy
import fid
from pathlib import Path
import h5py
import inspect
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.special import binom
from scipy.stats import truncnorm
from scipy import interpolate
import hyperopt

import tensorflow as tf
from tensorflow import keras
import nmrglue as ng


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
    ps_max: float
        Maximum scaler for phase shift randomizer (default 0.125)
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
            ps_max=0.125,
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
        ps_max: float
            Maximum scaler for phase shift randomizer (default 0.125)
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
        self.w_max = 2*np.pi*self.nf*self.df
        self.f = self.df*np.arange(0, self.nf)
        self.shift = ((self.shift_cutoff/self.nf)*np.arange(0, self.nf)).astype(dtype)
        self.hr = 1/RH
        self.r = r
        self.std = std
        self.ps_max = ps_max
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
            ps_max=0.250,
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
                ps_max=0.250,
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
            ps_max = self.ps_max*np.pi

            if random_number < 0.25:
                # slope for first order phase shift
                self.ps[0] = np.random.uniform(-2*ps_max/self.w_max, 2*ps_max/self.w_max)
                # y-intercept for zero order phase shift
                self.ps[1] = 0
            elif 0.50 < random_number:
                self.ps[0] = 0
                self.ps[1] = np.random.uniform(-2*ps_max, 2*ps_max) 
            else:
                b = np.random.uniform(-ps_max, ps_max)
                self.ps[1] = b 
                self.ps[0] = (1/self.w_max)*np.random.uniform(-ps_max-b, ps_max-b)

        # noiseless target signal if true
        if smoothness == True:
            self.smooth = 0
        else:
            self.smooth = 1

    # spectrometer measure method
    def measure(self, 
            moles, 
            noise=True, 
            extra_target=False, 
            second_std=None, 
            imaginary=False):
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
        second_std: None or float
            If None, noise for second target is the same as the first noise. If float, the second 
            noise will have std of the float
        imaginary: bool
            If true, if you have only phase shift artifacts, your output spectra will have imaginary
            part as well
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
        if extra_target and not second_std:
            real_noise = np.random.normal(0, self.std, self.ns)
            imag_noise = np.random.normal(0, self.std, self.ns)
            noise2 = real_noise + 1j*imag_noise 
            target_signal2 = self.dt*self.r*np.sum(target_fid, axis=0) + noise2
            target_FFT2 = np.fft.fft(target_signal2, n=pow(2, self.p))[:self.nf]
            self.target2 = target_FFT2.real.astype(self.dtype)
        elif second_std:
            real_noise = np.random.normal(0, second_std, self.ns)
            imag_noise = np.random.normal(0, second_std, self.ns)
            noise2 = real_noise + 1j*imag_noise 
            target_signal2 = self.dt*self.r*np.sum(target_fid, axis=0) + noise2
            target_FFT2 = np.fft.fft(target_signal2, n=pow(2, self.p))[:self.nf]
            self.target2 = target_FFT2.real.astype(self.dtype)
        else: 
            self.target2 = None

        # DFT calculations
        self.FFT = np.fft.fft(self.signal, n=pow(2, self.p))[:self.nf]
        self.target_FFT = np.fft.fft(self.target_signal, n=pow(2, self.p))[:self.nf]
        self.target = self.target_FFT.real.astype(self.dtype)

        # output spectrum might contain imaginary part or not
        if imaginary == False:
            self.spectra = self.FFT.real.astype(self.dtype) + self.spectra_artifact.astype(self.dtype)
        else:
            self.spectra = self.FFT

    def coinflip():
        return np.random.choice([True, False])

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

def load_spec_data_small(
        directory, 
        batch_size=32,
        numpy_array=False):
    """
    this function loads data from hdf5 files in a directory and returns datasets as either numpy arrays or TF datasets. There could be fewer than 10 hdf5 files in the directory.

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

    # set buffer size based on the number of data in each hdf5 file and preallocate X, y
    buffer_size = int(num_samples/32)
    X = np.zeros((num_samples*num_files, data_length), dtype=dtype)
    y = np.zeros((num_samples*num_files, data_length), dtype=dtype)

    # load the data into numpy arrays
    for index, file_path in enumerate(hdf5_files):
        start = num_samples*index
        with h5py.File(file_path, 'r') as f:
            X[start:start+num_samples] = f['data'][:]
            y[start:start+num_samples] = f['target'][:]

    # sometimes numpy array dataset is needed
    if numpy_array == True:
        return X, y 

    # create TF dataset batched and prefetched
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

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

def bruker_data(fid_path, data_length=32768, max_height=5.0, std=0.02):
    """
    bruker_data loads bruker NMR data from BMRB database. The output data will be rescaled according
    to max_height to fit into DNN model. Noise will be returned as well. Note that real and 
    and imaginary part might be swapped and inverted. For nmrglue processing, use nmrglue.bruker.read
    directly instead.
    
    parameters
    ----------
    fid_path: str
        directory of NMR sample data path. Inside of it is pdata directory
    max_height: float
        approximate maximum height threshold of time-domain data
        
    returns
    -------
    data_model: numpy array
        Time-domain NMR data with shape (32768, )
    """
    # read in the bruker formatted data
    dic, data_original = ng.bruker.read(fid_path)

    # remove the digital filter
    data = ng.bruker.remove_digital_filter(dic, data_original)

    # process the spectrum
    data = ng.proc_base.zf_size(data, data_length)  # zero fill to 32768 points
    data = ng.proc_base.fft(data)                   # Fourier transform

    # get maxima and minima of data
    maxr = np.max(data.real)
    minr = np.abs(np.min(data.real))
    maxi = np.max(data.imag)
    mini = np.abs(np.min(data.imag))
    MAXES = np.array([maxr, minr, maxi, mini])
    max_index = np.argmax(MAXES)

    if max_index == 0:
        data_model = data.real
    if max_index == 1:
        data_model = data.imag
    if max_index == 2:
        data_model = -data.real
    if max_index == 3:
        data_model = -data.imag

    # rescale the data
    scale_data = max_height/np.max(data_model)
    data_model = scale_data*data_model
    data_ng = scale_data*data
    
    # get noise
    noise = np.random.normal(0, std, size=data_length)

    # return data and noise for model, and raw FFT processed data
    return data_model, noise, data_ng

def common_peaks_with_accuracy(model_path, accuracy_path, sample_paths, sample_no):
    """
    this fn will return common peaks of model and ng for given sample number. Note that
    sample_no = sample_index + 1

    parameters
    ----------
    model_path: PosixPath or str
        model file (hdf5) path to load the model
    accuracy_path: str
        accuracy file path. Accuray list contains WAS_ATA_accu contains WAS_values, ATA_values, and
        accuracies for all the peaks of both model and ng processed data for all the samples
    sample_paths: list[str]
        list of NMR sample data (BMSE) path strings.
    sample_no: int
        sample number of NMR data with common peaks. sample_no = sample_index + 1

    returns
    -------
    peaks_model: list[tuple(float, int, float, float)]
        list of tuples for all the identified peaks. The tuple contains the following information:
            locations : Peak locations
            cluster_ids : Cluster numbers for peaks
            scales : Estimated peak scales (linewidths)
            amps : Estimated peak amplitudes
    peaks_ng: list[tuple(float, int, float, float)]
        same as peaks_model, but for nmrglue processed data
    """
    # load the model and accuracy
    model = keras.models.load_model(model_path, compile=False)
    with open(accuracy_path, "rb") as f:
        sample_accuracy = pickle.load(f)

    # find all indices of sample accuracy for common peaks
    indices_with_peaks = []
    for ind, sample in enumerate(sample_accuracy):
        if sample != 0:
            indices_with_peaks.append(ind)

    # load the data
    ind = sample_no - 1
    std = 0.02
    data_model, noise_real, data_ng = bruker_data(sample_paths[indices_with_peaks[ind]], std=std)

    # noises
    size = noise_real.shape[0]
    noise_imag = np.random.normal(0.0, std, size=size)
    noise = noise_real + 1j*noise_imag

    # obtain model and ng output
    model_input = np.array([data_model + noise_real])
    model_output = model(model_input)
    model_numpy = model_output.numpy()[0] - noise_real

    ng_output = ng.proc_autophase.autops(data_ng + noise, fn='acme', p0=0.1, p1=0.1) # +noise
    ng_output = ng.proc_bl.baseline_corrector(ng_output) - noise

    # peak, width, amplitude, location threshold for ng_output and model output
    pthres = 0.2
    wthres = 15.0
    Athres = 25.0
    loc_thres = 5

    # pick peaks
    peak_model = ng.analysis.peakpick.pick(model_numpy, pthres=pthres)
    peak_ng = ng.analysis.peakpick.pick(ng_output.real, pthres=pthres)

    # Find matching elements within the threshold
    peaks_ng = []
    peaks_model = []
    for i in range(len(peak_ng)):
        for j in range(len(peak_model)):
            if abs(peak_ng[i][0] - peak_model[j][0]) <= loc_thres:
                if Athres < peak_ng[i][3] and wthres < peak_ng[i][2]:
                    peaks_ng.append(peak_ng[i])
                    peaks_model.append(peak_model[j])

    return peaks_model, peaks_ng, model_numpy, ng_output

def common_peaks(model_path, sample_path):
    """
    common_peaks will return common peaks of model and ng for given index number for sample files
    as well as model and ng processed spectra

    parameters
    ----------
    model_path: PosixPath or str
        model file (hdf5) path to load the model
    sample_paths: list[str]
        list of NMR sample data (BMSE) path strings.
    index: int
        index of NMR data from sample_paths

    returns
    -------
    peaks_model: list[ndarray]
        list of numpy arrays for all the identified peaks. The array contains the following information:
            locations : Peak locations
            scales    : Estimated peak scales (linewidths)
            amps      : Estimated peak amplitudes
            syms      : Estimated peak symmetricity
    peaks_ng: list[ndarray]
        same as peaks_model, but for nmrglue processed data
    model_numpy: ndarray
        DNN model processed spectra
    ng_output: ndarray
        nmrglue processsed spectra
    model_input: ndarray
        uncorrected spectrum (real part) DNN model processed
    """
    # load the model and data
    model = keras.models.load_model(model_path, compile=False)
    data_model, noise_real, data_ng = bruker_data(sample_path)

    # noises
    size = noise_real.shape[0]
    std = np.std(noise_real)
    noise_imag = np.random.normal(0.0, std, size=size)
    noise = noise_real + 1j*noise_imag

    # obtain model and ng output
    model_input = np.array([data_model + noise_real])
    model_output = model(model_input)
    model_numpy = model_output.numpy()[0] - noise_real

    ng_output = ng.proc_autophase.autops(data_ng + noise, fn='acme', p0=0.1, p1=0.1) # +noise
    ng_output = ng.proc_bl.baseline_corrector(ng_output.real) - noise_real

    # peak, width, amplitude, location threshold for ng_output and model output
    pthres = 0.2
    wthres = 15.0
    Athres = 25.0
    loc_thres = 5

    # pick peaks
    peak_model = ng.analysis.peakpick.pick(model_numpy, pthres=pthres)
    peak_ng = ng.analysis.peakpick.pick(ng_output.real, pthres=pthres)

    # Find matching elements within the threshold
    peaks_ng = []
    peaks_model = []
    for i in range(len(peak_ng)):
        for j in range(len(peak_model)):
            if abs(peak_ng[i][0] - peak_model[j][0]) <= loc_thres:
                if Athres < peak_ng[i][3] and wthres < peak_ng[i][2]:
                    peaks_ng.append(peak_ng[i])
                    peaks_model.append(peak_model[j])

    # If no common peaks were found, do not proceed to calculate symmetricity
    peak_num = len(peaks_model)
    if peak_num == 0 :
        print(f"-----no common peaks found-----")
        return peaks_model, peaks_ng, model_numpy, ng_output, model_input

    # get symmetricities for both ng and model peaks
    for n in range(peak_num):
        # get loc, w, A of peaks (model and ng)
        loc1, _, w1, A1 = peaks_model[n]
        loc2, _, w2, A2 = peaks_ng[n]
        scale = 2

        # get symmetricity for model
        model_area = model_numpy[int(loc1-scale*w1):int(loc1+scale*w1+1)]
        model_len = len(model_area)
        difference_model = model_numpy[:(model_len//2)+1] - model_numpy[::-1][:model_len//2+1]
        sym1 = np.sum(difference_model)/A1
        peaks_model[n] = np.array([loc1, w1, A1, sym1])

        # get symmetricity for ng
        ng_area = ng_output[int(loc2-scale*w2):int(loc2+scale*w2+1)]
        ng_len = len(ng_area)
        difference_ng = ng_output[:(ng_len//2)+1] - ng_output[::-1][:ng_len//2+1]
        sym2 = np.sum(difference_ng)/A2
        peaks_ng[n] = np.array([loc2, w2, A2, sym2])

    return peaks_model, peaks_ng, model_numpy, ng_output, model_input

def spectrum_gen(abundance=50.0, T2=100.0, angle=0.0, cs=0.5, shift_minimum=0.6, std=0.0):
    """
    spectrum_gen generates single peak NMR spectrum based on input parameters (see below).
    The default maximum chemical shift range for FFT is 128.0 ppm.

    parameters
    ----------
    abundance: float
        proton abundance (default 50.0, and unit is spectrometer specific)
    T2: float
        T2 value, relaxation constant in ms (default 100.0)
    angle: float
        phase angle of time-domain signal (default 0.0)
    cs: float
        chemical shift of the peak (default 0.5)
    shift_minimum: float
        minimum chemical shift to be included in the chemical shift range (default 0.6)
    std: float
        STD for spectrum noise (default 0.0)

    returns
    -------
    spectrum: ndarray
        numpy array of NMR spectrum with single peak. The default length is 4096.
    """
    spec = spectrometer(shift_maximum=128.0, shift_minimum=shift_minimum, std=std)
    couplings = []

    # prepare sample and measure the spectrum
    hydrogens = {'a':(abundance, cs, T2)}
    mole = molecule(hydrogens=hydrogens, couplings=couplings)
    moles = {'A': (mole, 1)}
    spec.ps[1] = float(2*np.pi*angle/180.0)
    spec.measure(moles=moles)

    return spec.spectra

def estimate_WAS(ATA_values, pthres=0.2):
    """
    estimate_WAS estimates WAS_values, [width, amplitude, symmetricity], based on ATA_values,
    [H abundance, T2, angle].

    parameters
    ATA_values: ndarray
        numpy array of abundance, amplitude, symmetricity. The shape is (3,)
    pthres: float
        peak threshold for nmrglue peak picking process (default 0.2). Not to be touched.

    return
    ------
    WAS_values: ndarray
        numpy array of width, amplitude, symmetricity. The shape is (3,)
    """
    spec = spectrometer(shift_maximum=128.0, shift_minimum=0.6, std=0.0)
    couplings = []

    # prepare sample and measure the spectrum
    hydrogens = {'a':(ATA_values[0], 0.5, ATA_values[1])}
    mole = molecule(hydrogens=hydrogens, couplings=couplings)
    moles = {'A': (mole, 1)}
    spec.ps[1] = float(2*np.pi*ATA_values[2]/180.0)
    spec.measure(moles=moles)

    # get peak
    h = 0.3*np.max(spec.spectra)
    if h < pthres:
        pthres = h
    loc, _, w, A = ng.analysis.peakpick.pick(spec.spectra, pthres=pthres)[0]

    # get peak area
    scale = 2
    peak_area = spec.spectra[int(loc-scale*w):int(loc+scale*w+1)]
    area_len = len(peak_area)

    # get difference sum divided by amplitude around the peak
    difference = spec.spectra[:(area_len//2)+1] - spec.spectra[::-1][:area_len//2+1]
    sym = np.sum(difference)/A

    return np.array([w, A, sym], dtype='float32')

def guess_ATA(database_path, WAS_values):
    """
    guess_ATA first guesses proton abundance, T2, and phase angle of spectrum from WAS_values
    where WAS_values = [width, amplitude, symmetricity]. For this, width-amplitude-symmetricity
    database is used where indices represent [abundance - base_num, T2 - base_num, 10*angle]
    (the default base_num is 25). It calculates the shortest distance between the input
    WAS_values and the database WAS_values, and use the indices of such database WAS_values to
    rough estimate ATA values.

    parameters
    ----------
    WAS_values: ndarray
        numpy array that contains measured (by nmrglue) width, amplitude, symmetricity.
    database_path: PosixPath or str
        path for ATA_WAS database. database array has 4 dimensions: i, j, k and (w, A, s):
            i - proton abundance from base abundance to the number of iterations
            j - T2 values from base T2 to the number of iterations
            k - angle values: angle = 0.1*k
            (w, A, s) - width, amplitude, symmetricity measured by common_peaks method
        current database.shape is (400, 300, 200, 3)
    """
    database = np.load(str(database_path))
    WAS_copy = copy(WAS_values)

    # parameters of the database
    base_num = 25.0
    abundance_num = 400
    T2_num = 300
    angle_num = 200
    sym_sign = np.sign(WAS_copy[2])
    WAS_copy[2] *= -sym_sign

    # first rough estimate of ATA values
    weights = 1.0/np.abs(WAS_copy)
    disp = weights*np.abs(database - WAS_copy)
    distance = np.linalg.norm(disp, axis=-1)
    min_idx = np.unravel_index(np.argmin(distance), distance.shape)
    ATA_guesses = np.array(min_idx) + np.array([base_num, base_num, 0])
    ATA_guesses[2] *= -0.1*sym_sign

    return ATA_guesses

def estimate_ATA(WAS_values, database_path):
    """
    Estimate the ATA_values ([abundance, T2, angle]) of a sample using scipy optimizers.
    WAS_values ([width, amplitude, symmetricity]) of the sample must match the WAS values
    from estimate_WAS(ATA_values). This function uses a combination of least squares,
    root finding, and minimization algorithms from scipy to optimize the ATA values.
    The optimization process is guided by a Bayesian framework, which adaptively selects
    the best algorithm and scaling factor to use at each iteration.

    guess_ATA is first used to generate the initial ATA_guesses [abundance, T2, angle]
    (WAS_values and database_path are input arguments for guess_ATA). With initial_ATA,
    estimate_WAS is used to generate the first WAS_values. Two WAS_values arrays are then
    to be compared by optimizer numerically until the newly generated WAS_values become very
    close to the input WAS_values. This process repeats with different algorithms and scalars
    selected by Bayesian optimizations until the best result is obtained.

    parameters
    ----------
    WAS_values: ndarray
        numpy array that contains measured (by nmrglue) width, amplitude, symmetricity.
    database_path: PosixPath or str
        path for ATA_WAS database. database array has 4 dimensions: i, j, k and (w, A, s):
            i - proton abundance from base abundance to the number of iterations
            j - T2 values from base T2 to the number of iterations
            k - angle values: angle = 0.1*k
            (w, A, s) - width, amplitude, symmetricity measured by common_peaks method
        current database.shape is (400, 300, 200, 3)

    returns
    -------
    optimizer output: OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.
    """
    # WAS inputs, symmetricity sign, and optimizer inputs
    width, amplitude, sym = copy(WAS_values)
    initial_ATA = guess_ATA(database_path, WAS_values)
    abun, T2, angle = initial_ATA
    print(f"WAS values : {width, amplitude, sym}")
    print(f"ATA guess  : {initial_ATA}")
    print()

    #### least_squares dictionary for Bayesian opt ####
    least_squares = scipy.optimize.least_squares
    bounds0 = [(abun-4.0, T2-3.0, angle-1.0), (abun+4.0, T2+3.0, angle+1.0)]
    parameters = inspect.signature(least_squares).parameters
    opt_inputs = {param.name:param.default for param in parameters.values()}

    # cost function definition for least_squares optimizer
    def cost_fn0(initial_ATA):
        w, A, s = estimate_WAS(initial_ATA)
        output = np.abs([(w - width)/width, (A - amplitude)/amplitude, (s - sym)/sym])
        return output

    # set parameters and the final list for least_squares
    opt_inputs['fun']       = cost_fn0
    opt_inputs['x0']        = initial_ATA
    opt_inputs['bounds']    = bounds0
    opt_inputs['jac']       = '3-point'
    opt_inputs['xtol']      = 1e-15
    opt_inputs['ftol']      = 1e-15
    opt_inputs['gtol']      = 1e-15
    opt_inputs['max_nfev']  = 1000
    least_squares_dict      = [( least_squares, copy(opt_inputs) )]

    #### root dictionary for Bayesian opt ####
    root = scipy.optimize.root
    parameters = inspect.signature(root).parameters
    opt_inputs = {param.name:param.default for param in parameters.values()}

    # cost function definition for root optimizer
    def cost_fn1(initial_ATA):
        w, A, s = estimate_WAS(initial_ATA)
        output = np.array([(w - width)/width, (A - amplitude)/amplitude, (s - sym)/sym])
        return output

    # optimizer input parameters for root
    opt_inputs['fun']    = cost_fn1
    opt_inputs['x0']     = initial_ATA
    opt_inputs['tol']    = 1e-15

    # get that dict!!!
    root_method_list = [
        'hybr',
        'lm',
        'broyden1',
        'broyden2',
        'linearmixing',
        'excitingmixing'] # removed anderson method

    root_dict = []
    for method in root_method_list:
        temp_inputs = copy(opt_inputs)
        temp_inputs['method'] = method
        root_dict.append(( root, temp_inputs ))

    #### minimize dictionary for Bayesian opt ####
    minimize = scipy.optimize.minimize
    bounds1 = [(abun-4.0, abun+4.0), (T2-3.0, T2+3.0), (angle-1.0, angle+1.0)]
    parameters = inspect.signature(minimize).parameters
    opt_inputs = {param.name:param.default for param in parameters.values()}

    # cost function definition for minimize
    norm = partial(np.linalg.norm, ord=1) # ord=2 might be better...
    def cost_fn2(initial_ATA):
        w, A, s = estimate_WAS(initial_ATA)
        output = norm([(w - width)/width, (A - amplitude)/amplitude, (s - sym)/sym])
        return output

    # optimizer input parameters for minimize
    opt_inputs['fun'] = cost_fn2
    opt_inputs['x0']  = initial_ATA
    opt_inputs['tol'] = 1e-15

    # get that dict!!!
    minimize_method_list = [
        'Powell',
        'CG',
        'BFGS',
        'L-BFGS-B',
        'TNC',
        'COBYLA',
        'SLSQP',
        'Nelder-Mead'] # removed trust-constr

    no_bound_list = ['CG', 'BFGS']
    no_jac_list = ['Nelder-Mead','Powell','COBYLA']
    no_option_list = ['TNC']

    minimize_dict = []
    for method in minimize_method_list:
        temp_inputs = copy(opt_inputs)
        temp_inputs['method'] = method

        if method not in no_bound_list:
            temp_inputs['bounds'] = bounds1

        if method not in no_jac_list:
            temp_inputs['jac'] = '3-point'

        if method not in no_option_list:
            temp_inputs['options'] = {'maxiter': 1000, 'disp': False}

        minimize_dict.append(( minimize, temp_inputs ))

    # final method list for Bayesian optimization
    method_list = least_squares_dict + root_dict + minimize_dict
    ms_length = max([ len(s[1]['method'])  for s in method_list])

    # Bayesian objective function with the search space
    last_result = None # we want to store best result from optimizer
    best_cost = 1e+10
    space = {
        'scalar': hyperopt.hp.loguniform('scalar', 0, 35),
        'optimizer': hyperopt.hp.choice('optimizer', method_list)}

    def objective_fn(params):
        nonlocal last_result, best_cost # keep track of last result and best cost

        # params from search space
        scalar = params['scalar']
        optimizer, opt_inputs = params['optimizer']
        cost_fn = opt_inputs['fun']
        method =  opt_inputs['method'].rjust(ms_length)
        print(f"optimizer {method} with scalar {scalar}")

        # new loss fn with scalar still searching
        def cost_fn_scaled(initial_ATA):
            return scalar*cost_fn(initial_ATA)

        temp_inputs = copy(opt_inputs)
        temp_inputs['fun'] = cost_fn_scaled

        # with the rescaled loss fn get the result for Bayesian opt.
        try:
            last_result = optimizer(**temp_inputs)
        except IndexError:
            return np.random.uniform(0.1, 1.0) # punish this optimizer!!

        # the output for Bayesian opt must be scalar
        if isinstance(last_result.fun, np.ndarray):
            cost = np.linalg.norm(last_result.fun, ord=1)/scalar
        else:
            cost = last_result.fun/scalar

        # print out the new best method for sanity check
        if cost < best_cost:
            best_cost = cost
            print(f"new best method/scalar :{method}/{scalar}")

        return cost

    # let it Bayesian optimize!
    trials = hyperopt.Trials()
    max_evals=450
    opt_result = hyperopt.fmin(
        objective_fn,
        space,
        algo=hyperopt.tpe.suggest,
        max_evals=max_evals,
        loss_threshold=1e-15,
        trials=trials
    )

    # Bayesian optimization result
    scalar = opt_result['scalar']
    optimizer, opt_inputs = method_list[opt_result['optimizer']]
    print()
    print(f"Bayesian optimization done with:")
    print(f"optimizer -> {opt_inputs['method']}")
    print(f"scalar    -> {scalar}")
    print()

    # return the last result if optimization is done before max_evals
    if trials.best_trial['tid']+1 != max_evals: return last_result

    # in case Bayesian optimization evaluated until max_evals
    cost_fn = opt_inputs['fun']
    def cost_fn_scaled(initial_ATA):
        return scalar*cost_fn(initial_ATA)

    # optimize with scaled cost_fn
    opt_inputs['fun'] = cost_fn_scaled
    return optimizer(**opt_inputs)

class NMR_result():
    """
    NMR_result object contains nmr data processed with common_peaks (which outputs peaks 
    [location, width, amplitude, symmetricity]), corrected spectra, the original spectrum, 
    and number of common peaks for both the model and nmrglue processed data. To instantiate,
    you need to provide TF model path (hdf5 usually), sample path (directory with nmr data, 
    usually contains pdata directory), and database path (ATA to WAS database). If there is 
    at least one common peak, then ATA values are estimated for both the model and nmrglue
    peaks, which are then used to measure accuracy of the artifact correction measured by 
    the accuracy of peak amplitudes (original phase angle vs. zero phase angle).
    
    Attributes
    ----------
    peaks_model: list[ndarray]
        list of numpy arrays for all the identified peaks which contain the following info:
            locations : Peak locations
            scales    : Estimated peak scales (linewidths)
            amps      : Estimated peak amplitudes
            syms      : Estimated peak symmetricity
    peaks_ng: list[ndarray]
        same as peaks_model, but for nmrglue processed data
    spectrum_model: ndarray
        DNN model processed spectra
    spectrum_ng: ndarray
        nmrglue processsed spectra
    nmr_data: ndarray
        uncorrected spectrum (real part) DNN model processed    
    num_peaks: int
        number of common peaks
    ATA_model: list[ndarray]
        abundance (proton), T2, angle (phase shift) of peaks in model corrected spectra
    ATA_ng: list[ndarray]
        same as ATA_model, but for nmrglue processed spectrum
    accuracy_model: list[ndarray]
        this list contains accuracy of width and amplitude of model corrected peaks. The first
        element is for width, the second element is for peak amplitude. The unit is in percent
        (that is, 1.5 means 1.5% accuracy). The accuracy is measured against peak with zero
        phase shift angle. 
    accuracy_ng: list[ndarray]
        same as accuracy_model, but for nmrglue processed spectrum
    """
    def __init__(self, model_path, sample_path, database_path):
        # common peaks method and its output
        cp_outputs = common_peaks(model_path, sample_path)
        self.peaks_model    = cp_outputs[0]
        self.peaks_ng       = cp_outputs[1]
        self.spectrum_model = cp_outputs[2]
        self.spectrum_ng    = cp_outputs[3]
        self.nmr_data       = cp_outputs[4]
        self.num_peaks      = len(self.peaks_model)
        
        # no ATA values if no common peaks were found
        if len(self.peaks_model) == 0: return
        
        # estimate the ATA values for each peak
        self.ATA_model      = []
        self.ATA_ng         = []
        for LWAS_model, LWAS_ng in zip(self.peaks_model, self.peaks_ng):
            print("optimizing for DNN model....")
            self.ATA_model.append(estimate_ATA(LWAS_model[1:], database_path).x)
            print()
            
            print("optimizint for nmrglue model....")
            self.ATA_ng.append(estimate_ATA(LWAS_ng[1:], database_path).x)
            print()
            
        # get accuracy of width and amplitude for both model and nmrglue
        self.accuracy_model = []
        self.accuracy_ng = []
        for ind in range(self.num_peaks):
            ATA, WAS = self.ATA_model[ind], self.peaks_model[ind][1:]
            self.accuracy_model.append( self.accuracy(ATA, WAS) )
            
            ATA, WAS = self.ATA_ng[ind], self.peaks_ng[ind][1:]
            self.accuracy_ng.append( self.accuracy(ATA, WAS) )

    def accuracy(self, ATA_values, WAS_values):
        """
        returns accuracy of width and amplitude for the given ATA estimate
        """
        # get the target spectrum (zero phase shift angle)
        a, T2, angle = ATA_values
        spectrum_target = spectrum_gen(a, T2, 0.0)

        # get the target peak
        _,_,w,A = ng.analysis.peakpick.pick(spectrum_target, pthres=0.2)[0]

        # get accuracies of width and amplitude
        accuracy = np.array([
            100*np.abs(w-WAS_values[0])/w,
            100*np.abs(A-WAS_values[1])/A])
        
        return accuracy
            
    def WAS_list(self):
        """
        returns WAS values of model and nmrglue results
        """
        pzip = zip(self.peaks_model, self.peaks_ng)
        return [(lwasm[1:], lwasn[1:]) for (lwasm, lwasn) in pzip]
    
    def ATA_list(self):
        """
        returns ATA values of model and nmrglue results
        """
        azip = zip(self.ATA_model, self.ATA_ng)
        return [(atam, atan) for (atam, atan) in azip]

    def accuracy_list(self):
        """
        returns accuracy of model and nmrglue results
        """        
        azip = zip(self.accuracy_model, self.accuracy_ng)
        return [(acm, acn) for (acm, acn) in azip]
    
    def __repr__(self):
        # in case there is no common peaks
        if len(self.peaks_model) == 0: return "no common peaks were found; no result"
    
        lzip = zip(self.WAS_list(), self.ATA_list(), self.accuracy_list())
        output = "NMR_result(\n"
        output += "[the first element is for DNN model, the second element is for nmrglue]\n"

        for ind, (w, a, ac) in enumerate(lzip):
            output += f"  --- the {ind+1}-th peak ---\n"
            output += f"  WAS_values : {w}\n"
            output += f"  ATA_values : {a}\n"
            output += f"  accuracy   : {ac}\n"
            output += "\n"
        output += ")\n"

        return output
