# simple free induction decay code
# References
"""
Keeler pg.49 for general signal form
proton gyromagnetic ratio: https://physics.nist.gov/cgi-bin/cuu/Value?gammap
"""
import numpy as np

class fid():
    def __init__(self, timeunit='msec', B=1.5, shift=0, T2=2000, sip=40):
        self.B = B #approximately 2000 msec is T2 for water/CSF at 1.5T
        self.shift = shift
        self.T2 = T2
        self.gamma = 267.52218744*pow(10,6)
        # the lowest angular frequency used as a chemical shift reference
        self.ref_w = self.B*self.gamma

        # Signal frequency is adjusted according to the chemical shift
        if timeunit == 'msec':
            self.w = pow(10, -9)*self.shift*self.ref_w
        elif timeunit == 'micron':
            self.w = pow(10, -12)*self.shift*self.ref_w
        else:
            raise Exception('Incorrect time unit is specified')

        # sampling rate and independent time variables (sample duration ~ 6*T2)
        if self.shift == 0: # 103 samples total when there is no oscillation
            self.t = np.linspace(0, 6*T2, 6*17 + 1)
            self.SR = 17/T2
        else:
            self.SR = 0.5*sip*self.w/np.pi
            ns = int(6*T2*self.SR) + 1 # total number of samples including t = 0 
            self.t = np.linspace(0, (ns - 1)/self.SR , ns)

    def __call__(self):
        return np.exp(-self.t/self.T2)*(np.cos(self.w*self.t) + 1j*np.sin(self.w*self.t))
