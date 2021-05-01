# simple free induction decay code
# References
"""
Keeler pg.49 for general signal form
proton gyromagnetic ratio: https://physics.nist.gov/cgi-bin/cuu/Value?gammap
"""
import numpy as np

class fid():
    def __init__(self, B=1.5, shift=0, T2=2000, timeunit='msec'):
        self.B = B #approximately 2000 msec is T2 for water/CSF at 1.5T
        self.shift = shift
        self.T2 = T2
        self.gamma = 267.52218744*pow(10,6)

        # Signal frequency is adjusted according to the chemical shift
        if timeunit == 'msec':
            self.w = pow(10, -9)*self.shift*self.gamma*self.B
        elif timeunit == 'micron':
            self.w = pow(10, -12)*self.shift*self.gamma*self.B
        else:
            raise Exception('Incorrect time unit is specified')

        # Below sampling rate samples at least 20 in one period of larmor precession
        self.SR = round(0.31415/self.w, int(-np.log10(0.31415/self.w)) + 1)
        self.t = np.linspace(0, 6*T2 - self.SR, int(6*T2/self.SR))

    def __call__(self):
        return np.exp(-self.t/self.T2)*(np.cos(self.w*self.t) + 1j*np.sin(self.w*self.t))
