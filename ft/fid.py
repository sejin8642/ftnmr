# simple free induction decay code
import numpy as np

class fid():
    def __init__(self, B=1.5, T2=2000, timeunit='msec'):
        pass
        self.B = B #approximately 2000 msec is T2 for water/CSF at 1.5T
        self.t = np.linspace(1, 6*T2, 6*T2)
        self.T2 = T2
        self.gamma = 267.52219*pow(10,6)
        if timeunit == 'msec':
            self.w = 0.001*self.gamma*self.B
        elif timeunit == 'micron':
            self.w = 0.000001*self.gamma*self.B
        else:
            raise Exception('Incorrect time unit is specified')

    def __call__(self):
        return np.exp(-self.t/self.T2)*np.cos(self.t)

