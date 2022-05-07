# generates x values for sinusoidal function that results in equidistant points on the graph
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

def xarray(
        A=1.0,
        w=1.0,
        theta=0.0,
        x_min=0.0,
        x_max=2*np.pi,
        N=101):
    """
    generates x values for sinusoidal function that results in equidistant points on the graph

    Parameters
    ----------
    A: float
        Amplitude of the sine function (default 1.0)
    w: float
        Angular frequency of the sine function (default 1.0)
    theta: float
        Additional phase factor for sine argument (default 0.0)
    x_min: float
        Very first value of the output (default 0.0)
    x_max: float
        Very last value of the output (default 2*pi)
    N: int
        Number of points (output size) on the graph (default 101)

    Returns
    -------
    xArray: Numpy Array[float]
        x values for sine function that give equidistant points on the graph
    """

    # integrand for arch length
    def integrand(x, A, w, theta):
        return np.sqrt(pow( A*w*np.cos(w*x + theta) , 2) + 1)
    
    # total arc length and sub_arc size
    total_arc = integrate.quad(integrand, x_min, x_max, args=(A, w, theta))[0]
    sub_arc = total_arc/(N-1)
    
    # function that determines the upper limit value of sub_arc by applying optimize.fsolve
    # optimize.fsolve finds a root of the function
    # a and b are lower and upper bounds of the integral
    def step_comp(b, a, integrand, A, w, theta, sub_arc):
        return sub_arc - integrate.quad(integrand, a, b, args=(A, w, theta))[0]

    # create xarrays for equidistant points
    xarray = np.zeros(N)
    x_prev = x_min
    for n in range(N):
        xarray[n] = x_prev
        args=(x_prev, integrand, A, w, theta, sub_arc)
        x_prev = optimize.fsolve(step_comp, x_prev, args=args)[0]

    # rescale xarray to offset errors and returns it
    return (x_max/xarray[-1])*xarray

