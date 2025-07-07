from uncertainties.core import AffineScalarFunc
from astropy.constants import c
from astropy.units import cm, day
from uncertainties import unumpy,ufloat
import numpy as np

def montecarlo_errors(x_data,x_error,N=10000):
    medians = []
    for _ in range(N):
    # Resample each data point from its normal distribution
        x_sampled = np.random.normal(x_data, x_error)
    # Compute the median of the resampled data
        medians.append(np.median(x_sampled))

    # Calculate the overall median and its uncertainty
    median_estimate = np.median(medians)
    median_uncertainty = np.std(medians)

    print(fr"Median = {median_estimate:.3f} $\pm$ {median_uncertainty:.3f}")
    return median_estimate,median_uncertainty


def lday_to_cm(lday_value):
    # Speed of light in cm/s
    speed_of_light_cm_per_s = 2.998e10
    # Number of seconds in a day
    seconds_per_day = 86400
    
    # Convert light-days to cm
    cm_value = lday_value * speed_of_light_cm_per_s * seconds_per_day
    return cm_value
def cm_to_lday(cm_value):
    # Convert cm to meters
    meters = cm_value * cm.to('m')
    # Convert meters to light days
    light_days = meters / (c.to('m/day').value)
    return light_days

def RxtoR2500(Rx,x,Rx_in_log=True,ref=2500):
    """
    Rx radio at x lambda
    x lambda in A
    ref is the ref labda use to to be will be the radius
    
    """
    if Rx_in_log:
        Rx=10**(Rx)
    return np.log10(Rx*((ref/x)**(4/3)))
def log10_to_number(rx):
    """_summary_

    Args:
        rx (_type_): rx is a tuple (rx,sigma_rx) log10

    Returns:
        _type_: numpy array(2,) with rx and sigma_rx non log10
    """
    return to_return(10**(unumpy.uarray(*rx)))

def r_x_to_r_y(rx,x,y=2500,log10=False):
    """
    rx: tuple (rx,sigma_rx) at x lambda with error
    x: lambda in A at which rx is measured
    y: lambda in A at which ry will be re-scaled
    return and (2,len(rx)) array with ry and error
    
    """
    if log10:
        return to_return(unumpy.log10(unumpy.uarray(*rx)*(y/x)**(4/3)))
    else:
        return to_return(unumpy.uarray(*rx)*(y/x)**(4/3))

def to_return(f):
    if isinstance(f,AffineScalarFunc):
        f= np.array([f.nominal_value,f.s])
    elif len(f.shape)>1:
        f= np.array([[i.nominal_value,i.s] for i in f[0].T])
    elif len(f.shape)==1:
        f= np.array([[i.nominal_value,i.s] for i in f])
    else:
        f = f.reshape(1,)[0]
        f= np.array([f.nominal_value,f.s])
    return f