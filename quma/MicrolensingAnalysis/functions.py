import numpy as np
from lmfit import Model
from scipy.integrate import quad
import warnings

import multiprocessing
from functools import partial

warnings.simplefilter(action='ignore', category=FutureWarning)

def linear_func(x, slope, intercept):
        return slope * x + intercept

def linear_model(x,y):
    linear_fit_model = Model(linear_func)
    params = linear_fit_model.make_params(slope=1, intercept=0)
    result = linear_fit_model.fit(y,x=x, params=params)
    slope_fit, intercept_fit = result.params['slope'].value, result.params['intercept'].value
    return slope_fit, intercept_fit


#    for key, y_noisy in y_noisy_list.items():
#         y_noisy_between_barriers = np.concatenate((y_noisy[(x >= x_barrier1) & (x <= x_barrier2)], y_noisy[(x >= x_barrier3) & (x <= x_barrier4)]))
#         ax1_1.scatter(x_between_barriers, y_noisy_between_barriers, alpha=1, label=f'Data (between limits) imagen {key}', zorder=3) #pensar en colores
#         ax1_1.plot(X,Y[key],label=f'spectrum imagen {key}', alpha=0.6, zorder=1)
#         # Fit a single linear model to the data between the barriers
#         linear_fit_model = Model(linear_fit)
#         params = linear_fit_model.make_params(slope=1, intercept=0)
#         result = linear_fit_model.fit(y_noisy_between_barriers, x=x_between_barriers, params=params)
#         slope_fit, intercept_fit = result.params['slope'].value, result.params['intercept'].value
#         ax1_1.plot(x_between_barriers, linear_fit(x_between_barriers, slope_fit, intercept_fit), label=f'Fitted Linear Function for {key}', color='r', zorder=4)
        
#         # Calculate the area under the linear function between Barrier 1 and Barrier 4
#         area, _ = quad(linear_fit, x_barrier1, x_barrier4, args=(slope_fit, intercept_fit))

#         # Calculate the curve after subtracting the area obtained from the fit
#         y_curve = y_noisy - linear_fit(x, slope_fit, intercept_fit)
#         Y_curve = Y[key] - linear_fit(X, slope_fit, intercept_fit)
#         suma = np.sum(y_curve[(x_barrier5 <= x) & (x_barrier6 >= x)])

#         # Plot the curve resulting from subtracting the area
        
#         ax1_2.plot(X, Y_curve, label=f'Line without continium {key}', color='grey',alpha=0.5)
#         ax1_2.hlines(y=[0, 0], xmin=min(x), xmax=max(x), colors='k', linestyles='dashed', zorder=3)

def bootstrap_single_iteration(x, y, _):
    pto_medio=np.min(x)+(np.max(x)-np.min(x))/2
    if isinstance(x,np.ndarray):
        x_L,y_L=x[x<pto_medio],y[x<pto_medio]
        x_R,y_R=x[x>pto_medio],y[x>pto_medio]
    else:
        x_L,y_L=x.values[x<pto_medio],y.values[x<pto_medio]
        x_R,y_R=x.values[x>pto_medio],y.values[x>pto_medio]
    # Resample the dataset with replacement
    resampled_indices_R = np.random.choice(len(x_R), len(x_R), replace=True)
    resampled_x_R = x_R[resampled_indices_R]
    resampled_y_R = y_R[resampled_indices_R]
   
    resampled_indices_L = np.random.choice(len(x_L), len(x_L), replace=True)

    resampled_x_L = x_L[resampled_indices_L]
    resampled_y_L = y_L[resampled_indices_L]
    
    model = Model(linear_fit)
    params = model.make_params(slope=1, intercept=0)
    resampled_x=np.concatenate((resampled_x_L, resampled_x_R))
    
    resampled_y=np.concatenate((resampled_y_L, resampled_y_R))
    result = model.fit(resampled_y, params, x=resampled_x)

    fitted_m = result.params["slope"].value
    fitted_b = result.params["intercept"].value

    
    x_min = min(resampled_x)
    x_max = max(resampled_x)

    area = (fitted_m * (x_max**2 / 2) + fitted_b * x_max) - (fitted_m * (x_min**2 / 2) + fitted_b * x_min)
    return fitted_m, fitted_b, area

def bootstrap_Area(x, y, n_bootstrap=5000):
    ncpu = multiprocessing.cpu_count()
    bootstrap_func = partial(bootstrap_single_iteration, x, y)

    with multiprocessing.Pool(ncpu) as pool:
        results = pool.map(bootstrap_func, range(n_bootstrap))

    slope_fit_estimates, intercept_estimates, area_estimates = zip(*results)
    mean_slope_fit_estimates,std_slope_fit_estimates = np.median(slope_fit_estimates),np.std(slope_fit_estimates)
    mean_intercept_estimates,std_intercept_estimates = np.median(intercept_estimates),np.std(intercept_estimates)
    mean_area,std_area = np.median(area_estimates),np.std(area_estimates)
    
    
    return mean_slope_fit_estimates,std_slope_fit_estimates,mean_intercept_estimates,std_intercept_estimates,mean_area,std_area, slope_fit_estimates, intercept_estimates, area_estimates