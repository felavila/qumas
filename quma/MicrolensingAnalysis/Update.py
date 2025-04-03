import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button,RangeSlider
from lmfit import Model
from scipy.integrate import quad
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import os
import multiprocessing
from functools import partial
from .utils import format_scientific
from screeninfo import get_monitors
from .functions  import linear_fit




def update(Lp,Lr,Wslider_lc,Wslider_core,Wslider_rc,Wslider_Lp,Wslider_Rp,Fslider_Lp,Fslider_Rp, Y):
    
    Wslider_Lp_min, Wslider_Lp_max = Wslider_Lp.val[0], Wslider_Lp.val[1]
    Lp.set_xlim(Wslider_Lp_min,Wslider_Lp_max)
    plt.draw()


    
#     global slider_barrier12,barrier_values,y_curve,y_noisy,x_between_barriers,y_noisy_between_barriers,range_values  #barrier_values, y_curve, y_noisy_between_barriers
#     # Extract barrier positions from sliders
#     x_barrier1, x_barrier2 = slider_barrier12.val[0], slider_barrier12.val[1]
#     x_barrier3, x_barrier4 = slider_barrier34.val[0], slider_barrier34.val[1]
#     x_barrier5, x_barrier6 = slider_barrier56.val[0], slider_barrier56.val[1]

#     # Additional barrier positions
#     x_slider_range, y_slider_range = slider_range.val, slider_range_y.val
#     x_slider_range2, y_slider_range2 = slider_range2.val, slider_range_y2.val

#     # Combine the data between barriers 1 and 2, and between barriers 3 and 4
#     barrier_values = [x_barrier1, x_barrier2, x_barrier3, x_barrier4, x_barrier5, x_barrier6]
#     range_values = [ *slider_range.val, *slider_range_y.val,*slider_range2.val, *slider_range_y2.val]
#     x_between_barriers = np.concatenate((x[(x >= x_barrier1) & (x <= x_barrier2)], x[(x >= x_barrier3) & (x <= x_barrier4)]))
    
#     ax1_1.clear()
#     ax1_2.clear()
    
#     for key, y_noisy in y_noisy_list.items():
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
#         # Display the area as text inside the plot
#     # Plot the fitted linear function   
    
#     ax1_1.axvline(x=line_info["line_center"], color='k', label='Line',linestyle ="--")
#     ax1_1.fill_betweenx([-100,100], x_barrier1, x_barrier2,
#                         label='left continium', color='purple', alpha=0.2)

#     # Create a region between x_barrier3 and x_barrier4
#     ax1_1.fill_betweenx([-100,100], x_barrier3, x_barrier4,
#                         label='right continium', color='g', alpha=0.2)

# # Rest of the code remains the same up to this point

#     ax1_1.legend(fontsize='xx-small', frameon=False,ncol=3)
#     ax1_1.set_xlim(x_slider_range[0], x_slider_range[1])
#     ax1_1.set_ylim(y_slider_range[0], y_slider_range[1])
#     ax1_1.set_xlabel('wavelength')
#     ax1_1.set_ylabel('Flux')
    
#     ax1_2.fill_between(x[(x_barrier5 <= x) & (x_barrier6 >= x)], 
#                    -100,100, 
#                    label='line core region', color='r', alpha=0.3, zorder=2)
#     #ax1_2.axvline(x=x_barrier5, color='m', linestyle='--', label='Barrier 5')
#     #ax1_2.axvline(x=x_barrier6, color='c', linestyle='--', label='Barrier 6')
#     ax1_2.axvline(x=line_info["line_center"], color='k', label='Line',linestyle ="--")
#     ax1_2.set_xlim(x_slider_range2[0], x_slider_range2[1])
#     ax1_2.set_ylim(y_slider_range2[0], y_slider_range2[1])
#     ax1_2.set_xlabel('wavelenght')
#     ax1_2.set_ylabel('Flux')
#     ax1_2.legend(fontsize='small', frameon=False,ncol=3)
    
#     plt.draw()