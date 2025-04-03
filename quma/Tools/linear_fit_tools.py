import numpy as np
import scipy.odr as odr
from matplotlib.table import Table
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

def linear_func(params, x):
    m, b = params
    return m * x + b
def evaluate_with_uncertainties(x_new, x_new_err, slope, intercept, slope_err, intercept_err):
    # Predicted y value
    y_new = slope * x_new + intercept
    
    # Uncertainty in predicted y value (including uncertainty in x_new)
    y_err_new = np.sqrt(
        (x_new * slope_err)**2 +              # Uncertainty due to slope
        intercept_err**2 +                    # Uncertainty due to intercept
        (slope * x_new_err)**2                # Uncertainty due to x_new
    )
    
    return np.array([y_new, y_err_new])
def make_linear_fit(x,y,y_error=None,x_error=None,plot=False,names=[],title="title",save="",add_table=False,xlabel=r"$r_{s}$(thin disk) [lday]",ylabel=r"$r_{s}$(microlensing) [lday]"):
    linear_model = odr.Model(linear_func)
    data = odr.RealData(x, y, sx=x_error, sy=y_error)
    # Set initial guesses for slope (m) and intercept (b)
    beta0 = [1.0, 1.0]  # initial guess for [slope, intercept]
    # Create ODR object and run the fitting
    odr_fit = odr.ODR(data, linear_model, beta0=beta0)
    output = odr_fit.run()
    # Print the results
    slope, intercept = output.beta
    slope_err, intercept_err = output.sd_beta
    print(f"Slope: {slope} ± {slope_err}")
    print(f"Intercept: {intercept} ± {intercept_err}")
    # Generate the fit line
    mfc = "k"
    if add_table:
        mfc = 'white'
    if plot:
        x_fit = np.linspace(min(x)*0.01, max(x)*1.2, 100)
        y_fit = linear_func((slope, intercept),x_fit)

        # Propagate uncertainty across the entire fit line
        y_fit_upper = linear_func((slope + slope_err, intercept + intercept_err),x_fit)
        y_fit_lower = linear_func((slope - slope_err, intercept - intercept_err),x_fit)

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.errorbar(x, y, yerr=y_error, xerr=x_error, fmt='o', markersize=10, mfc=mfc, mec='k', capsize=5,ecolor="k")
        #.errorbar(x, y,xerr=x_error,yerr=y_error, fmt='o',c="k", capsize=2, s=500, facecolors='none', edgecolors='blue')
        ax.plot(x_fit,y_fit, linewidth=1, color='r', label=r"Linear regression", ls='--', zorder=0)
        ax.fill_between(x_fit,y_fit_lower,y_fit_upper,
                                color="red",label=r'$\sigma$ uncertainty',alpha=0.3)
        
        if add_table:
            if len(names) ==0:
                names = [str(i+1) for i in range(len(x))]
            
            names = {i+1:name for i,name in enumerate(names)}
            for i, (num,name) in enumerate(names.items()):
                ax.text(x[i], y[i], str(num), fontsize=8, ha='center', va='center_baseline',zorder=10)
            # Create a table for the definitions
            table_data = [[str(num), fr"{definition}|$r_{{td}}=${x[num-1]:.4f} $\pm$ {x_error[num-1]:.4f} vs $r_{{m}}=${y[num-1]:.4f} $\pm$ {y_error[num-1]:.4f}"] for num, definition in zip(names.keys(), names.values())]

            # Add the table on the side
            
            table = Table(ax, bbox=[1.1, 0., 0.5, 1])  # Adjust bbox to control position and size of the table
            for i, (num, definition) in enumerate(table_data):
                table.add_cell(i, 0, width=0.2, height=0.1, text=num, loc='center')
                table.add_cell(i, 1, width=0.5, height=0.1, text=definition.split("|")[0], loc='left')
                table.add_cell(i, 2, width=1.5, height=0.1, text=definition.split("|")[1], loc='left')

            # Add the table to the plot
            ax.add_table(table)

            # Show plot 
            #ax.plot([12,17], [12,17], color='grey', label=r'$R_s=R_s$',alpha=0.8,ls='--',zorder=3)
            #plt.xlim(12.6,16)
            #plt.ylim(14,17.5)
            #plt.text(15,13,size=20)
        #
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(fontsize=20)
        ax.set_xlim(x_fit[[0,-1]])
        #ax.set_ylim(min(y_fit)*0.95, max(y_fit)*1.1)
        ax.set_title(title)
        plt.tight_layout()
        if len(save)>0:
            plt.savefig(f"{save}.png",bbox_inches='tight')
        plt.show()
    return slope, intercept, slope_err, intercept_err