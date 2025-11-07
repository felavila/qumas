import numpy as np 
import os 
import pandas as pd 
import warnings
import json
import matplotlib.pyplot as plt 
from copy import deepcopy

from scipy.integrate import quad
from matplotlib.widgets import Button,RangeSlider,Slider
import pickle
from mpl_toolkits import axisartist
import ipywidgets as widgets
from IPython.display import display, clear_output
from qumas.MicrolensingAnalysis.Utils.emission_line import _compute_metrics_for_image
from qumas.MicrolensingAnalysis.Utils.continuum import _compute_metrics_for_image_continuum
from qumas.MicrolensingAnalysis.Utils.extras import calculate_line,convert_none_to_nan





module_dir = os.path.dirname(os.path.abspath(__file__))
windows_rest_frame = os.path.join(module_dir,"rest_frame_windows","rest_frame_windows.json")
with open(windows_rest_frame, "r") as f:
    windows_rest_frame = json.load(f)
def _close_existing_figures():
    import matplotlib.pyplot as plt
    try:
        # Close all known figures cleanly
        from matplotlib._pylab_helpers import Gcf
        for manager in list(Gcf.get_all_fig_managers()):
            try:
                plt.close(manager.canvas.figure)
            except Exception:
                pass
    except Exception:
        # Fallback if helper import changes between matplotlib versions
        plt.close('all')






def build_limits(spectra_dict, row):
    line_name = row.line_name.values[0]

    center_window = float(np.mean(row["core_range"].values[0]))
    window = [center_window - 500.0, center_window + 500.0]

    # Band wavelength limits from spectra_dict
    band_limits = {
        band: np.array(spectra_dict[band]["wavelength"])[0][[0, -1]]
        for band in spectra_dict.keys()
    }

    right_window_init = list(row["right_range"].values[0])
    left_window_init  = list(row["left_range"].values[0])
    core_init         = list(row["core_range"].values[0])

    # ------------------------------------------------------------------
    # Choose best band (as you had)
    # ------------------------------------------------------------------
    wl, wr = float(min(window)), float(max(window))
    w_center = 0.5 * (wl + wr)

    candidates = []
    for name, (b_min, b_max) in band_limits.items():
        b_min, b_max = float(b_min), float(b_max)
        b_center = 0.5 * (b_min + b_max)
        b_width  = b_max - b_min

        full_cover = (b_min <= wl) and (b_max >= wr)
        overlap = max(0.0, min(b_max, wr) - max(b_min, wl))
        center_dist = abs(b_center - w_center)

        candidates.append({
            "name": name,
            "full_cover": full_cover,
            "overlap": overlap,
            "center_dist": center_dist,
            "band_width": b_width,
        })

    full = [c for c in candidates if c["full_cover"]]
    pool = full if full else candidates

    pool_sorted = sorted(
        pool,
        key=lambda c: (
            0 if c["full_cover"] else 1,
            -c["overlap"],
            c["center_dist"],
            c["band_width"],
        )
    )
    band = pool_sorted[0]["name"]

    # ------------------------------------------------------------------
    # Get X, Y for the chosen band and compute band min/max
    # ------------------------------------------------------------------
    X = np.array(spectra_dict[band]["wavelength"])
    Y = np.array(spectra_dict[band]["flux"])
    OBJS = spectra_dict[band]["obj"]

    # X is usually 2D: (nobj, npix), so use global min/max
    x_min = float(np.min(X))
    x_max = float(np.max(X))

    # ------------------------------------------------------------------
    # Continuum-only lines: build windows around core and then clip them
    # ------------------------------------------------------------------
    if "cont" in line_name:
        core_min = float(min(core_init))
        core_max = float(max(core_init))
        right_window_init = [core_max, core_max + 100.0]
        left_window_init  = [core_min - 100.0, core_min]
        # print(right_window_init, left_window_init, core_init)

    # ------------------------------------------------------------------
    # Helper to clip any [lo, hi] window to [x_min, x_max]
    # ------------------------------------------------------------------
    def _clip_window(win, lo=x_min, hi=x_max):
        a, b = float(win[0]), float(win[1])
        # enforce ordering
        if b < a:
            a, b = b, a
        # clip to data range
        a = max(a, lo)
        b = min(b, hi)
        # if everything was outside and b < a after clipping, collapse to boundary
        if b < a:
            a = b = lo
        return [a, b]

    # Clip all windows to wavelength coverage
    left_window_init  = _clip_window(left_window_init)
    right_window_init = _clip_window(right_window_init)
    core_init         = _clip_window(core_init)

    # Optionally: clip center_window itself if you want it inside X range too
    center_window = float(np.clip(center_window, x_min, x_max))

    print(right_window_init, left_window_init, core_init, line_name)

    return {
        "X": X,
        "Y": Y,
        "objs": OBJS,
        "center_window": center_window,
        "right_window_init": right_window_init,
        "left_window_init": left_window_init,
        "core_init": core_init,
        "band": band,
        "line_name": line_name,
    }
        
class WindowAnalysis:
    def __init__(self,results,zs=0.0,rest_frame=True,save_name = None,obj_name=None,path_previous_results=None):
        """_summary_

        Args:
            results (_type_): _description_
            zs (_type_, optional): _description_. Defaults to None.
            rest_frame (bool, optional): _description_. Defaults to True.
        """
        assert isinstance(zs,float) or isinstance(zs,int) , "zs have to be float or int"
        self.zs = zs
        self.save_name = save_name
        #self._previous_results =  self._read_previous_results(path_previous_results)
        
        if not self.save_name:
            self.save_name = "flux_cont_core.csv"
        if self.zs == 0:
            print("Warning: zs set to default value 0.0.")
        self.pre_define_windows = convert_none_to_nan(windows_rest_frame)
        if not rest_frame:
            for i in [ "left_range","right_range","core_range"]:
                self.pre_define_windows[i] = (np.array(self.pre_define_windows[i])*(1+self.zs)).tolist()
        self.pre_define_windows = pd.DataFrame(self.pre_define_windows)
        if isinstance(results,str):
            print("WORK IN PROGRESS")
        elif isinstance(results,dict):
            results = deepcopy(results)
            band_list = []
            self.spectra_dict = {}
            for obj,value in results.items():
                if "G" in obj:
                    print("We will discard the spectra's galaxy ")
                    continue
                band = value.get("band",obj.split("_")[1])
                band_list.append(band)
                if band not in self.spectra_dict.keys():
                    self.spectra_dict[band] = {}
                    for i in ["wavelength","flux","error","obj"]:
                        self.spectra_dict[band][i] = []
                wavelength = value.get("wavelength")
                if rest_frame:
                    wavelength = wavelength/(1+self.zs)
                self.spectra_dict[band]["wavelength"].append(wavelength)
                flux = value.get("flux")
                self.spectra_dict[band]["flux"].append(flux)
                self.spectra_dict[band]["error"].append(value.get("error",np.ones_like(flux)))
                self.spectra_dict[band]["obj"].append(obj)
                
            self.kwargs_h ={}

        self.band_limits = {band: np.array(self.spectra_dict[band]["wavelength"])[0][[0,-1]] for band in self.spectra_dict.keys()}
        
        for number,line_name in enumerate(self.pre_define_windows["line_name"].values):
            row = self.pre_define_windows[self.pre_define_windows["line_name"]==line_name]
            self.kwargs_h[number] = build_limits(self.spectra_dict,row)

    
    
    def _interactive_microlensing(self,n_bootstrap=5_000,random_state=0,figsize= (15, 5)):
        _close_existing_figures()
        output = widgets.Output()
        n_max = len(self.kwargs_h.keys())
        current_index = [0]
        
        # Initialize as class attribute if it doesn't exist
        if not hasattr(self, 'saved_parameters'):
            self.saved_parameters = {}
        
        # Flag to control the routine
        stop_routine = [False]
        
        def save_parameters(index, slider_left_window, slider_right_window, slider_line_core,
                        slider_xlim_left, slider_ylim_left, slider_xlim_right, slider_ylim_right,
                        line_name, band, objs,X,Y,n_bootstrap=n_bootstrap,random_state=random_state):
            """Save current slider values and parameters"""
            self.saved_parameters[index] = {'line_name': line_name,'band': band,'objects': objs,
                'left_window': slider_left_window.value,'right_window': slider_right_window.value,'line_core': slider_line_core.value,
                'xlim_left': slider_xlim_left.value,'ylim_left': slider_ylim_left.value,'xlim_right': slider_xlim_right.value,'ylim_right': slider_ylim_right.value,}
            
            if "cont" not in line_name:
                _numerics = _compute_metrics_for_image(X,Y,slider_left_window.value,slider_right_window.value,slider_line_core.value,n_bootstrap=n_bootstrap,random_state=random_state)
            else:
                _numerics = _compute_metrics_for_image_continuum(X,Y,slider_line_core.value,n_bootstrap=n_bootstrap,random_state=random_state)
            
            self.saved_parameters[index].update(_numerics)
            # Also update the kwargs_h with current values
            self.kwargs_h[index]['right_window_init'] = slider_right_window.value
            self.kwargs_h[index]['left_window_init'] = slider_left_window.value
            self.kwargs_h[index]['core_init'] = slider_line_core.value
            
            #return self.saved_parameters[index]
        
        def export_to_csv(filename=None):
            """Export saved parameters to CSV"""
            if not self.saved_parameters:
                print("No parameters to export!")
                return
            
            if filename is None:
                filename = f"microlensing_params_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Flatten the nested dictionary
            rows = []
            for idx, params in self.saved_parameters.items():
                row = {
                    'plot_index': idx,
                    'line_name': params['line_name'],
                    'band': params['band'],
                    'objects': ','.join(params['objects']) if isinstance(params['objects'], list) else params['objects'],
                    'left_window_min': params['left_window'][0],
                    'left_window_max': params['left_window'][1],
                    'right_window_min': params['right_window'][0],
                    'right_window_max': params['right_window'][1],
                    'line_core_min': params['line_core'][0],
                    'line_core_max': params['line_core'][1],
                    'xlim_left_min': params['xlim_left'][0],
                    'xlim_left_max': params['xlim_left'][1],
                    'ylim_left_min': params['ylim_left'][0],
                    'ylim_left_max': params['ylim_left'][1],
                    'xlim_right_min': params['xlim_right'][0],
                    'xlim_right_max': params['xlim_right'][1],
                    'ylim_right_min': params['ylim_right'][0],
                    'ylim_right_max': params['ylim_right'][1],
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)
            print(f"Parameters exported to: {filename}")
            #return filename
        
        def export_to_pickle(filename=None):
            """Export saved parameters to pickle"""
            if not self.saved_parameters:
                print("No parameters to export!")
                return
            
            if filename is None:
                filename = f"microlensing_params_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            with open(filename, 'wb') as f:
                pickle.dump(self.saved_parameters, f)
            print(f"Parameters exported to: {filename}")
            #return filename
       
        def show_plot(index):
            # Check if routine should stop
            if stop_routine[0]:
                print("Routine stopped by user.")
                return
            
            X, Y_original, objs, center_window, right_window_init, left_window_init, core_init, band, line_name  = self.kwargs_h[index].values()
           
            
            with output:
                clear_output(wait=True)
                _close_existing_figures()
                fig = plt.figure(figsize=figsize, layout="constrained")
                fig.canvas.draw()
                
                gs = fig.add_gridspec(1, 2, width_ratios=None, height_ratios=None, hspace=0.2)
                Leftplot = fig.add_subplot(gs[0, 0], axes_class=axisartist.Axes)
                Rightplot = fig.add_subplot(gs[0, 1], axes_class=axisartist.Axes)
                
                Rightplot.axis["right"].toggle(all=True)
                #Rightplot.axis["left"].set_visible(True)
                Rightplot.axis["left"].major_ticks.set_ticksize(0)     # hide tick marks
                Rightplot.axis["left"].major_ticklabels.set_visible(False)  # hide tick labels
                Rightplot.axis["left"].label.set_visible(False)
                ##############set-up-basic-plot##############
                Y_factor = int(np.log10(np.nanmedian(Y_original)))
                Y = Y_original*10**-Y_factor
                lines_left = []
                if Y.ndim == 2:
                    for n, (x, y) in enumerate(zip(X, Y)):
                        line, = Leftplot.plot(x, y, alpha=0.7, label=f'Object {objs[n]}')
                        lines_left.append(line)
                    Leftplot.legend()
                else:
                    line, = Leftplot.plot(X, Y)
                    lines_left.append(line)
                ##############set-labels-and-titles##############
                Leftplot.set_title(f"Emission region with local continuum")#{index+1}/{n_max}")
                Rightplot.set_title(f"Emission region withouth local continuum")#{index+1}/{n_max}")
                Leftplot.set_xlabel(r'$\rm Rest \ Wavelength$ ($\rm \AA$)', fontsize=20)
                Rightplot.set_xlabel(r'$\rm Rest \ Wavelength$ ($\rm \AA$)', fontsize=20)
                Leftplot.set_ylabel(r"F$_\lambda\,(\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}) \times$"+f"1e{Y_factor}", fontsize=20)
                Rightplot.set_ylabel(r"F$_\lambda\,(\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}) \times$"+f"1e{Y_factor}", fontsize=20)
                
                ##############set-slider##############
                if "cont" not in line_name:
                    slider_left_window = widgets.FloatRangeSlider(value=left_window_init, min=np.min(X), max=np.max(core_init), step=0.1,
                        description='Left window', layout=widgets.Layout(width='100%'),
                        style={'description_width': '120px'})
                    
                    slider_right_window = widgets.FloatRangeSlider(
                        value=right_window_init, min=np.min(core_init), max=np.max(X), step=0.1,
                        description='Right window', layout=widgets.Layout(width='100%'),
                        style={'description_width': '120px'})
                
                
                slider_line_core = widgets.FloatRangeSlider(
                    value=core_init, min=min(core_init)-100, max= max(core_init)+100, step=0.1,
                    description='Line core', layout=widgets.Layout(width='100%'),
                    style={'description_width': '120px'})
                
                ######################################
                fill_objects = {"left_line":None}
                
                lines_right = []     # continuum-subtracted lines (Rightplot)
                cont_lines  = []     # fitted continuum lines (Leftplot, dashed)
                ##############Lines###################
            
                if "cont" not in line_name:
                    if Y.ndim == 2:
                        # initial compute for current slider windows
                        Y_wo, CONT = calculate_line(X, Y, slider_left_window.value, slider_right_window.value)
                        for n, (x, ywo, cont) in enumerate(zip(X, Y_wo, CONT)):
                            # right: continuum-subtracted spectra
                            rline, = Rightplot.plot(x, ywo, alpha=0.7, label=f'Object {objs[n]}')
                            lines_right.append(rline)

                            # left: show the fitted continuum as dashed overlay
                            cline, = Leftplot.plot(x, cont, linestyle="--", linewidth=2,color="k")
                            cont_lines.append(cline)

                        Rightplot.legend()
                    else:
                        # 1D case
                        Y_wo, CONT = calculate_line([X], [Y], slider_left_window.value, slider_right_window.value)
                        Y_wo, CONT = Y_wo[0], CONT[0]

                        rline, = Rightplot.plot(X, Y_wo)
                        lines_right.append(rline)

                        cline, = Leftplot.plot(X, CONT, linestyle="--", linewidth=2,color="k")
                        cont_lines.append(cline)
                else:
                    if Y.ndim == 2:
                        for n, (x, y) in enumerate(zip(X, Y)):
                            rline, = Rightplot.plot(x, y, alpha=0.7, label=f'Object {objs[n]}')
                            lines_right.append(rline)
                        Rightplot.legend()
                    else:
                        rline, = Rightplot.plot(X, Y)
                        lines_right.append(rline)
                    
                ##############xlim-ylim-left-right-sliders##############                
                slider_xlim_left = widgets.FloatRangeSlider(
                    value=[center_window-150,center_window+150], 
                    min=np.min(X), max=np.max(X), step=0.1,
                    description='Left Wavelength', layout=widgets.Layout(width='100%'),
                    style={'description_width': '120px'})
                
                max_y = np.max(Y[(X > center_window-150) & (X< center_window+150)])

                slider_ylim_left = widgets.FloatRangeSlider(
                    value=[np.min(Y), max_y*1.1], 
                    min=np.min(Y), max=np.max(Y), 
                    step=(np.max(Y) - np.min(Y)) / 100,
                    description='Left Flux', layout=widgets.Layout(width='100%'),
                    style={'description_width': '120px'})
                slider_ylim_right = slider_ylim_left
                slider_xlim_right = widgets.FloatRangeSlider(
                    value=[center_window-150,center_window+150], 
                    min=np.min(X), max=np.max(X), step=0.1,
                    description='Right Wavelength', layout=widgets.Layout(width='100%'),
                    style={'description_width': '120px'})
                
                if "cont" not in line_name:
                    max_y_wo = np.max(Y_wo[(X > center_window-150) & (X< center_window+150)])
                    slider_ylim_right = widgets.FloatRangeSlider(
                        value=[np.min(Y_wo), max_y_wo*1.1], 
                        min=np.min(Y_wo), max=np.max(Y_wo), 
                        step=(np.max(Y) - np.min(Y)) / 100,
                        description='Right Flux', layout=widgets.Layout(width='100%'),
                        style={'description_width': '120px'})
                
                ##############set-xlim-ylim##############
                
                ##############initial-plots##############
                fill_objects.update({'left_window': None,"rigth_window":None, "line_core_left":None, "center_core_window_left":None, "line_core_right":None, "center_core_window_right":None})
                if "cont" not in line_name:
                    fill_objects['left_window'] = Leftplot.fill_betweenx([-100, 100], slider_left_window.value[0], slider_left_window.value[1],label='left window', color='purple', alpha=0.2)
                    fill_objects['right_window'] = Leftplot.fill_betweenx([-100, 100], slider_right_window.value[0], slider_right_window.value[1],label='right window', color='green', alpha=0.2)
                    
                
                fill_objects['line_core_left'] = Leftplot.fill_betweenx([-100, 100], slider_line_core.value[0], slider_line_core.value[1],label='line core', color='black', alpha=0.2)
                fill_objects['center_core_window_left'] = Leftplot.axvline(x=np.mean(slider_line_core.value), color='k', linestyle="--")
                
                
                fill_objects['line_core_right'] = Rightplot.fill_betweenx([-100, 100], slider_line_core.value[0], slider_line_core.value[1],label='line core', color='black', alpha=0.2)
                fill_objects['center_core_window_right'] = Rightplot.axvline(x=np.mean(slider_line_core.value), color='k', linestyle="--")
                
                zero_line = Rightplot.axhline(y=0.0, color='k', linestyle='--', linewidth=1)
            
                Leftplot.set_xlim(slider_xlim_left.value)
                Leftplot.set_ylim(slider_ylim_left.value)
                
                Rightplot.set_xlim(slider_xlim_right.value)
                Rightplot.set_ylim(slider_ylim_right.value)
                
                
            
                def update_plot(change):
                    
                   
                    
                    if fill_objects['line_core_left'] is not None:
                        fill_objects['line_core_left'].remove()
                    if fill_objects['center_core_window_left'] is not None:
                        fill_objects['center_core_window_left'].remove()
                    if fill_objects['line_core_right'] is not None:
                        fill_objects['line_core_right'].remove()
                    if fill_objects['center_core_window_right'] is not None:
                        fill_objects['center_core_window_right'].remove()
                    
                    
                    
                    line_core_val = slider_line_core.value
                    

                    xlim_left = slider_xlim_left.value
                    ylim_left = slider_ylim_left.value
                    xlim_right = slider_xlim_right.value
                    ylim_right = slider_ylim_right.value
                    
                    # Update left plot limits
                    Leftplot.set_xlim(xlim_left[0], xlim_left[1])
                    Leftplot.set_ylim(ylim_left[0], ylim_left[1])
                    
                    Rightplot.set_xlim(xlim_right[0], xlim_right[1])
                    Rightplot.set_ylim(ylim_right[0], ylim_right[1])
                    
                    
                    # Redraw fill with new values
                    if "cont" not in line_name:
                        if fill_objects['left_window'] is not None:
                            fill_objects['left_window'].remove()
                        if fill_objects['right_window'] is not None:
                            fill_objects['right_window'].remove()
                        
                        left_window_val = slider_left_window.value
                        right_window_val = slider_right_window.value
                        
                        fill_objects['left_window'] = Leftplot.fill_betweenx([-100, 100], left_window_val[0], left_window_val[1],label='left window', color='purple', alpha=0.2)
                        
                        fill_objects['right_window'] = Leftplot.fill_betweenx([-100, 100], right_window_val[0], right_window_val[1],label='right window', color='green', alpha=0.2)
                        
                    fill_objects['line_core_left'] = Leftplot.fill_betweenx([-100, 100], line_core_val[0], line_core_val[1],label='line core', color='black', alpha=0.2)
                    fill_objects['center_core_window_left'] =Leftplot.axvline(x=np.mean(line_core_val), color='k', linestyle="--")
                    
                    fill_objects['line_core_right'] = Rightplot.fill_betweenx([-100, 100], line_core_val[0], line_core_val[1],label='line core', color='black', alpha=0.2)
                    fill_objects['center_core_window_right'] = Rightplot.axvline(x=np.mean(line_core_val), color='k', linestyle="--")
                    if "cont" not in line_name:
                        if Y.ndim == 2:
                            Y_wo, CONT = calculate_line(X, Y, left_window_val, right_window_val)

                            # update Rightplot lines (continuum-subtracted)
                            for line, x, ywo in zip(lines_right, X, Y_wo):
                                line.set_data(x, ywo)

                            # update Leftplot continuum overlays
                            for cline, x, cont in zip(cont_lines, X, CONT):
                                cline.set_data(x, cont)
                        else:
                            Y_wo, CONT = calculate_line([X], [Y], left_window_val, right_window_val)
                            Y_wo, CONT = Y_wo[0], CONT[0]

                            lines_right[0].set_data(X, Y_wo)
                            cont_lines[0].set_data(X, CONT)

                    
                    zero_line.set_ydata([0.0, 0.0])
                    fig.canvas.draw_idle()
                
                # Attach update function to all sliders
                if "cont" not in line_name:
                    slider_right_window.observe(update_plot, names='value')
                    slider_left_window.observe(update_plot, names='value')
                slider_line_core.observe(update_plot, names='value')
                
                slider_xlim_left.observe(update_plot, names='value')
                slider_ylim_left.observe(update_plot, names='value')
                slider_xlim_right.observe(update_plot, names='value')
                slider_ylim_right.observe(update_plot, names='value')
                
                # Create status label
                status_label = widgets.Label(value="", layout=widgets.Layout(width='100%'))
                
                
                def _save_current_plots():
                    from pathlib import Path
                    # Folder to store plots
                    save_dir = Path("microlensing_plots")
                    save_dir.mkdir(exist_ok=True)

                    # Base name: e.g. "000_CIV_G_band"
                    base = f"{index:03d}_{line_name}_{band}".replace(" ", "_")

                    # 1) Save full figure
                    full_path = save_dir / f"{base}_both.pdf"
                    fig.savefig(full_path, dpi=150, bbox_inches="tight")

                    # 2) Save left and right panels individually
                    # Make sure layout is up to date
                    fig.canvas.draw()

                    # Left panel
                    left_extent = Leftplot.get_window_extent().transformed(
                        fig.dpi_scale_trans.inverted()
                    )
                    # left_path = save_dir / f"{base}_left.png"
                    # fig.savefig(left_path, bbox_inches=left_extent, dpi=150)

                    # # Right panel
                    # right_extent = Rightplot.get_window_extent().transformed(
                    #     fig.dpi_scale_trans.inverted()
                    # )
                    # right_path = save_dir / f"{base}_right.png"
                    # fig.savefig(right_path, bbox_inches=right_extent, dpi=150)

                    # Optionally, store filenames in saved_parameters
                    self.saved_parameters.setdefault(index, {})
                    self.saved_parameters[index].update(
                        {
                            "plot_file_both": str(full_path),
                        }
                    )
                def on_save_clicked(b):
                    # params = save_parameters(index, slider_left_window, slider_right_window, 
                    #                         slider_line_core, slider_xlim_left, slider_ylim_left,
                    #                         slider_xlim_right, slider_ylim_right, line_name, band, objs,X,Y_original,n_bootstrap=n_bootstrap,random_state=random_state)
                    _save_current_plots()
                    status_label.value = f"✓ Saved parameters for {line_name} (Plot {index+1}/{n_max})"
                    
                    # print(f"Saved parameters for plot {index+1}:")
                    # print(f"  Line: {params['line_name']}, Band: {params['band']}")
                    # print(f"  Left window: {params['left_window']}")
                    # print(f"  Right window: {params['right_window']}")
                    # print(f"  Line core: {params['line_core']}")
                
                def on_previous_clicked(b):
                    plt.close(fig)
                    if current_index[0] > 0:
                        current_index[0] -= 1
                        show_plot(current_index[0])
                    else:
                        print("Already at first plot!")
                
                def on_next_clicked(b):
                   
                    plt.close(fig)
                    current_index[0] += 1
                    if current_index[0] < n_max:
                        show_plot(current_index[0])
                    else:
                        print("All plots completed!")
                        print(f"\nTotal saved parameters: {len(self.saved_parameters)}/{n_max}")
                
                def on_stop_clicked(b):
                    stop_routine[0] = True
                    plt.close(fig)
                    print(f"\n{'='*60}")
                    print(f"Routine stopped at plot {index+1}/{n_max}")
                    print(f"Saved parameters for {len(self.saved_parameters)} plot(s)")
                    print(f"{'='*60}")
                    
                    # Show saved plots summary
                    if self.saved_parameters:
                        print("\nSaved plots:")
                        for idx, params in sorted(self.saved_parameters.items()):
                            print(f"  Plot {idx+1}: {params['line_name']} ({params['band']})")
                
                def on_export_csv_clicked(b):
                    if self.saved_parameters:
                        filename = export_to_csv()
                        status_label.value = f"✓ Exported to {filename}"
                    else:
                        status_label.value = "⚠ No parameters to export"
                        print("No parameters saved yet!")
                
                def on_export_pickle_clicked(b):
                    if self.saved_parameters:
                        filename = export_to_pickle()
                        status_label.value = f"✓ Exported to {filename}"
                    else:
                        status_label.value = "⚠ No parameters to export"
                        print("No parameters saved yet!")
                
                # Create buttons
                save_button = widgets.Button(
                    description="Save",
                    button_style='info',
                    icon='check',
                    layout=widgets.Layout(width='120px', height='35px'))
                
                previous_button = widgets.Button(
                    description="Previous",
                    button_style='warning',
                    icon='arrow-left',
                    disabled=(index == 0),
                    layout=widgets.Layout(width='120px', height='35px'))
                
                next_button = widgets.Button(
                    description="Next" if index < n_max - 1 else "Finish",
                    button_style='success',
                    icon='arrow-right',
                    layout=widgets.Layout(width='120px', height='35px'))
                
                stop_button = widgets.Button(
                    description="Stop & Exit",
                    button_style='danger',
                    icon='stop',
                    layout=widgets.Layout(width='120px', height='35px'))
                
                export_csv_button = widgets.Button(
                    description="Export CSV",
                    button_style='',
                    icon='download',
                    layout=widgets.Layout(width='120px', height='35px'))
                
                export_pickle_button = widgets.Button(
                    description="Export PKL",
                    button_style='',
                    icon='save',
                    layout=widgets.Layout(width='120px', height='35px'))
                
                # Attach callbacks
                save_button.on_click(on_save_clicked)
                previous_button.on_click(on_previous_clicked)
                next_button.on_click(on_next_clicked)
                stop_button.on_click(on_stop_clicked)
                export_csv_button.on_click(on_export_csv_clicked)
                export_pickle_button.on_click(on_export_pickle_clicked)
                
                print(f"Plot {index+1}/{n_max} - {line_name}")
                if index in self.saved_parameters:
                    print("  (Previously saved parameters loaded)")
            
            # CHANGED ORDER: Plot output first, then sliders below
            # First display the plot
            plt.show()
            
            # Then display sliders and controls below the plot
            
            if "cont" not in line_name:
                display(widgets.VBox([
                    widgets.HTML(f"<h3 style='text-align: center; margin-top: 10px;'>{line_name} </h3>"),
                    widgets.HTML("<hr style='margin: 10px 0;'>"),
                    widgets.HTML("<b>Continuum Windows:</b>"),
                    slider_left_window,
                    slider_line_core,
                    slider_right_window,
                    widgets.HTML("<br><b>Left Plot Limits:</b>"),
                    slider_xlim_left,
                    slider_ylim_left,
                    widgets.HTML("<br><b>Right Plot Limits:</b>"),
                    slider_xlim_right,
                    slider_ylim_right,
                    widgets.HTML("<hr style='margin: 10px 0;'>"),
                    status_label,
                    widgets.HBox([previous_button, save_button, next_button, stop_button], 
                                layout=widgets.Layout(justify_content='center', width='100%')),
                    widgets.HTML("<br><b>Export Options:</b>"),
                    widgets.HBox([export_csv_button, export_pickle_button], 
                                layout=widgets.Layout(justify_content='center', width='100%'))
                ], layout=widgets.Layout(width='100%', padding='20px')))
            else:
                display(widgets.VBox([
                    widgets.HTML(f"<h3 style='text-align: center; margin-top: 10px;'>{line_name} </h3>"),
                    widgets.HTML("<hr style='margin: 10px 0;'>"),
                    widgets.HTML("<b>Continuum Windows:</b>"),
                    slider_line_core,
                    widgets.HTML("<br><b>Left Plot Limits:</b>"),
                    slider_xlim_left,
                    slider_ylim_left,
                    widgets.HTML("<br><b>Right Plot Limits:</b>"),
                    slider_xlim_right,
                    slider_ylim_right,
                    widgets.HTML("<hr style='margin: 10px 0;'>"),
                    status_label,
                    widgets.HBox([previous_button, save_button, next_button, stop_button], 
                                layout=widgets.Layout(justify_content='center', width='100%')),
                    widgets.HTML("<br><b>Export Options:</b>"),
                    widgets.HBox([export_csv_button, export_pickle_button], 
                                layout=widgets.Layout(justify_content='center', width='100%'))
                ], layout=widgets.Layout(width='100%', padding='20px')))
                
        display(output)
        show_plot(0)
        
 