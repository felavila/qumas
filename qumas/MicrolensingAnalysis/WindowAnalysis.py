import numpy as np 
import os 
import pandas as pd 
import warnings
import json
import matplotlib.pyplot as plt 
from copy import deepcopy
from .functions import linear_model,linear_func
from scipy.integrate import quad
from matplotlib.widgets import Button,RangeSlider,Slider
import pickle
from mpl_toolkits import axisartist
import ipywidgets as widgets
from IPython.display import display, clear_output


module_dir = os.path.dirname(os.path.abspath(__file__))
windows_rest_frame = os.path.join(module_dir,"rest_frame_windows","rest_frame_windows.json")
with open(windows_rest_frame, "r") as f:
    windows_rest_frame = json.load(f)
def convert_none_to_nan(item):
    if item is None:
        return np.nan
    elif isinstance(item, list):
        return [convert_none_to_nan(x) for x in item]
    elif isinstance(item, dict):
        return {key: convert_none_to_nan(value) if isinstance(value[0],str) else convert_none_to_nan(value) for key,value in item.items()}
    else:
        return item




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
        self._previous_results =  self._read_previous_results(path_previous_results)
        
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
        print(self.spectra_dict.keys())
        band_limits = {band: np.array(self.spectra_dict[band]["wavelength"])[0][[0,-1]] for band in self.spectra_dict.keys()}
        
        
        print(band_limits)
        for number,line_name in enumerate(self.pre_define_windows["line_name"].values):
            row = self.pre_define_windows[self.pre_define_windows["line_name"]==line_name]
            center_window = np.mean(row["core_range"].values[0])
            window = [center_window-500,center_window+500]
            #
            lr_init = row["right_range"].values[0]
            lc_init = row["left_range"].values[0]
            core_init = row["core_range"].values[0]
            #
            wl, wr = float(min(window)), float(max(window))
            w_center = 0.5 * (wl + wr)

            candidates = []
            for name, (b_min, b_max) in band_limits.items():
                b_min, b_max = float(b_min), float(b_max)
                b_center = 0.5 * (b_min + b_max)
                b_width  = b_max - b_min

                # coverage & overlap
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

            # 1) full cover first
            full = [c for c in candidates if c["full_cover"]]
            pool = full if full else candidates

            # 2) sort by: (full_cover desc), overlap desc, center_dist asc, band_width asc
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
            X, Y, OBJS = np.array(self.spectra_dict[band]["wavelength"]), np.array(self.spectra_dict[band]["flux"]),self.spectra_dict[band]["obj"]
            #   x,y,objs = np.array(self.spectra_dict[band]["wavelength"]), np.array(self.spectra_dict[band]["flux"]),self.spectra_dict[band]["obj"]
            w_mask = (X[0]>= min(window)) & (X[0]<= max(window))
            Y_local = Y[:,w_mask]
            max_local = np.max(Y_local) #
            min_local = np.min(Y_local) #
            aspect_ratio = 1.5 #
            q4 = np.percentile(Y_local, 99.97) 
            
            self.kwargs_h[number] = {"X":X,"Y":Y,"objs":OBJS,"center_window":center_window,"window":window,
                                    "lr_init":lr_init,"lc_init":lc_init,"core_init":core_init,"band":band,"line_name":line_name,"Y_local":0,
                                    "max_local":max_local,"min_local":min_local,"aspect_ratio":aspect_ratio,"q4":q4}

                    
            # #self.interactive_plot(**row)
            #X,Y,OBJ =
            #for band in self.spectra_dict.keys():
            #   x,y,objs = np.array(self.spectra_dict[band]["wavelength"]), np.array(self.spectra_dict[band]["flux"]),self.spectra_dict[band]["obj"]
            #  w_mask = (x[0]>= min(window)) & (x[0]<= max(window))
            # Y_local = y[:,w_mask]
            #max_local = np.max(Y_local) #
            #min_local = np.min(Y_local) #
            #aspect_ratio = 1.5 #
            #q4 = np.percentile(Y_local, 99.97) 
            #if center_window < np.min(x) or center_window > np.max(y):
            #   continue
            #self.interactive_plot(X,Y,objs,center_window,window,lr_init,lc_init,core_init,band,line_name)
    
    
    
    
    def interactive_plot(self,X,Y,objs,center_window,window,lr_init,lc_init,core_init,band,line_name):
        #X,Y,objs = np.array(self.spectra_dict[band]["wavelength"]), np.array(self.spectra_dict[band]["flux"]),self.spectra_dict[band]["obj"]
        #if center_window < np.min(X[0]) or center_window > np.max(X[0]):
        #   return "cant do nothing"
        #else:
            w_mask = (X[0]>=min(window)) & (X[0]<=max(window)) #soft coming sooon 
            Y_local = Y[:,w_mask]
            q2 = np.percentile(Y_local, 89)
            q3 = np.percentile(Y_local, 95) 
            q4 = np.percentile(Y_local, 99.97) 
            max_local = np.max(Y_local)
            min_local = np.min(Y_local)
            aspect_ratio = 1.5
            
            fig = plt.figure(figsize=(20, 15 / 1.5))
            
            grid = plt.GridSpec(2, 2, width_ratios=[2, 2], height_ratios=[3, 1], hspace=0.4)
            Lp = plt.subplot(grid[0, 0])
            Rp = plt.subplot(grid[0, 1])
            bbox_Lp = Lp.get_position()
            bbox_Rp = Rp.get_position()
            gap_left = bbox_Lp.x0/4 + bbox_Lp.width
            #the val init aqui luego 
            #line -> todo el resto adentro ? lo q seria un doble looop?
            #if self._previous_results.get(f"{}_{}"):
                #Wrange_lc_Lp,Wrange_rc_Lp,Wrange_core,Wrange_Lp,Wrange_Rp,Frange_Lp,Frange_Rp
            
            #window = [center_window - 500, center_window + 500]
            _d = 0.15
            Wslider_lc = plt.axes([gap_left, bbox_Lp.y0 -_d,  bbox_Lp.width, 0.03])
            Wslider_core = plt.axes([gap_left, bbox_Lp.y0 -_d - 0.05,  bbox_Lp.width, 0.03])
            Wslider_rc = plt.axes([gap_left, bbox_Lp.y0 - _d -0.10,  bbox_Lp.width, 0.03])
            
            Wslider_Lp = plt.axes([bbox_Lp.x0, bbox_Lp.y1*1.01, bbox_Lp.width, 0.03]) 
            Wslider_Rp = plt.axes([bbox_Rp.x0, bbox_Rp.y1*1.01, bbox_Rp.width, 0.03]) 
            
            Fslider_Lp = plt.axes([bbox_Lp.x0 + bbox_Lp.width, bbox_Lp.y0, 0.03, bbox_Lp.height])
            Fslider_Rp = plt.axes([bbox_Rp.x0 + bbox_Rp.width , bbox_Rp.y0, 0.03, bbox_Rp.height]) 
            #Wslider_core_Rp = plt.axes([bbox_Rp.x0, bbox_Rp.y0-0.1, bbox_Rp.width, 0.03]) 
            Wrange_lc_Lp = RangeSlider(Wslider_lc, "left \ncontinium",np.min(X),center_window , valinit=lc_init,color="purple",alpha=0.5) 
            Wrange_rc_Lp = RangeSlider(Wslider_rc, "right \ncontinium",center_window,np.max(X),valinit=lr_init,color="green",alpha=0.2) 
            Wrange_core = RangeSlider(Wslider_core ,"line core",center_window-100,center_window+100,valinit=core_init,color="r",alpha=0.2)
            
            Wrange_Lp = RangeSlider(Wslider_Lp,None,np.min(X),np.max(X),valinit=window)
            Wrange_Rp = RangeSlider(Wslider_Rp,None,np.min(X),np.max(X),valinit=window)
            Frange_Lp = RangeSlider(Fslider_Lp, None, 0, max_local , valinit=[0,q4], orientation='vertical')
            Frange_Rp = RangeSlider(Fslider_Rp,None , -max_local*0.5, max_local , valinit=[-max_local*0.01,max_local*0.5], orientation='vertical')
            
            Wrange_Lp.valtext.set_visible(False)
            Wrange_Rp.valtext.set_visible(False)
            Frange_Lp.valtext.set_visible(False)
            Frange_Rp.valtext.set_visible(False)
            Wrange_lc_Lp.valtext.set_visible(False)
            Wrange_rc_Lp.valtext.set_visible(False)
            Wrange_core.valtext.set_visible(False)
            
            # Use the helper method for the initial plot
            self._window_plot(Lp, Rp, X, Y, objs, center_window, line_name, band, 
                            Wrange_lc_Lp, Wrange_rc_Lp, Wrange_core, 
                            Wrange_Lp, Frange_Lp, Wrange_Rp, Frange_Rp)
            
            # Connect sliders to update the plot using the same helper method in the callback
            slider_update = lambda val: self._window_plot(Lp, Rp, X, Y, objs, center_window, line_name, band, Wrange_lc_Lp, Wrange_rc_Lp, Wrange_core, Wrange_Lp, Frange_Lp, Wrange_Rp, Frange_Rp)
            
            Wrange_Lp.on_changed(slider_update)
            Frange_Lp.on_changed(slider_update)
            Wrange_Rp.on_changed(slider_update)
            Frange_Rp.on_changed(slider_update)
            Wrange_lc_Lp.on_changed(slider_update)
            Wrange_rc_Lp.on_changed(slider_update)
            Wrange_core.on_changed(slider_update)
            
            save_button = plt.axes([0.4, 0.02, 0.2, 0.04])
            button_save = Button(save_button, 'Save', color='lightgoldenrodyellow', hovercolor='0.975')
            
            button_save.on_clicked(lambda event: self._on_save_button_clicked(event,Lp, Rp, X, Y, objs, center_window, line_name, band, Wrange_lc_Lp, Wrange_rc_Lp, Wrange_core, Wrange_Lp, Frange_Lp, Wrange_Rp, Frange_Rp))
            
            #if name_file in os.listdir(os.getcwd()):
            #   if any((panda_read[["name"]].values == [line_name]).all(axis=1)):
            #      ax_save_button = plt.axes([0.6, 0.02, 0.2, 0.04])
            #     button_remove = Button(ax_save_button, 'remove line', color='lightgoldenrodyellow', hovercolor='0.975')
            #    button_remove.on_clicked(on_remove_line_clicked)
            # Update the plot when slider values change
            #ax_save_button = plt.axes([0.01, 0.95, 0.2, 0.04])
            #button_close = Button(ax_save_button, 'Close', color='lightgoldenrodyellow', hovercolor='0.975')
            #button_close.on_clicked(on_close_all)
            
            plt.show()
            
            
            return X, Y, window, Lp.get_position()
    
    
    
    def _interactive_microlensing(self):
        output = widgets.Output()
        n_max = len(self.kwargs_h.keys())
        current_index = [0]
        
        def show_plot(index):
            X, Y, objs, center_window, window, lr_init, lc_init,  core_init, band, line_name, Y_local, max_local, min_local, aspect_ratio, q4 = self.kwargs_h[index].values()
            #print(lr_init,lc_init,core_init)
            with output:
                clear_output(wait=True)
                fig = plt.figure(figsize=(15, 10), layout="constrained")
                fig.canvas.draw()
                gs = fig.add_gridspec(2, 2, width_ratios=[2, 2], height_ratios=[3, 1], hspace=0.4)
                Leftplot = fig.add_subplot(gs[0, 0], axes_class=axisartist.Axes)
                Rightplot = fig.add_subplot(gs[0, 1], axes_class=axisartist.Axes)
                
                # Initialize x data for parabola
                x_data = np.linspace(0, 1, 100)
                
                # Initial parabola parameters
                a_init = 1.0
                b_init = 0.0
                y_data = a_init * x_data**2 + b_init
                
                (line_left,) = Leftplot.plot(X.T, Y.T)
                line_right, = Rightplot.plot(x_data, y_data)

                Leftplot.set_title(f"Interactive left {index+1}/{n_max}")
                Rightplot.set_title(f"Interactive right {index+1}/{n_max}")
                Leftplot.set_xlabel("X")
                Rightplot.set_xlabel("X")
                Leftplot.set_ylabel("Amplitude")
                Rightplot.set_ylabel("Amplitude")
                
                # Initial axis limits
                Leftplot.set_xlim(np.min(X),np.max(X))
                Leftplot.set_ylim(np.min(Y),np.max(Y))
                Rightplot.set_xlim(0, 1)
                Rightplot.set_ylim(0, 2)
                
                
                
                
                # Create six sliders - separate limits for each plot
                slider_a = widgets.FloatSlider(value=1.0, min=-5.0, max=5.0, step=0.1, 
                                            description='a (x²)')
                slider_b = widgets.FloatSlider(value=0.0, min=-5.0, max=5.0, step=0.1, 
                                            description='b')
                slider_c = widgets.FloatSlider(value=0.0, min=-5.0, max=5.0, step=0.1, 
                                            description='c')
                
                # Left plot limits
                slider_xlim_left = widgets.FloatRangeSlider(value=[np.min(X),np.max(X)], min=np.min(X), max=np.max(X), step=0.1,
                                                            description='Left X')
                slider_ylim_left = widgets.FloatRangeSlider(value=[np.min(Y),np.max(Y)], min=np.min(Y), max=np.max(Y), step=0.1,
                                                            description='Left Y')
                
                # Right plot limits
                slider_xlim_right = widgets.FloatRangeSlider(value=[0.0, 1.0], min=-2.0, max=5.0, step=0.1,
                                                            description='Right X')
                slider_ylim_right = widgets.FloatRangeSlider(value=[0.0, 2.0], min=-5.0, max=10.0, step=0.1,
                                                            description='Right Y')
                
                def update_plot(change):
                    # Get current values
                    a = slider_a.value
                    b = slider_b.value
                    c = slider_c.value
                    xlim_left = slider_xlim_left.value
                    ylim_left = slider_ylim_left.value
                    xlim_right = slider_xlim_right.value
                    ylim_right = slider_ylim_right.value
                    
                    # Update left plot
                    x_left = np.linspace(xlim_left[0], xlim_left[1], 100)
                    y_left = a * x_left**2 + b*x_left + c
                    line_left.set_data(x_left, y_left)
                    Leftplot.set_xlim(xlim_left[0], xlim_left[1])
                    Leftplot.set_ylim(ylim_left[0], ylim_left[1])
                    
                    # Update right plot
                    x_right = np.linspace(xlim_right[0], xlim_right[1], 100)
                    y_right = a * x_right**2 + b*x_right + c
                    line_right.set_data(x_right, y_right)
                    Rightplot.set_xlim(xlim_right[0], xlim_right[1])
                    Rightplot.set_ylim(ylim_right[0], ylim_right[1])
                    
                    fig.canvas.draw_idle()

                # Attach update function to all sliders
                slider_a.observe(update_plot, names='value')
                slider_b.observe(update_plot, names='value')
                slider_c.observe(update_plot, names='value')
                slider_xlim_left.observe(update_plot, names='value')
                slider_ylim_left.observe(update_plot, names='value')
                slider_xlim_right.observe(update_plot, names='value')
                slider_ylim_right.observe(update_plot, names='value')
                
                next_button = widgets.Button(description="Next Plot" if index < n_max - 1 else "Finish",
                                            button_style='success')
                
                def on_next_clicked(b):
                    plt.close(fig)
                    current_index[0] += 1
                    if current_index[0] < n_max:
                        show_plot(current_index[0])
                    else:
                        print("All plots completed!")
                
                next_button.on_click(on_next_clicked)
                
                print(f"Plot {index+1}/{n_max}")
            
            # Display widgets OUTSIDE the output context
            # Organize sliders: parabola params, then left plot limits, then right plot limits
            display(widgets.VBox([
                widgets.Label("Parabola Parameters (y = ax² + b):"),
                widgets.HBox([slider_a, slider_b,slider_c]),
                widgets.Label("Left Plot Limits:"),
                widgets.VBox([slider_xlim_left, slider_ylim_left]),
                widgets.Label("Right Plot Limits:"),
                widgets.VBox([slider_xlim_right, slider_ylim_right]),
                next_button
            ]))
            plt.show()
        
        display(output)
        show_plot(0)
        
        
        
        
    def _interactive_microlensing2(self):
        output = widgets.Output()
        n_max = len(self.kwargs_h.keys())
        current_index = [0]

        def show_plot(index):
            with output:
                clear_output(wait=True)

                fig = plt.figure(figsize=(15, 5), layout="constrained")

                # Top-level: 2 columns (Left/Right)
                outer = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1, 1], wspace=0.25)

                # Each column gets a 2-row subgrid: [plot, slider]
                left_gs  = outer[0].subgridspec(nrows=2, ncols=1, height_ratios=[30, 1], hspace=0.05)
                right_gs = outer[1].subgridspec(nrows=2, ncols=1, height_ratios=[30, 1], hspace=0.05)

                # Axes for plots (axisartist if you need it)
                Leftplot  = fig.add_subplot(left_gs[0],  axes_class=axisartist.Axes)
                Rightplot = fig.add_subplot(right_gs[0], axes_class=axisartist.Axes)

                # Axes *dedicated* to sliders; they share the exact same width as their plot
                axSliderL = fig.add_subplot(left_gs[1])
                axSliderR = fig.add_subplot(right_gs[1])

                # Optional: make slider axes neat
                for axS in (axSliderL, axSliderR):
                    axS.set_facecolor("none")
                    for spine in axS.spines.values():
                        spine.set_visible(False)
                    axS.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

                # Example lines
                (line_left,)  = Leftplot.plot([0, 1], [0, 1])
                (line_right,) = Rightplot.plot([0, 1], [0, 1])

                Leftplot.set_title(f"Interactive left {index+1}/{n_max}")
                Rightplot.set_title(f"Interactive right {index+1}/{n_max}")
                Leftplot.set_xlabel("X");  Leftplot.set_ylabel("Amplitude")
                Rightplot.set_xlabel("X"); Rightplot.set_ylabel("Amplitude")

                # Sliders with identical widths to their plots
                sliderL = Slider(axSliderL, "Amplitude (%)", 0, 100, valinit=50)
                sliderR = Slider(axSliderR, "Amplitude (%)", 0, 100, valinit=50)

                def update_left(val):
                    new_amp = sliderL.val / 50.0
                    line_left.set_ydata(new_amp * np.array([0, 1]))
                    fig.canvas.draw_idle()

                def update_right(val):
                    new_amp = sliderR.val / 50.0
                    line_right.set_ydata(2 * new_amp * np.array([0, 1]))
                    fig.canvas.draw_idle()

                sliderL.on_changed(update_left)
                sliderR.on_changed(update_right)

                next_button = widgets.Button(
                    description=("Next Plot" if index < n_max-1 else "Finish"),
                    button_style='success'
                )

                def on_next_clicked(_):
                    plt.close(fig)
                    current_index[0] += 1
                    if current_index[0] < n_max:
                        show_plot(current_index[0])
                    else:
                        print("All plots completed!")

                next_button.on_click(on_next_clicked)

                print(f"Plot {index+1}/{n_max}")

            # Jupyter widgets live outside the matplotlib Output
            display(widgets.HBox([next_button]))
            plt.show()

        display(output)
        show_plot(0)

    
    def _experiment(self):
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        
        output = widgets.Output()
        current_index = [0]
        
        def show_plot(index):
            with output:
                clear_output(wait=True)
                
                x = np.linspace(0, 10, 100)
                initial_amplitude = 1
                y = initial_amplitude * np.sin(x)

                fig, ax = plt.subplots(figsize=(10, 5))
                line, = ax.plot(x, y)
                ax.set_title(f"Interactive Sine Wave {index+1}/3")
                ax.set_xlabel("X")
                ax.set_ylabel("Amplitude")

                slider = widgets.IntSlider(value=50, min=0, max=100, step=1, 
                                        description='Amplitude (%)')

                def update_plot(change):
                    new_amplitude = slider.value / 50.0
                    line.set_ydata(new_amplitude * np.sin(x))
                    fig.canvas.draw_idle()

                slider.observe(update_plot, names='value')
                
                next_button = widgets.Button(description="Next Plot" if index < 2 else "Finish",
                                            button_style='success')
                
                def on_next_clicked(b):
                    plt.close(fig)
                    current_index[0] += 1
                    if current_index[0] < 3:
                        show_plot(current_index[0])
                    else:
                        print("All plots completed!")
                
                next_button.on_click(on_next_clicked)
                
                # Just display the controls, matplotlib handles the plot
                print(f"Plot {index+1}/3")
            
            # Display widgets OUTSIDE the output context
            display(widgets.HBox([slider, next_button]))
            plt.show()
        
        display(output)
        show_plot(0)
        
    def windows_analysis(self,band="nir",):
        """_summary_
            overall i will build all around the idea of that the two bands share the same number of pixels because is more easier but
            maybe is more consistent have the idea of this is not always the case.
        """
        
        row = self.pre_define_windows.iloc[8]
        line = row['line_name']
        center_window = np.mean(row["core_range"])
        window = [center_window-500,center_window+500]
        lr_init = row["right_range"]
        lc_init = row["left_range"]
        core_init = row["core_range"]
        #band = "nir"
        X,Y,objs = np.array(self.spectra_dict[band]["wavelength"]), np.array(self.spectra_dict[band]["flux"]),self.spectra_dict[band]["obj"]
        
        if center_window < np.min(X[0]) or center_window > np.max(X[0]):
            return "cant do nothing"
        else:
            w_mask = (X[0]>=min(window)) & (X[0]<=max(window)) #soft coming sooon 
            Y_local = Y[:,w_mask]
            q2 = np.percentile(Y_local, 89)
            q3 = np.percentile(Y_local, 95) 
            q4 = np.percentile(Y_local, 99.97) 
            max_local = np.max(Y_local)
            min_local = np.min(Y_local)
            aspect_ratio = 1.5
            fig = plt.figure(figsize=(20, 15 / 1.5))
            grid = plt.GridSpec(2, 2, width_ratios=[2, 2], height_ratios=[3, 1], hspace=0.4)
            Lp = plt.subplot(grid[0, 0])
            Rp = plt.subplot(grid[0, 1])
            bbox_Lp = Lp.get_position()
            bbox_Rp = Rp.get_position()
            gap_left = bbox_Lp.x0/4 + bbox_Lp.width
            #the val init aqui luego 
            #line -> todo el resto adentro ? lo q seria un doble looop?
            #if self._previous_results.get(f"{}_{}"):
                #Wrange_lc_Lp,Wrange_rc_Lp,Wrange_core,Wrange_Lp,Wrange_Rp,Frange_Lp,Frange_Rp
            
            window = [center_window - 500, center_window + 500]
            _d = 0.15
            Wslider_lc = plt.axes([gap_left, bbox_Lp.y0 -_d,  bbox_Lp.width, 0.03])
            Wslider_core = plt.axes([gap_left, bbox_Lp.y0 -_d - 0.05,  bbox_Lp.width, 0.03])
            Wslider_rc = plt.axes([gap_left, bbox_Lp.y0 - _d -0.10,  bbox_Lp.width, 0.03])
            
            Wslider_Lp = plt.axes([bbox_Lp.x0, bbox_Lp.y1*1.01, bbox_Lp.width, 0.03]) 
            Wslider_Rp = plt.axes([bbox_Rp.x0, bbox_Rp.y1*1.01, bbox_Rp.width, 0.03]) 
            
            Fslider_Lp = plt.axes([bbox_Lp.x0 + bbox_Lp.width, bbox_Lp.y0, 0.03, bbox_Lp.height])
            Fslider_Rp = plt.axes([bbox_Rp.x0 + bbox_Rp.width , bbox_Rp.y0, 0.03, bbox_Rp.height]) 
            #Wslider_core_Rp = plt.axes([bbox_Rp.x0, bbox_Rp.y0-0.1, bbox_Rp.width, 0.03]) 
            Wrange_lc_Lp = RangeSlider(Wslider_lc, "left \ncontinium",np.min(X),center_window , valinit=lc_init,color="purple",alpha=0.5) 
            Wrange_rc_Lp = RangeSlider(Wslider_rc, "right \ncontinium",center_window,np.max(X),valinit=lr_init,color="green",alpha=0.2) 
            Wrange_core = RangeSlider(Wslider_core ,"line core",center_window-100,center_window+100,valinit=core_init,color="r",alpha=0.2)
            
            Wrange_Lp = RangeSlider(Wslider_Lp,None,np.min(X),np.max(X),valinit=window)
            Wrange_Rp = RangeSlider(Wslider_Rp,None,np.min(X),np.max(X),valinit=window)
            Frange_Lp = RangeSlider(Fslider_Lp, None, 0, max_local , valinit=[0,q4], orientation='vertical')
            Frange_Rp = RangeSlider(Fslider_Rp,None , -max_local*0.5, max_local , valinit=[-max_local*0.01,max_local*0.5], orientation='vertical')
            
            Wrange_Lp.valtext.set_visible(False)
            Wrange_Rp.valtext.set_visible(False)
            Frange_Lp.valtext.set_visible(False)
            Frange_Rp.valtext.set_visible(False)
            Wrange_lc_Lp.valtext.set_visible(False)
            Wrange_rc_Lp.valtext.set_visible(False)
            Wrange_core.valtext.set_visible(False)
            
            # Use the helper method for the initial plot
            self._window_plot(Lp, Rp, X, Y, objs, center_window, line, band, 
                            Wrange_lc_Lp, Wrange_rc_Lp, Wrange_core, 
                            Wrange_Lp, Frange_Lp, Wrange_Rp, Frange_Rp)
            
            # Connect sliders to update the plot using the same helper method in the callback
            slider_update = lambda val: self._window_plot(Lp, Rp, X, Y, objs, center_window, line, band, Wrange_lc_Lp, Wrange_rc_Lp, Wrange_core, Wrange_Lp, Frange_Lp, Wrange_Rp, Frange_Rp)
            
            Wrange_Lp.on_changed(slider_update)
            Frange_Lp.on_changed(slider_update)
            Wrange_Rp.on_changed(slider_update)
            Frange_Rp.on_changed(slider_update)
            Wrange_lc_Lp.on_changed(slider_update)
            Wrange_rc_Lp.on_changed(slider_update)
            Wrange_core.on_changed(slider_update)
            
            save_button = plt.axes([0.4, 0.02, 0.2, 0.04])
            button_save = Button(save_button, 'Save', color='lightgoldenrodyellow', hovercolor='0.975')
            
            button_save.on_clicked(lambda event: self._on_save_button_clicked(event,Lp, Rp, X, Y, objs, center_window, line, band, Wrange_lc_Lp, Wrange_rc_Lp, Wrange_core, Wrange_Lp, Frange_Lp, Wrange_Rp, Frange_Rp))
            #if name_file in os.listdir(os.getcwd()):
            #   if any((panda_read[["name"]].values == [line_name]).all(axis=1)):
            #      ax_save_button = plt.axes([0.6, 0.02, 0.2, 0.04])
            #     button_remove = Button(ax_save_button, 'remove line', color='lightgoldenrodyellow', hovercolor='0.975')
            #    button_remove.on_clicked(on_remove_line_clicked)
            # Update the plot when slider values change
            #ax_save_button = plt.axes([0.01, 0.95, 0.2, 0.04])
            #button_close = Button(ax_save_button, 'Close', color='lightgoldenrodyellow', hovercolor='0.975')
            #button_close.on_clicked(on_close_all)
            plt.show()
            
            
            return X, Y, window, Lp.get_position()
    
    def _on_save_button_clicked(self,event,Lp, Rp, X, Y, objs, center_window, line, band, Wrange_lc_Lp, Wrange_rc_Lp, Wrange_core, Wrange_Lp, Frange_Lp, Wrange_Rp, Frange_Rp):
        #Add rutine to change the multiple dictionaries in case of been necesary 
        fig, (Lp, Rp) = plt.subplots(1, 2, figsize=(30, 10), gridspec_kw={'width_ratios': [2, 2]})
        self._window_plot(Lp, Rp, X, Y, objs, center_window, line, band, Wrange_lc_Lp, Wrange_rc_Lp, Wrange_core, Wrange_Lp, Frange_Lp, Wrange_Rp, Frange_Rp)
        plt.savefig("aja.png", dpi=300, bbox_inches='tight')
        plt.close()
        with open('microlensing.pkl', 'wb') as f:
            print("saved as microlensing.pkl")
            pickle.dump(self.local_list, f)
        
    def _local(self,line_name,imagen,band,x,Wrange_lc_Lp,Wrange_rc_Lp,Wrange_core,Wrange_Lp,Wrange_Rp,Frange_Lp,Frange_Rp):
        result_ = {"line_name":line_name,"band": band,'min_x':np.min(x),
                'max_x': np.max(x),'Wrange_lc_Lp_max': max(Wrange_lc_Lp.val),
                'Wrange_lc_Lp_min': min(Wrange_lc_Lp.val),
                'Wrange_rc_Lp_max': max(Wrange_rc_Lp.val),
                'Wrange_rc_Lp_min': min(Wrange_rc_Lp.val),
                'Wrange_core_max': max(Wrange_core.val),
                'Wrange_core_min': min(Wrange_core.val),
                'Wrange_Lp_max': max(Wrange_Lp.val),
                'Wrange_Lp_min': min(Wrange_Lp.val),
                'Wrange_Rp_max': max(Wrange_Rp.val),
                'Wrange_Rp_min': min(Wrange_Rp.val),
                'Frange_Lp_max': max(Frange_Lp.val),
                'Frange_Lp_min': min(Frange_Lp.val),
                'Frange_Rp_max': max(Frange_Rp.val),
                'Frange_Rp_min': min(Frange_Rp.val)}
        #if self._previous_results:
            #print(imagen)
            #print(self._previous_results[imagen])
         #   prev=self._previous_results.get(f"{imagen}_{band}")
            #print(prev.keys())
            #print([[prev[key],current[key]] for key in current.keys()])
          #  same = all([prev.get(key) == result_.get(key) for key in result_.keys()])
            
            #print(same)
        return result_
    
    
    def _read_previous_results(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            with open(path, 'rb') as f:
                loaded_list = pd.DataFrame(list(pickle.load(f).values()))
                #print(loaded_list)
            return loaded_list
        else:
            return None
        
    #def _compare_dict()
    
    
    
    
    
    def _window_plot(self, Lp, Rp, X, Y, objs, center_window, line, band, 
                     Wrange_lc_Lp, Wrange_rc_Lp, Wrange_core, 
                     Wrange_Lp, Frange_Lp, Wrange_Rp, Frange_Rp):
        """
        Plot the common elements on the provided axes.
        """
        Lp.clear()
        Rp.clear()
        Lp.set_xlim(Wrange_Lp.val)
        Lp.set_ylim(Frange_Lp.val)
        Rp.set_xlim(Wrange_Rp.val)
        Rp.set_ylim(Frange_Rp.val)
        
        Lp.plot(X.T, Y.T, label=objs)
        Lp.legend(framealpha=0, fontsize=12)
        Lp.text(Lp.get_xlim()[0] + 0.01, Lp.get_ylim()[1] * 0.95, f"Window {line} in {band}", fontsize=12)
        Lp.axvline(x=center_window, color='k', linestyle="--")
        Rp.axvline(x=center_window, color='k', linestyle="--")
        
        Lp.fill_betweenx([-100, 100], *Wrange_lc_Lp.val, label='left continuum', color='purple', alpha=0.2)
        Lp.fill_betweenx([-100, 100], *Wrange_rc_Lp.val, label='right continuum', color='g', alpha=0.2)
        Lp.fill_betweenx([-100, 100], *Wrange_core.val, label='Core', color='r', alpha=0.2)
        Rp.fill_betweenx([-100, 100], *Wrange_core.val, label='Core', color='r', alpha=0.2)
        Rp.hlines(y=[0, 0], xmin=np.min(X), xmax=np.max(X), colors='k', linestyles='dashed', zorder=3)
        self.local_list = {}
        for i, (x, y, key) in enumerate(zip(X, Y, objs)):
            mask_lc = (x >= min(Wrange_lc_Lp.val)) & (x <= max(Wrange_lc_Lp.val))
            mask_rc = (x >= min(Wrange_rc_Lp.val)) & (x <= max(Wrange_rc_Lp.val))
            x_combined = np.concatenate((x[mask_lc], x[mask_rc]))
            y_combined = np.concatenate((y[mask_lc], y[mask_rc]))
            slope_fit, intercept_fit = linear_model(x_combined, y_combined)
            y_fit = linear_func(x_combined, slope_fit, intercept_fit)
            Lp.plot(x_combined, y_fit, label=f'Fitted Linear Function for {key}', color='k', linestyle="--")
            y_without_cont = y - linear_func(x, slope_fit, intercept_fit)
            Rp.plot(x, y_without_cont, label=key, alpha=0.5)
            self.local_list[f"{key}_{band}"] = self._local(line,key,band,x,Wrange_lc_Lp,Wrange_rc_Lp,Wrange_core,Wrange_Lp,Wrange_Rp,Frange_Lp,Frange_Rp)
            #.append(self._local(line,key,band,x,Wrange_lc_Lp,Wrange_rc_Lp,Wrange_core,Wrange_Lp,Wrange_Rp,Frange_Lp,Frange_Rp))
            #self.local_ = 
            #         # Calculate the area under the linear function between Barrier 1 and Barrier 4
            #area, _ = quad(linear_fit, x_barrier1, x_barrier4, args=(slope_fit, intercept_fit))
            #y_curve = y_noisy - linear_fit(x, slope_fit, intercept_fit)
            #Y_curve = Y[key] - linear_fit(X, slope_fit, intercept_fit)
            #suma = np.sum(y_curve[(x_barrier5 <= x) & (x_barrier6 >= x)])

        Rp.legend(framealpha=0, fontsize=12)
        Lp.tick_params(which="both", length=10, width=2, labelsize=20)
        Rp.tick_params(which="both", length=10, width=2, labelsize=20)
        Lp.set_ylabel(r"$f_\lambda\,(\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}})$", fontsize=20)
        Lp.set_xlabel(r'$\rm Rest \ Wavelength$ ($\rm \AA$)', fontsize=20)
        Rp.set_ylabel(r"$f_\lambda\,(\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}})$", fontsize=20)
        Rp.set_xlabel(r'$\rm Rest \ Wavelength$ ($\rm \AA$)', fontsize=20)
        plt.draw()
        
        