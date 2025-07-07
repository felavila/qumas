import numpy as np 
import os 
import pandas as pd 
import warnings
import json
import matplotlib.pyplot as plt 
from copy import deepcopy
from .functions import linear_model,linear_func
from scipy.integrate import quad
from matplotlib.widgets import Button,RangeSlider
import pickle

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
    
    def __call__(self,**kwargs):
        lines = self.pre_define_windows["line_name"].values
        for line_name in lines:
            #print(line_name)
            row = self.pre_define_windows[self.pre_define_windows["line_name"]==line_name]
            center_window = np.mean(row["core_range"].values[0])
            window = [center_window-500,center_window+500]
            lr_init = row["right_range"].values[0]
            lc_init = row["left_range"].values[0]
            core_init = row["core_range"].values[0]
            #print(center_window,lr_init,lr_init,lc_init,core_init)
            if self._previous_results is not None and line_name in self._previous_results.line_name.values:
                row = self._previous_results[self._previous_results.line_name == line_name].iloc[0].to_dict() #assuming all the rows share the same values
                row.update({["X","Y","objs"][n]: value for n,value in enumerate(np.array(self.spectra_dict[row.band]["wavelength"]), np.array(self.spectra_dict[row.band]["flux"]),self.spectra_dict[row.band]["obj"])})
                self.interactive_plot(**row)
            for band in self.spectra_dict.keys():
                X,Y,objs = np.array(self.spectra_dict[band]["wavelength"]), np.array(self.spectra_dict[band]["flux"]),self.spectra_dict[band]["obj"]
                if center_window < np.min(X) or center_window > np.max(X):
                    continue
                else:
                    #print( center_window,np.min(X),np.max(X))
                    self.interactive_plot(X,Y,objs,center_window,window,lr_init,lc_init,core_init,band,line_name)
    
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
        
        