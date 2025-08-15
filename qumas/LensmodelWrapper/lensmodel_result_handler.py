import numpy as np 
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from .utils import ModelNotFoundError
#q=1-d_r["p[4]"][i]
#theta_E=np.sqrt((1+q**2)/(2*q))*d_r["p[1]"][i]#theta_E_gravlens





class Result_Handler(
):  
    mass_models_parameters = {'SIS':3,"SIE":5,"POW":6,"SIS+shear":5,"SIE+shear":7,"POW+shear":8}
    def __init__(self,lensmodel_system: dict,reduced_chi=False,max_separation=0.01,look_for_best_model=False,older_version=False):
        
        if "mcmc_chain" in lensmodel_system.keys():
            #print(self.lensmodel_system["model_name"]) 
            self.lensmodel_system = {}
            self.lensmodel_system[lensmodel_system["model_name"]] = lensmodel_system
        else:
            self.lensmodel_system = lensmodel_system
        #agregar system_name to model setup? 
        self.older_version = older_version 
        self.max_separation = max_separation
        self.reduced_chi = reduced_chi
        self.models = list(self.lensmodel_system.keys())
        self.pandas_from_results = pd.concat([self.make_pandas_from_results(model).reset_index(drop=True) for model in self.models])
        self.pandas_model_stats = self.get_stats()
        self.acepted_models = [model for model, sep in self.pandas_model_stats[["model_name","max(delta_images)"]].values if ((np.max(sep) <= self.max_separation)& (np.min(sep) != 0)) ]
        self.can_end = (len(self.acepted_models)) > 0
        if self.can_end:
            self.current_best = self.pandas_model_stats[[i in self.acepted_models for i in self.pandas_model_stats.model_name]].sort_values('log_Chi').iloc[0]
            #self.pandas_model_stats[[i in self.acepted_models for i in self.pandas_results.T["n_model"]]].T.loc["log_Chi"].astype(float).idxmin()                      
        else:
            self.current_best = self.pandas_model_stats.sort_values('log_Chi').iloc[0]
        self.current_best_n_model = self.current_best.model_name                   
        #self.panda_stats,self.pandas_full = self.make_pandas_result()
        #self.pandas_full_model_result = 
    
    def get_model_from_pandas(self,model:str=None):
        """model str"""
        model = model or self.current_best_n_model
        if model not in self.models:
            raise ModelNotFoundError(model, self.models)
        return self.pandas_from_results[self.pandas_from_results.model_name==model]
    
    def get_necessary_to_mcmc(self,model:str=None):
        """model str"""
        model = model or self.current_best_n_model
        if model not in self.models:
            raise ModelNotFoundError(model, self.models)
        dic_model = self.get_model(model=model)
        stats=self.pandas_model_stats[self.pandas_model_stats.model_name==model]
        return dic_model,stats.iloc[0]
    
    def get_model(self,model:str=None):
        """model str"""
        model = model or self.current_best_n_model
        if model not in self.models:
            raise ModelNotFoundError(model, self.models)
        return self.lensmodel_system[model]
    
    def get_lens_params(self,model:str=None):
        model = model or self.current_best_n_model
        if model not in self.models:
            raise ModelNotFoundError(model, self.models)
        local_model = self.get_model(model)
        print("mass_distribution:",local_model.get("model_setup").get("model_setup").get("mass_distribution"))
        return local_model.get('final_step').get('LENS PARMS')
    
    def make_pandas_from_results(self,model:str=None):
        model = model or self.current_best_n_model
        if model not in self.models:
            raise ModelNotFoundError(model, self.models)
        model_dic = self.get_model(model)
        model_k = model_dic["kappa_gamma"]["kappa_gamma"]
        model_h = model_dic['final_step']
        model_s = model_dic["model_setup"]["model_setup"]
        model_er = model_dic["RE"]["RE"]
        combined_df = pd.merge(pd.DataFrame(model_k), pd.DataFrame(model_h["images"]),left_on='x', right_on='ra_imput')
        #A = pd.DataFrame(list(model_h['LENS PARMS'].values()) * len(combined_df))
        #A.reset_index(drop=True)
        B =pd.DataFrame(list(model_h['SOURCE PARMS'].values()) * len(combined_df))
        C = pd.DataFrame([list(model_h['CHISQ'].values())]* len(combined_df),columns=['chis2_'+i for i in model_h['CHISQ'].keys()])
        D = pd.DataFrame([list(model_s.values())]* len(combined_df),columns=[i for i in model_s.keys()])
        pandas_comb = pd.concat([combined_df,B.reset_index(drop=True),C.reset_index(drop=True),D.reset_index(drop=True)], axis=1) 
        pandas_comb["magnitudes"] = model_s["magnitudes"]
        pandas_comb['astrometry_error'] = pandas_comb['astrometry_error'].values[0]
        pandas_comb['center_mass_error'] = np.array([*pandas_comb['center_mass_error'].values])
        pandas_comb['flux_error_by_image'] = pandas_comb['flux_error'].values.T[1][0]
        pandas_comb['flux_error'] = max(pandas_comb['flux_error_by_image'])
        pandas_comb = pandas_comb.loc[:, ~pandas_comb.columns.duplicated()]
        pandas_comb["einstein_radii"] = [model_er]*len(pandas_comb["magnitudes"])
        pandas_comb["magnitudes_corrected"] = -2.5*np.log10(10**(-(np.abs(pandas_comb['flux_output'])+ np.min(pandas_comb["magnitudes"]))/2.5)/np.abs(pandas_comb["magnification"]))
        pandas_comb["delta_images"] = np.sqrt( (pandas_comb['ra_imput'] - pandas_comb["ra_output"])**2+ (pandas_comb['dec_imput'] - pandas_comb["dec_output"])**2) #sqrt delta images
        pandas_comb["model_name"] = [model]*len(pandas_comb["magnitudes"])
        pandas_comb['log_Chi'] = abs(np.log10(pandas_comb['chis2_tot']))
        pandas_comb["median_einstein_radii"] = [np.nan]*len(pandas_comb["magnitudes"])
        pandas_comb["std_einstein_radii"] = [np.nan]*len(pandas_comb["magnitudes"])
        pandas_comb["component"] = model_s["component"]
        pandas_comb["ra"] = model_s["ra"]
        pandas_comb["dec"] = model_s["dec"]
        if 'mcmc' in model_dic.keys():
            sampled_re = model_dic["mcmc"]['mcmc_chain']['RE'].values
            pandas_comb["median_einstein_radii"] = [np.median(sampled_re)]*len(pandas_comb["magnitudes"])
            pandas_comb["std_einstein_radii"] = [np.std(sampled_re)]*len(pandas_comb["magnitudes"])
        return pandas_comb[["component"] + [col for col in pandas_comb.columns if col not in ["component", "index"]]]

   #Chi = {key : self.get_chis2(key)['tot'] for key in self.models}
    #     log_Chi = {key : abs(np.log10(float(self.get_chis2(key)['tot']))) for key in self.models}
    #     pandas_results = pd.DataFrame([lensmodel_system,BIC,AICc,AIC,n_model,mass_models,mass_models_parameters,einstein_radii,rmse,max_delta_images,median_demag,std_demag,Chi,log_Chi]
    #                                   ,index=["name","BIC","AICc","AIC","n_model","mass_models","mass_models_parameters","einstein_radii","rmse","max(delta_images)","median_demag","std_demag","Chi","log_Chi"])
    #   
    
    def get_stats(self):
        pandas_stats = []
        relevant_keys = ["name","model_name",'mass_distribution','einstein_radii','median_einstein_radii','std_einstein_radii','flux_error','astrometry_error','center_mass_error', \
            'band_to_model','chis2_tot', 'chis2_pos','chis2_flux','log_Chi']
        if not self.older_version:
            relevant_keys += ['zl', 'zs', 'photometric_system', 'Telescope', 'instrument', 'zpt', 'lambda_cen']
        for model_name in self.models:
            panda_loc = self.pandas_from_results[self.pandas_from_results['model_name']==model_name]
            #print(model_name,max(panda_loc.delta_images))
            max_sep = max(panda_loc.delta_images)
            if max_sep > self.max_separation:
                max_sep = np.round(max(panda_loc.delta_images),2)
            median_demag =  np.median(panda_loc.magnitudes_corrected)
            std_demag =  np.std(panda_loc.magnitudes_corrected)
            #print(panda_loc[["model_name",'einstein_radius']].drop_duplicates().values)
            pandas_stats.append([*panda_loc[relevant_keys].drop_duplicates().values[0],max_sep,median_demag,std_demag])
        return pd.DataFrame(pandas_stats,columns=relevant_keys+["max(delta_images)",'median_demag','std_demag'])
    
    def make_plot(self, model=None, add_info=False, add_critical=False, save='',remove_axis=False, x_zoom_list=None, y_zoom_list=None, zoom_size=0.2, zoom_positions=None):
        """
        Creates a scatter plot with optional zoomed inset subplots.

        Parameters:
        - model: The model to plot.
        - add_info: Whether to add additional info text.
        - add_critical: Whether to add critical curves.
        - save: Filename to save the plot.
        - remove_axis: Whether to remove the main plot axes.
        - x_zoom_list: List of tuples defining x-limits for zoom regions, e.g., [(x1_min, x1_max), (x2_min, x2_max)].
        - y_zoom_list: List of tuples defining y-limits for zoom regions, e.g., [(y1_min, y1_max), (y2_min, y2_max)].
        - zoom_size: Size of the inset axes relative to the main plot (default is 0.2).
        - zoom_positions: List of tuples defining the position of each inset axes in [left, bottom, width, height] format.
                        If None, positions will be automatically arranged at the bottom.
        """
        model = model or self.current_best_n_model
        if model not in self.models:
            raise ModelNotFoundError(model, self.models)
        model_dic = self.lensmodel_system[model]
        ra_imput = model_dic['final_step']['images']['ra_imput']
        dec_imput = model_dic['final_step']['images']['dec_imput']

        ra_output = model_dic['final_step']['images']['ra_output']
        dec_output = model_dic['final_step']['images']['dec_output']
        ra_lens_model, dec_lens_model = [], []
        ra_lens_in, dec_lens_in = [], []
        
        for i, key in enumerate(model_dic['final_step']['LENS PARMS'].keys()):
            ra_lens_model.append(model_dic['final_step']['LENS PARMS'][key]['p[1]'])
            dec_lens_model.append(model_dic['final_step']['LENS PARMS'][key]['p[2]'])
            ra_lens_in.append(model_dic["model_setup"]["model_setup"]['lens_ra'][i])
            dec_lens_in.append(model_dic["model_setup"]["model_setup"]['lens_dec'][i])
        components = model_dic["model_setup"]['model_setup']['component']
        num_zooms = 0
        if x_zoom_list and y_zoom_list:
            if len(x_zoom_list) != len(y_zoom_list):
                raise ValueError("x_zoom_list and y_zoom_list must have the same length.")
            else:
                num_zooms = len(x_zoom_list)
        fig, ax = plt.subplots(1+num_zooms, 1, figsize=(20, 15*(1+num_zooms)))
        ax = np.atleast_1d(ax)
        ax[0].scatter(ra_imput, dec_imput, label="Input Coordinates", facecolors='none', edgecolors='blue')
        ax[0].scatter(ra_output, dec_output, label="Output Coordinates", color="g", alpha=0.7)
        ax[0].scatter(ra_lens_model, dec_lens_model, label="Output Coordinates Lens", color="k", alpha=0.7)
        ax[0].scatter(ra_lens_in, dec_lens_in, label="Input Coordinates Lens", facecolors='none', edgecolors='red')

        # Add text labels
        for i, v in enumerate(components):
            ax[0].text(ra_imput[i], dec_imput[i], v, fontsize=40, alpha=0.8, horizontalalignment="right")

        # Add critical curves if requested
        if add_critical:
            critical = model_dic["critical_caustic"]["critical_caustic"]
            x = critical["x"]
            u = critical["u"]
            v = critical["v"]
            y = critical["y"]
            step = np.argmax(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            ax[0].plot(x[:step+1], y[:step+1], label="Critical Curves", alpha=0.5)
            ax[0].plot(u[:step+1], v[:step+1], label="Caustic Curves", alpha=0.5)

        # Add additional info text if requested
        if add_info:
            stats = self.pandas_model_stats[self.pandas_model_stats.model_name == model]
            info_text = rf"Mass distribution:{stats.mass_distribution.values[0]}"+"\n"+rf"$max(\Delta images) = {stats['max(delta_images)'].values[0]:.3f}$"+"\n"+fr"$\chi^2 (total) = {stats['chis2_tot'].values[0]}$"

            ax[0].text(
                ax[0].get_xlim()[0] - ax[0].get_xlim()[1] * 0.2,
                ax[0].get_ylim()[0] + ax[0].get_ylim()[1] * 0.11,
                info_text,
                fontsize=20
            )

        ax[0].set_ylabel(r"$\Delta \delta \quad [\mathrm{arcsec}]$")
        ax[0].set_xlabel(r"$\Delta \alpha \quad [\mathrm{arcsec}]$")
        ax[0].invert_xaxis()
        ax[0].legend(loc="upper left", bbox_to_anchor=(1.05, 0.75), borderaxespad=0)  # Legend outside
        
        # Remove axes if requested
        if remove_axis:
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Remove ticks
            ax[0].set_xticks([])
            ax[0].set_yticks([])

            # Optionally remove tick labels
            ax[0].set_xticklabels([])
            ax[0].set_yticklabels([])
        if save:
            plt.savefig(f"{save}.jpg")
            plt.close()
        else:
            plt.show()
        
        # # Handle zoomed inset plots
        # if x_zoom_list and y_zoom_list:
        #     if len(x_zoom_list) != len(y_zoom_list):
        #         raise ValueError("x_zoom_list and y_zoom_list must have the same length.")

        #     num_zooms = len(x_zoom_list)

            
        #     # Define default positions if not provided
        #     if not zoom_positions:
        #         # Arrange zooms horizontally at the bottom
        #         zoom_positions = []
        #         spacing = 0.05  # Spacing between insets
        #         total_width = num_zooms * zoom_size + (num_zooms - 1) * spacing
        #         start_x = 0.5 - total_width / 2  # Center the zooms
        #         for i in range(num_zooms):
        #             zoom_positions.append([start_x + i * (zoom_size + spacing), 0.05, zoom_size, zoom_size])

        #     for i in range(num_zooms):
        #         zoom_ax = inset_axes(
        #             ax,
        #             width=f"{zoom_size * 100}%",  # width as a percentage of parent_bbox
        #             height=f"{zoom_size * 100}%",
        #             loc='lower center',
        #             bbox_to_anchor=(zoom_positions[i][0], zoom_positions[i][1], zoom_positions[i][2], zoom_positions[i][3]),
        #             bbox_transform=ax.transAxes,
        #             borderpad=1
        #         )

        #         # Scatter plots in zoomed inset
        #         zoom_ax.scatter(ra_imput, dec_imput, facecolors='none', edgecolors='blue')
        #         zoom_ax.scatter(ra_output, dec_output, color="g", alpha=0.7)
        #         zoom_ax.scatter(ra_lens_model, dec_lens_model, color="k", alpha=0.7)
        #         zoom_ax.scatter(ra_lens_in, dec_lens_in, facecolors='none', edgecolors='red')

        #         # Set limits for zoom
        #         zoom_ax.set_xlim(x_zoom_list[i])
        #         zoom_ax.set_ylim(y_zoom_list[i])

        #         # Optional: Remove ticks from inset
        #         zoom_ax.set_xticks([])
        #         zoom_ax.set_yticks([])

        #         # Optional: Add rectangle on the main plot to indicate zoom area
        #         rect = plt.Rectangle(
        #             (x_zoom_list[i][0], y_zoom_list[i][0]),
        #             x_zoom_list[i][1] - x_zoom_list[i][0],
        #             y_zoom_list[i][1] - y_zoom_list[i][0],
        #             linewidth=1, edgecolor='red', facecolor='none', linestyle='--'
        #         )
        #         ax.add_patch(rect)

        #         # Optionally, connect the inset to the zoom area
        #         # mark_inset(ax, zoom_ax, loc1=2, loc2=4, fc="none", ec="0.5")

        # Save or show the plot
        
    
    # def make_plot(self,model=None,add_info=False,add_critial=False,save='',remove_axis=False):
    #     model = model or self.current_best_n_model
    #     if model not in self.models:
    #         raise ModelNotFoundError(model, self.models)
    #     model_dic = self.lensmodel_system[model]
    #     ra_imput =model_dic['final_step']['images']['ra_imput']
    #     dec_imput =model_dic['final_step']['images']['dec_imput']

    #     ra_output =model_dic['final_step']['images']['ra_output']
    #     dec_output = model_dic['final_step']['images']['dec_output']
    #     ra_lens_model,dec_lens_model = [],[]
    #     ra_lens_in,dec_lens_in = [],[]
    #     for i,key in enumerate(model_dic['final_step']['LENS PARMS'].keys()):
    #         ra_lens_model.append(model_dic['final_step']['LENS PARMS'][key]['p[1]'])
    #         dec_lens_model.append(model_dic['final_step']['LENS PARMS'][key]['p[2]'])#, model_dic['final_step']['LENS PARMS']['alpha1']['p[2]']
    #         ra_lens_in.append( model_dic["model_setup"]["model_setup"]['lens_ra'][i]),dec_lens_in.append( model_dic["model_setup"]["model_setup"]['lens_dec'][i])
    #     #ra_lens_in,dec_lens_in = model_dic["model_setup"]["model_setup"]['lens_ra'][0],model_dic["model_setup"]["model_setup"]['lens_dec'][0]

    #     fig, axes = plt.subplots(1, 1, figsize=(20, 10))
    #     axes.scatter(ra_imput,dec_imput,label="input coordinates",facecolors='none', edgecolors='blue')
    #     axes.scatter(ra_output,dec_output,label="output coordinates",color="g",alpha=0.7)
    #     axes.scatter(ra_lens_model,dec_lens_model,label="output coordinates lens",color="k",alpha=0.7)
    #     axes.scatter(ra_lens_in,dec_lens_in,label="input coordinates lens",facecolors='none', edgecolors='red')
    #     #print(component)
    #     [axes.text(ra_imput[i],dec_imput[i],v,fontsize=40,alpha=0.8,horizontalalignment="right") for i,v in enumerate(model_dic["model_setup"]['model_setup']['component'])]
        
    #     if add_critial:
    #         x = model_dic["critical_caustic"]["critical_caustic"]["x"]
    #         u = model_dic["critical_caustic"]["critical_caustic"]["u"]
    #         v = model_dic["critical_caustic"]["critical_caustic"]["v"]
    #         y = model_dic["critical_caustic"]["critical_caustic"]["y"]
    #         step = np.argmax(np.sqrt(np.diff(x)**2+np.diff(y)**2))
    #         axes.plot(x[:step+1],y[:step+1],label="critical curves",alpha=0.5)
    #         axes.plot(u[:step+1],v[:step+1],label="caustic curves",alpha=0.5)
    #     if add_info:
    #         stats=self.pandas_model_stats[self.pandas_model_stats.model_name==model]
    #         axes.text(axes.get_xlim()[0]-axes.get_xlim()[1]*0.2,axes.get_ylim()[0]+axes.get_ylim()[1]*0.11,rf"Mass distribution:{stats.mass_distribution.values[0]}"+"\n"+rf"$max(\Delta images) = {stats['max(delta_images)'].values[0]:.3f}$"+"\n"+fr"$\chi^2 (total) = {stats['chis2_tot'].values[0]}$",fontsize=20)
    #     axes.set_ylabel(r"$\Delta \delta \quad [\mathrm{arcsec}]$")
    #     axes.set_xlabel(r"$\Delta \alpha \quad [\mathrm{arcsec}]$")
    #     axes.invert_xaxis()
    #     axes.legend(loc="upper left", bbox_to_anchor=(1.05, 0.75), borderaxespad=0)  # Legend outside
    #     plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make room for the legend
    #     if remove_axis:
    #         for spine in axes.spines.values():
    #             spine.set_visible(False)

    #         # Remove ticks
    #         axes.set_xticks([])
    #         axes.set_yticks([])

    #         # Optionally remove tick labels
    #         axes.set_xticklabels([])
    #         axes.set_yticklabels([])
        
    #     if save:
    #         plt.savefig(f"{save}.jpg")
    #         plt.close()
    #     else:
    #         plt.show()
    
    
    
    
    #return model_dic 
    
    # def make_pandas_result(self):
    #     pandas_stats = []
    #     pandas_full = []
    #     for model_name in self.models:
    #         #delta_images = self.get_delta_images(model_name)
    #         #max_delta_images = np.max(abs(delta_images))
    #         pandas_stats.append([model_name,max_delta_images])
        
    #     return pd.DataFrame(pandas_stats,columns=['model_name',"max_delta_images"]),pandas_full
        #self.delta_images = {i : self.get_delta_images(i) for i in self.models}
        #self.pandas_results = self.make_pandas_model_stats(self.models)
        
        #self.rmse = {i : self.get_rmse(i) for i in self.models}
        #self.mass_models = {i : self.get_model_setup(i)['mass_distribution'] for i in self.models}
        #self.data_constrain = {i : self.get_constrains(i) for i in self.models}
        #self.einstein_radii = {i : self.get_einstein_radii(i) for i in self.models}
        #self.free_parameters = {i : self.data_constrain[i] - self.mass_models_parameters[self.mass_models[i]] for i in self.models}
        #self.demag = {i : self.get_demag(i) for i in self.models}
        #self.chis2 = pd.DataFrame({i : self.get_chis2(i,reduced_chi=self.reduced_chi) for i in self.models})
        #self.chis2["reduced_chi"] = [self.reduced_chi]*len(pd.DataFrame({i : self.get_chis2(i,reduced_chi=self.reduced_chi) for i in self.models}))
        #self.BIC = {i : self.get_BIC(i) for i in self.models}
        #self.AIC = {i : self.get_AIC(i) for i in self.models}
        #self.AICc = {i : self.get_AIC(i,aicc=True) for i in self.models}
        #self.einstein_radii = {i : self.get_einstein_radii(i) for i in self.models}
        #self.JJV = {i : self.get_JJV(i) for i in self.models}
        
        # self.acepted_models = [key for key, sep in self.delta_images.items() if ((np.max(sep) <= self.max_separation)& (np.min(sep) != 0)) ]
        # self.can_end = (len(self.acepted_models)) > 0
        
        # if self.can_end:
        #     self.current_best =self.pandas_results.T[[i in self.acepted_models for i in self.pandas_results.T["n_model"]]].T.loc["log_Chi"].astype(float).idxmin()                      
        # else:
        #     self.current_best = self.pandas_results.loc["Chi"].astype(float).idxmin()
        # self.current_best_info = self.pandas_results[self.current_best]
        # self.current_best_n_model = self.current_best_info.n_model
        #self.best_current_model = self.get_model(self.current_best_info.n_model)   #nnn not sure about this 
        #self.best_current_model["model"] = self.current_best_info.n_model
        #self.pandas_best = self.pandas_results.loc[self.current_best]
    #def get_info_to_mcmc(self,model):
     #   model = self.get_model(model)
        
    # def get_stats(self,model=None):
    #     model = model or self.current_best_n_model
    #     if model not in self.models:
    #         raise ModelNotFoundError(model, self.models)
    #     return self.pandas_results.T[self.pandas_results.T.n_model==model]
    
   
    
    # def get_mcmc_info(self,model):
    #     model = model or self.current_best_n_model
    #     if model not in self.models:
    #         raise ModelNotFoundError(model, self.models)
    #     dic_model = self.get_model(model=model)
    #     basic_info = self.pandas_results.T[self.pandas_results.T.n_model==model].iloc[0].to_dict()
    #     basic_info.update({keys:dic_model["model_setup"]["model_setup"][keys] for keys in ["zl","zs"]})
    #     #basic_info.update(keys:self.pandas_results.T[pandas_results.T.n_model==model])
    #     return dic_model,basic_info

    
            
    # def get_JJV(self,model=None):
    #     #mmmm
    #     model = model or self.current_best_n_model
    #     if model not in self.models:
    #         raise ModelNotFoundError(model, self.models)
    #     model_h_ = deepcopy(self.get_model(model)["kappa_gamma"]["kappa_gamma"])
    #     dicts_ = deepcopy(self.get_model_setup(model))
    #     main_shape = np.array(dicts_['astrometry_error']).shape
    #     for key,i in dicts_.items():
    #         if np.array(i).shape!=main_shape:
    #             if isinstance(i,str):
    #                 dicts_[key] = np.array([i]*main_shape[0])
    #             else:
    #                 dicts_[key] = np.array(i*main_shape[0])
    #     model_h_.update(dicts_)
    #     model_h_.pop('lens_dec', None)
    #     model_h_.pop('lens_ra', None)
    #     return pd.DataFrame(model_h_)
    
    
    # def get_kappa_gamma(self,model):
    #     model_h = self.get_model(model)["kappa_gamma"]["kappa_gamma"]
    #     return model_h
    # def get_constrains(self,model):
    #     # based on https://iopscience.iop.org/article/10.1088/0004-637X/773/1/35/pdf 
    #     n,l = len(self.get_model_setup(model)['component']),len(self.get_model_setup(model)['lens_ra'])*2
    #     return 3*n -1 + l
    # def get_BIC(self,model,chikey="pos"):
    #     #Fotios K. Anagnostopoulos + 2019
    #     #BIC = -2ln(L) + kln(Ntot)
    #     # -2ln(L) == chi2  (TESTING THE DARK ENERGY WITH GRAVITATIONAL LENSING STATISTIC, Shuo Cao), prior uniforme (?)
    #     #k: number of parameters of the model 
    #     #Ntot number of contrictions o data of the model 
    #     return self.get_chis2(model,reduced_chi=False)[chikey] + self.mass_models_parameters[self.mass_models[model]] * np.log(self.data_constrain[model])
    # def get_AIC(self,model,chikey="pos",aicc=False):
    #     #https://en.wikipedia.org/wiki/Akaike_information_criterion
    #     #k: number of parameters of the model
    #     # -2ln(L) == chi2  (TESTING THE DARK ENERGY WITH GRAVITATIONAL LENSING STATISTIC, Shuo Cao), prior uniforme (?)
    #     # denotes the number of parameters
    #     k = self.mass_models_parameters[self.mass_models[model]]
    #     n = self.data_constrain[model]
    #     #n denotes the sample 
    #     AICc = 0 if aicc==False else (2*(k)**2+2*k)/(n-k-1)
    #     return self.get_chis2(model,reduced_chi=False)[chikey] +  AICc
    # def get_chis2(self,model,reduced_chi=False):
    #     model_h = self.get_model(model)['final_step']['CHISQ']
    #     if reduced_chi:
    #         return {key: values/self.free_parameters[model] for key,values in model_h.items()}
    #     return model_h
    # @staticmethod
    # def rmse(v1, v2):
    #         return np.sqrt(np.mean(np.sum((v1 - v2)**2)))
    
    # def get_rmse(self,model):
    #     model_h = self.get_model(model)['final_step']
    #     imput_x,imput_y = np.array(model_h['images']['ra_imput']),np.array(model_h['images']['dec_imput'])
    #     model_x,model_y = np.array(model_h['images']['ra_output']),np.array(model_h['images']['dec_output'])
    #     rms_x = Result_Handler.rmse(imput_x, model_x)
    #     rms_y = Result_Handler.rmse(imput_y, model_y)
    #     rms_radial = Result_Handler.rmse(np.hypot(model_x, model_y), np.hypot(imput_x, imput_y))
    #     return rms_radial
    
    # def get_demag(self,model):
    #     model_h = self.get_model_setup(model)
    #     #first we recover the new magnitudes from the out put results and then we apply the magnification correction
    #     magnitudes = model_h['magnitudes']
    #     model_f_out_put = np.abs(np.array(self.get_model(model)['final_step']['images']['flux_output']))
    #     new_magnitudes = -2.5*np.log10(model_f_out_put)+ np.min(magnitudes)
    #     magnification =  np.array(self.get_kappa_gamma(model)['magnification'])
    #     mag_corrected = -2.5*np.log10((10**(-new_magnitudes/2.5))/np.abs(magnification))
    #     return mag_corrected
    
    # # def get_best_model(self,max_separation=0.001,current_model=None,print_best=False,get_all=False,best_panda=False,only_good_models=False,**kwargs):
    # #     #
    # #     #pandas_results = pd.DataFrame({})
    # #     #if len(acepted_models)>0:
    # #      #   can_end = True
    # #       #  pandas_results = self.make_pandas_model_stats(acepted_models)
    # #        # current_best = pandas_results.loc["AIC"].astype(float).idxmax()
    # #        # if print_best:
    # #         #    print(f"The best model for {self.name} that have a separation bellow {max_separation} is {current_best}, it is the best model taking in consideration the separation and AIC")
    # #     #if get_all or len(acepted_models)==0 :
    # #     #acepted_models = 
    # #     self.pandas_results = self.make_pandas_model_stats(self.models)
    # #     self.current_best = self.pandas_results.loc["AIC"].astype(float).idxmax()
    # #     #if len(self.models)>0:
    # #      #   self.can_end = True
            
            
    # #         #print(f"Any model have a separation bellow {max_separation}, but the better one is {current_best} taking in consideration just AIC")
    # #     #if best_panda:
    # #      #   pandas_results = pandas_results[current_best]
    # #       #  if only_good_models and not can_end:
    # #        #     pandas_results = pd.DataFrame({})
    # #     #self.pandas_results = pandas_results
    # #     #self.current_best = current_best
    # #     #print(pandas_results[current_best].mass_models)
    # #     #return current_best,can_end,pandas_results

    # def make_pandas_model_stats(self,models,extra_fields=False):
    #     "given a list of models make a pandas data frame"
    #     lensmodel_system = {key: self.name for key in models}
    #     BIC = {key: self.BIC[key] for key in models}
    #     AICc = {key: self.AICc[key] for key in models}
    #     AIC = {key: self.AIC[key] for key in models}
    #     n_model = {key: key  for key in models}
    #     #except:
    #      #   n_model = {key: key.split("(")[0] for key in models}
    #     mass_models = {key: self.mass_models[key] for key in models}
    #     mass_models_parameters = {key: int(self.mass_models_parameters[self.mass_models[key]]) for key in models}
    #     einstein_radii = {key: self.einstein_radii[key] for key in models}
    #     demag = {key: self.demag[key] for key in models}
    #     median_demag = {key: np.median(demag[key]) for key in models}
    #     std_demag = {key: np.std(demag[key]) for key in models}
    #     rmse = {key: self.rmse[key] for key in models}
    #     max_delta_images = {key: np.max(self.delta_images[key]) for key in models}
    #     Chi = {key : self.get_chis2(key)['tot'] for key in self.models}
    #     log_Chi = {key : abs(np.log10(float(self.get_chis2(key)['tot']))) for key in self.models}
    #     pandas_results = pd.DataFrame([lensmodel_system,BIC,AICc,AIC,n_model,mass_models,mass_models_parameters,einstein_radii,rmse,max_delta_images,median_demag,std_demag,Chi,log_Chi]
    #                                   ,index=["name","BIC","AICc","AIC","n_model","mass_models","mass_models_parameters","einstein_radii","rmse","max(delta_images)","median_demag","std_demag","Chi","log_Chi"])
    #     pandas_results = pandas_results.rename(columns={key:f"{key}({mass_model})" for key,mass_model in mass_models.items()})
    #     if extra_fields:
    #         return pandas_results
    #     return  pandas_results
        
    # #    def make_pandas_model_stats(self,models):
    # #     "given a list of models make a pandas data frame"
    # #     BIC = {key: self.BIC[key] for key in models}
    # #     AICc = {key: self.AICc[key] for key in models}
    # #     AIC = {key: self.AIC[key] for key in models}
    # #     mass_models = {key: self.mass_models[key] for key in models}
    # #     mass_models_parameters = {key: int(self.mass_models_parameters[self.mass_models[key]]) for key in models}
    # #     einstein_radii = {key: self.einstein_radii[key] for key in models}
    # #     delta_images = {key: self.delta_images[key] for key in models}
    # #     rmse = {key: self.rmse[key] for key in models} 
    # #     demag = {key: self.demag[key] for key in models} 
    # #     pandas_results = pd.DataFrame([mass_models,BIC,AICc,AIC,mass_models_parameters,einstein_radii,delta_images,rmse,demag],\
    # #                                   index=["mass_models","BIC","AICc","AIC","mass_models_parameters","einstein_radii","delta_images","rmse","demag"])
    # #     pandas_results = pandas_results.rename(columns={key:f"{key}" for key,mass_model in mass_models.items()})
    # #     return  pandas_results
    # def get_einstein_radii(self,model):
    #     return self.get_model(model)["RE"]["RE"]
    
    
    
    
    
    # def get_pandas(self,model):
    #     return 