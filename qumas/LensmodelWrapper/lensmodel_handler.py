import os
from pathlib import Path
import numpy as np
from scipy.spatial import distance
import json
import stat
import subprocess
from .utils import columns_to_float
import pandas as pd
from .mass_models import mass_models

# Get the path to the current file
module_dir = Path(__file__).resolve().parent

# Navigate to the root of the qumas package (i.e., one level up from LensmodelWrapper)
qumas_root = module_dir.parent

# Define the path to the CSV file (e.g., in the data/ subdirectory)
csv_path = qumas_root / "Tables" / "filter_unique.csv"
class lensmodel_handler:
    """modeling_path: where will go the results of our modeling
    system:and pandas data frame from our census with all information to model our system
    center_mass_error
    relative_flux_error
    number_of_lens
    astrometry_error
    use_real_error_flux
    R_chi"""
    
    
    def __init__(self, modeling_path:str,system:pd.DataFrame
                 ,json_models=f"{module_dir}/models"): 
        self.model_setup_path = json_models
        self.modeling_path = modeling_path
        self.system = system
        self.system_name = self.system.name.values[0]
        self.n_images = int(self.system.model_images.values[0]) #images can be modeled
        self.mass_distribution = "SIE"
        ###set_up_data#####
        self.number_of_lens = 1
        self.zl,self.zs=[i for i in self.system[["z_l","z_s"]].values[0]]
        self.available_bands = self.system.available_bands.values[0]
        self.band_to_model = self.available_bands[0]
        self.images,self.lens = columns_to_float(self.system,self.band_to_model)
        #self.models_file = "double_models.json" if len(self.images)==2 else "quad_models.json"
        self.models_file = "quad_models.json" 
        with open(os.path.join(self.model_setup_path,self.models_file), 'r') as file_mass:
             self.mass_models = json.load(file_mass)
        #prior Er
        self.p1 = round(np.max(distance.pdist(np.array([self.images["RA"].values,self.images["DEC"].values]).T,metric='euclidean')/2),2)
        #if is not create yet create directories for the results and a directory in where will run the code.
        if modeling_path != None:
            self.data_path = os.path.join(self.modeling_path,f"{self.system_name}.dat")
            if not os.path.isdir(self.modeling_path):
                os.mkdir(self.modeling_path)
            #if os.path.isdir(os.path.join(self.modeling_path,"final_result"))==False:
             #   os.mkdir(os.path.join(self.modeling_path,"final_result"))
        else:
            self.data_path = None
    def __str__(self):
        return f"set up modeling for {self.system_name}"
    
    #def __call__(self):

    #maybe if moment to use **kwargs?
    def data_writter(self,use_real_error_flux=False,use_real_astrometry_error=False,free_center_of_mass=False, \
                    astrometry_error=0.003,center_mass_error=0.003,relative_flux_error=0.2,band_to_model=None,use_informed_flux=False):
        band_to_model = band_to_model if band_to_model in self.available_bands else self.band_to_model
        images, lens = columns_to_float(self.system, band_to_model) if band_to_model else (self.images, self.lens)
        #how can handle system with more image that the ones i want to use in the first model ?
        images_lens_centered = images[["RA","DEC"]].values-lens[["RA","DEC"]].values[0]
        lens_lens_centered = lens[["RA","DEC"]].values-lens[["RA","DEC"]].values[0]
        error_lens_lens_centered = np.sqrt(lens[["dRA","dDEC"]].values**2+lens[["dRA","dDEC"]].values[0]**2)
        magnitudes = images[band_to_model].values
        band_info = self.system.loc[images[band_to_model].index][["photometric_system","Telescope","instrument"]].values[0]
        if not isinstance(band_info[0],str):
            band_info[0] = "Vega"
        mask = ((filter_info["photometric_system"] == "Vega" if "v"  in band_info[0] else filter_info["photometric_system"] == band_info[0]) & (filter_info["Telescope"] == band_info[1]) &(filter_info["instrument"] == band_info[2]) & 
                (filter_info["band_to_model"] == band_to_model.replace("band_", "")))
        
        band_matching_rows = filter_info[mask]
        if len(band_matching_rows) == 0:
            print('no band_info specific band info for this case')
            band_info = {i: None for i in filter_info.columns}
            band_info["band_to_model"] = band_to_model.replace("band_", "")
        else:
            band_info = band_matching_rows.iloc[0].to_dict()
        #print(band_matching_rows.iloc[0].to_dict())
        if use_informed_flux:
            flux = magnitudes
        else:
            flux=np.array([((-1)**(ii))*10**(((-magnitudes[ii])+min(magnitudes))/2.5) for ii in range(self.n_images)]) # divided flux
        flux_error = relative_flux_error*np.abs(flux)
        if use_real_error_flux:
            flux_error = images[band_to_model.replace("band",'error')].values
            #return print("Not define yet a formula for the propagation of the error in the flux")
            #flux_error = image[2] #error propagation from flux
        astrometry_error = self.n_images*[astrometry_error]
        if use_real_astrometry_error:
            error_images_lens_centered = np.sqrt(images[["dRA","dDEC"]].values**2+lens[["dRA","dDEC"]].values[0]**2)
            #could be the max image error or the error in the specific image
            astrometry_error=np.max(error_images_lens_centered,axis=1)
            astrometry_error = self.n_images*[np.max(error_images_lens_centered.flatten())]
        number_of_lens = self.number_of_lens
        
        no_lens = True if number_of_lens<1 else False
        if free_center_of_mass:
            #what is the idea behind this ? maybe take the center of mass
            #between the two images ?
            # x1,x2=images_lens_centered
            # y1,y2=images_lens_centered
            # p1=np.array([x1,y1])
            # p2=np.array([x2,y2])
            # p3=np.array([0,0])
            # error_cm=[np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)]
            if len(images_lens_centered)==2:
                return print("not yet")
            if len(images_lens_centered)>2:
                return print("not yet")
        self.lens_lens_centered = lens_lens_centered
        if "-" in self.mass_distribution:
            number_of_lens = int(self.mass_distribution.split("-")[1][0])
            #print("number_of_lens",number_of_lens)
        if no_lens:
            return print("not yet")
        if isinstance(center_mass_error, float):
            center_mass_error = [center_mass_error]
            if number_of_lens>1:
                center_mass_error+= [0.5]*(number_of_lens-1)
        time_delay,time_delay_error = 0,1000
        if self.data_path:
            f = open(os.path.join(self.data_path), 'w')
            f.write(f"#The system name is: {self.system_name} \n")
            f.write(f"{number_of_lens} # number of lens galaxy\n")
            for ii in range(number_of_lens):
                ra,dec = lens_lens_centered[ii]
                f.write(f"{ra:.3f} {dec:.3f} {center_mass_error[ii]:.3f} {center_mass_error[ii]:.3f} #position \n")
                f.write(f"0.0 1000. # R_eff observed: 0.59 +/- 0.06 \n")
                f.write(f"0.0 1000. # PA unconstrained in observations \n")
                f.write(f"0.0 1000. # observed e < 0.07 at 1sigma \n")
            f.write(f"\n")
            f.write(f"1 # nºsource \n")
            f.write(f"{self.n_images} # n° images of the source \n")
            f.write(f"#band used {band_to_model} \n \n")
            f.write("#Ra Dec flux sigmax sigmaflux tdel sigma(tdel) part "+'\n') 
            for n,image in enumerate(images[["component",band_to_model,f"error_{band_to_model[5:]}"]].values): #peligroso
                ra,dec=images_lens_centered[n]
                #flux=((-1)**(ii))*10**(((-image[1])+min(magnitudes))/2.5) # divided flux
                #if np.max(image[1])<=5:
                #   flux=((1)**(ii))*image[1]
                #if np.max(image[1])>30:
                #   flux = ((-1)**(ii))*image[1]/np.max(image[1])
                f.write(f"{ra:.3f} {dec:.3f} {flux[n]:.3f} {astrometry_error[n]:.3f} {flux_error[n]:.3f} {time_delay:.3f} {time_delay_error:.3f} {image[0]} \n")
            f.close()
        data_model = {"name":self.system_name,"component":images["component"].values,"ra":images_lens_centered[:,0],"dec":images_lens_centered[:,1],"astrometry_error":astrometry_error \
                    ,"flux":flux,"flux_error":flux_error,"magnitudes":magnitudes,"band_to_model":band_to_model,"lens_ra":lens_lens_centered[:,0],"lens_dec":lens_lens_centered[:,1] \
                    ,"center_mass_error":center_mass_error,"zl":self.zl,"zs":self.zs}
        data_model.update(band_info)
        return data_model
    def model_writter(self,mass_distribution=None,name_run ="model_run",kwards_special={}):
        with open(os.path.join(self.model_setup_path,"model_setup.json"), 'r') as file:
            setup_model = json.load(file)
        if mass_distribution==None:
            mass_distribution = self.mass_distribution
        number_of_lens = self.number_of_lens
        if "-" in self.mass_distribution:
            number_of_lens = int(self.mass_distribution.split("-")[1][0])
            mass_distribution = self.mass_distribution.split("-")[0]
        if mass_distribution not in self.mass_models.keys():
                return print(f"{mass_distribution} mass model is not available")
        mass = self.mass_models[mass_distribution]
        if self.data_path != None:
            f = open(os.path.join(self.modeling_path,f"{name_run}.dat"), 'w')
            f.write(f"#mass_distribution = {mass_distribution} \n")
            #this should be go to data_model
            kwards_basic = {"set_data":self.data_path,'set_zlens' :self.zl,'set_zsrc' :self.zs,"set_lens_number": number_of_lens}
            for keys in kwards_basic.keys(): setup_model["lensmodel_values"][keys] =kwards_basic[keys]
            for keys in kwards_special.keys(): setup_model["lensmodel_values"][keys] =kwards_special[keys]
            for key in setup_model["lensmodel_values"]: f.write(f"{setup_model['lensmodel_keys'][key]} {setup_model['lensmodel_values'][key]} {setup_model['lensmodel_coments'][key]}")
            f.write(mass_models(mass_distribution,number_of_lens, self.p1, self.lens_lens_centered, self.modeling_path))
            #for key in mass: f.write(f"{mass[key].replace('p1',str(self.p1)).replace('path',self.modeling_path)}")
            f.close()
        return
    
    def writter_model_data(self,mass_distribution=None, name_run="model_run",**kwards):
        #mass_distribution
        #if mass_distribution==None:
        #print(mass_distribution)
        self.mass_distribution = mass_distribution or self.mass_distribution 
        data_model = self.data_writter(**kwards)
        self.model_writter(mass_distribution=mass_distribution,name_run =name_run)
        data_model["name_run"]= name_run
        data_model["mass_distribution"]= mass_distribution
        return data_model