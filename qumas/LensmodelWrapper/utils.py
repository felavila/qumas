import pandas as pd
import numpy as np
from copy import deepcopy
import re
import os
#from .modeling import *
import pickle
#import h5py

module_dir = os.path.dirname(os.path.abspath(__file__))


def write_list_to_file(file_path, data_lines):
    """
    Write the content of a list to a file.
    
    Args:
        file_path (str): Path to the file to save.
        data_lines (list of str): Lines to write to the file.
    """
    try:
        with open(file_path, "w") as file:
            file.writelines(data_lines)
        print(f"File successfully written to: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def clean_number(txt):
    if txt == "-":
        return np.nan
    if "/" in str(txt):
        return np.nan
    else:
        return str(txt).translate(str.maketrans('', '', '()~?'))
    
    
def bands_that_can_be_modeled(L:pd.DataFrame,print_bands=False,keyword="IS"):
    "given a panda data frame for and unique object choose the best band for our experiment e.g. I/F814/i"
    
    try:
        L=L.loc[L.index.get_level_values('type')=="ima"]
    except:
        L=L.loc[L[keyword]=="ima"]
    bands = [i for i in L.columns if "band" in i if sum(~np.isnan(L[i].replace("-",np.nan).astype(float)))==len(L[L[keyword]=="ima"])]
    if print_bands:
        print(f"avaliable bans {bands}")
    if len(bands)>0:
        for i in ["band_i",'band_I',"band_F814W"]:
            if i in bands:
                bands = [i]+list(set(bands) - set([i]))
    else:
        bands = ["-"]
    return bands
#This function will require take in consideration the cases in where the number of images dosent mach the positions
def can_be_modeled(L:pd.DataFrame,keywords=["IS","lens"],to_pandas=True):
    "cbm:can be modeled ? given a panda data frame for and unique object "
    system_name = L.name.drop_duplicates().values[0]
    bands=bands_that_can_be_modeled(L)
    #maybe here will be require change the way is added the images
    try:
        imagenes=L[L[keywords[0]]=="ima"][["RA","DEC","dDEC","dRA"]].astype(float) #informacion de las imagenes
    except:
        imagenes=[] #informacion de las imagenes
    n_images = len(imagenes)
    dif=L[L[keywords[0]]==keywords[1]][[i for i in L[L[keywords[0]]==keywords[1]].columns if i not in ["RA","DEC","dDEC","dRA"]]]
    problems=[]
    cbmd=False
    conditions = {
    n_images == 0: "0 images",
    n_images == 1: "ring",
    n_images > 5: "more than 5 images",
    n_images == 3: "it has 3 images",
    bands == ["-"]: "no band information",
    len(dif) == 0: "no information about the galaxy"
}
    for condition, message in conditions.items():
        if condition:
            problems.append(message)
    cbmd = len(problems) == 0
    # If there are no problems, set problems to ["-"]
    if cbmd:
        problems = ["-"]
        if to_pandas==True:
            return [cbmd,problems,bands] 
    #np.tile(np.array([cbmd,*problems,bands]), (len(L),1 ))
        return [cbmd,system_name,problems]
    else:
        return [cbmd,problems,bands] 

def columns_to_float(system,band):
    dif= system[["RA","DEC","dDEC","dRA"]][system['IS']=="lens"][["RA","DEC","dDEC","dRA"]].astype(float)
    columns_to_float = ['RA', 'DEC', 'dDEC', 'dRA', band,band.replace("band","error")]
    images = system[system['IS'] == 'ima'][["component"] + ['RA', 'DEC', 'dDEC', 'dRA', band,band.replace("band","error")]].astype({col: float for col in columns_to_float})
    return images,dif

def is_know_redshift(Lenses:pd.DataFrame):
    "Given a data frame of lens system look for redshift in the system in the current table and replace - for 0.5 or 2.0"
    if "z_l" not in Lenses.columns:
        Lenses["z_l"] = ["-"] * len(Lenses)
    if "z_s" not in Lenses.columns:
        Lenses["z_s"] = ["-"] * len(Lenses)
    Lenses["zl_known"] = [bool(zl != "-" and ~np.isnan(float(clean_number(zl)))) for zl in Lenses["z_l"].values]
    Lenses["zs_known"] = [bool(zs != "-" and ~np.isnan(float(clean_number(zs)))) for zs in Lenses["z_s"].values]
    Lenses["old_z_s"] = [clean_number(zs) if known else np.nan for known,zs in Lenses[["zs_known","z_s"]].values]
    Lenses["old_z_l"] = [clean_number(zl) if known else np.nan for known,zl in Lenses[["zl_known","z_l"]].values]
    Lenses["z_s"] = [clean_number(zs) if known else 2.0 for known,zs in Lenses[["zs_known","z_s"]].values]
    Lenses["z_l"] = [clean_number(zl) if known else 0.5 for known,zl in Lenses[["zl_known","z_l"]].values]
    return Lenses

def model_available(imagenes:float,dif:float):
    #all use SIE
    Models=[]
    if imagenes==1:
        Models=["SIS"]
    if imagenes==2:
        Models=["SIS","SIE","SIS+shear"]
    if imagenes==3:
        Models=["SIS","SIE","SIS+shear"]
    if imagenes==4:
        Models=["SIS","SIE","SIE+shear","POW","POW+shear"]
        if dif>1:
            Models.append("SIE-2G")
    if imagenes>=5 and dif>1:
        Models.append("SIE-2G")
    return Models

def image_whit_photometry(L:pd.DataFrame,verbose=False):
    system = deepcopy(L)
    img_all = system[system["IS"]=="ima"]["component"].values
    img_can_be_moleded = system[system["IS"]=="ima"].dropna(subset=[i for i in system.columns if "band" in i], how='all')
    if len(img_can_be_moleded)==0:
        if verbose:
            print(f"No image for system {system.name.values[0]}")
        return img_can_be_moleded
    
    system["total_images"] = np.repeat(len(img_all),len(system))
    system_clean = pd.concat([img_can_be_moleded, system[system["IS"]!="ima"]], ignore_index=True)
    image_cant_be_modeled =list(set(system[system["IS"]=="ima"]["component"].values)-set(img_can_be_moleded["component"].values))
    system_clean["model_images"] = np.repeat(len(img_can_be_moleded),len(system_clean))
    if len(image_cant_be_modeled)==0:
        image_cant_be_modeled = ["-"]
    system_clean["image_cant_be_modeled"] = len(system_clean)*image_cant_be_modeled
    system_clean = system_clean.dropna(subset=["RA","DEC"])
    system_clean = system_clean.drop_duplicates(subset=['RA',"DEC"])
    system_clean["total_lens"] = np.repeat(len(system_clean[system_clean["IS"]=="lens"]),len(system_clean))
    system_clean["model_available"] = len(system_clean)*[model_available(system_clean.model_images.values[0],system_clean.total_lens.values[0])]
    cbmd,problems,bands = can_be_modeled(system_clean)
    system_clean["can_be_modeled"] =len(system_clean) * [cbmd]
    system_clean["issues"] =len(system_clean) * problems
    system_clean["available_bands"] =len(system_clean) * [bands]
    return system_clean

def pandas_to_model(Lenses:pd.DataFrame,verbose=False,only_astrometry=False):
    if "IS" not in Lenses.columns:
        if verbose:
            print("IS not in columns")
        Lenses["can_be_modeled"] = [False]*len(Lenses)
        return Lenses
    if "ima" not in Lenses.IS.values:
        return Lenses
    if not any(["G" in str(i) for i in Lenses["component"].values]):
        if verbose:
            print("not G in the system")
        Lenses["can_be_modeled"] = [False]*len(Lenses)
        return Lenses
    Lenses_f = pd.concat([image_whit_photometry(Lenses[Lenses["name"]==name]) for name in Lenses.name.drop_duplicates()],ignore_index=True)
    if len(Lenses_f)==0:
        Lenses["can_be_modeled"] = [False]*len(Lenses)
        return Lenses
    Lenses_f = is_know_redshift(deepcopy(Lenses_f))
    if 'photometric_system' not in Lenses_f.columns:
        Lenses_f['photometric_system'] = None
    if only_astrometry:
        Lenses_f = Lenses_f[['name', 'z_l', 'z_s', 'ra', 'dec', 'component', 'IS', 'RA', 'dRA','DEC', 'dDEC','Bibcode', 'file', 'year', 'Bibcode_zl', 'Bibcode_zs','known_names', 'can_be_modeled', 'total_images', 'model_images','image_cant_be_modeled', 'total_lens', 'model_available', 'issues', 'zl_known', 'zs_known', 'old_z_s', 'old_z_l','photometric_system','Telescope', 'instrument']]
    return Lenses_f



def get_paths_before(target_path):
    # Get the absolute path of the target
    target_abs_path = os.path.abspath(target_path)
    
    # Split the path into components
    path_components = target_abs_path.split(os.sep)
    
    # Remove the last component (the target)
    path_components.pop()
    
    # Reconstruct the paths before the target
    paths_before = []
    for i in range(len(path_components)):
        paths_before.append(os.sep.join(path_components[:i+1]))
    
    return paths_before


def make_dictionary(list,prefix="p"):
    return {f"{prefix}[{i}]":float(list[i]) for i in range(len(list))}

def get_numeric_values(text):
    numeric_values=re.findall(r'[-+]?\d*\.\d+|\d+|\(?\s*[-+]?\d*\.?\d*e[+-]?\d*\s*\)?', text)
    return numeric_values

def get_images_result(list_lensmodel):
    images_result = {"ra_imput":[],"dec_imput":[],"radec_error":[],"flux_imput":[],"flux_error":[],"time_delay_imput":[],"time_delay_error":[],"ra_output":[],"dec_output":[],"flux_output":[],"time_delay_output":[]}
    for j,imagen in enumerate(list_lensmodel):
        if imagen=='\n':
            continue
        imagen = imagen.replace("\n","").split("->")
        numeric_values =  [i.replace(")","").replace("(","") for i in  get_numeric_values(imagen[0])]+[float(i) for i in get_numeric_values(imagen[1])]
        for i,key in enumerate(images_result.keys()): images_result[key].append(float(numeric_values[i]))
    return images_result

def get_kappa_gamma(list_of_lines):
    kappa_gamma = {"index":[],"x":[],"y":[],"kappa":[],"gamma":[],"theta":[]}
    for j in range(2,len(list_of_lines)):
        linea = [i for i in list_of_lines[j].replace("\n","").split(" ") if i!=""]
        for i,key in enumerate(kappa_gamma.keys()): kappa_gamma[key].append(float(linea[i]))
    kappa_gamma = {key: np.array(value) for key, value in kappa_gamma.items()}
    kappa_gamma["magnification"] =((1-kappa_gamma['kappa'])**2 - kappa_gamma['gamma']**2)**-1
    return {"kappa_gamma":kappa_gamma}

def get_result_lensmodel(list_of_lines):
    possible_keys=['LENS PARMS','SOURCE PARMS','CHISQ','Source','images','Extra model images']
    where_appear= np.array([[pk,i] for pk in possible_keys for i,line in enumerate(list_of_lines) if pk == line.split(":")[0]])
    where_appear_final = np.vstack((where_appear.T,np.hstack((where_appear[1:,1],-1)))).T
    result = {str(k):{} for k in where_appear_final[:,0]}
    for k,f_i,u_i in where_appear_final:
        #print(k,f_i,u_i)
        local_list = list_of_lines[int(f_i):int(u_i)]
        if k=='LENS PARMS':
        #local_list = list_of_lines[int(f_i):int(u_i)]
            for i,line in enumerate(local_list):
                if "alpha" in line:
                    result["LENS PARMS"][f"alpha{i}"] = make_dictionary([i.replace("\n","") for i in line.split(" ") if (len(i)>0 and i!='alpha')])
        if k=="SOURCE PARMS":
            for i,line in enumerate(local_list):
                if "ptsrc" in line:
                        result["SOURCE PARMS"][f"ptsrc{i}"] = make_dictionary([i.replace("\n","") for i in line.split(" ") if (len(i)>0 and i!='ptsrc')],prefix="s")
        if k=="CHISQ":
            chi_values,chi_columns = list_of_lines[int(f_i)].split("#")
            result["CHISQ"]= {v: float(k) for k, v in zip(chi_values.replace("CHISQ:","").split(" ")[1:-1],chi_columns.replace("\n","").split(" ")[1:])}
            result["Source"] = {local_list[i].replace("\n","").split(":")[0].replace(" ",""):local_list[i].replace("\n","").split(":")[1]  for i in range(3,len(local_list[1:])+1)} #only works for one source 
        if k=="images":
            result["images"] = get_images_result(local_list[1:])
    return result
def get_critical_caustic(list_of_lines):
    return {"critical_caustic":{key:np.array([[i for i in i.split(" ")[1:] if len(i)>0] for i in [i.replace("\n","") for i in list_of_lines][1:] if len(i)>0],dtype="object").astype(float)[::,n] for n,key  in enumerate(["x","y","u","v"])}}

def get_grid(list_of_lines):
    return {"grid":{key:np.array([[i for i in i.split(" ") if len(i)>0] for i in [i.replace("\n","") for i in list_of_lines][1:] if len(i)>0],dtype="object").astype(float)[::,n] for n,key  in enumerate(["x","y","u","v"])}}

def get_RE(list_of_lines):
   re = {'RE':float(list_of_lines[0].split(" ")[0])}
   return re

def full_modeling_result(modeling_path,model_setup,remove_files=True):
    #where can add the name of the system xd
    model_result = {}
    ###
    if "final_step.start" in os.listdir(modeling_path):
            model_result["best_model_file"] = open(os.path.join(modeling_path,"final_step.start"), "r").readlines()
    if f"{modeling_path.split('/')[-1]}.dat" in os.listdir(modeling_path):
        model_result["model_setup_file"] = open(os.path.join(modeling_path,f"{modeling_path.split('/')[-1]}.dat"), "r").readlines()
    if f"{model_setup['name_run']}.dat" in os.listdir(modeling_path):
        model_result["modeling_file"] = open(os.path.join(modeling_path,f"{model_setup['name_run']}.dat"), "r").readlines()
    if "final_step.dat" in os.listdir(modeling_path):
        model_result["final_step_file"] = open(os.path.join(modeling_path,"final_step.dat"), "r").readlines()
    if "final_step.best" in os.listdir(modeling_path):
        model_result["final_step_file"] = open(os.path.join(modeling_path,"final_step.best"), "r").readlines()
    ####
    for step in ["step_1","step_2","final_step","kappa_gamma","critical_caustic","grid","RE"]: #only step+"kappa_gamma","critical_caustic","grid"   
        file_name = f"{step}.dat"
        if "step" in step:
            func = get_result_lensmodel
            if f"{step}.best" in os.listdir(modeling_path):
                file_name = f"{step}.best"
        elif step=="kappa_gamma":
            func = get_kappa_gamma
        elif step=="critical_caustic":
            func = get_critical_caustic
        elif step=="grid":
            func = get_grid
        elif step=="RE":
            func =get_RE
        if file_name not in  os.listdir(modeling_path):
            continue
        file = open(os.path.join(modeling_path,file_name), "r")
        list_of_lines = file.readlines()
        model_result[step] = func(list_of_lines)
        if remove_files:
            [os.remove(os.path.join(modeling_path,i) )for i in os.listdir(modeling_path) if step in i]
    model_result["model_setup"] = {"model_setup":model_setup}
    return model_result


def compare_dicts(dict1, dict2):
    # Check if both dictionaries have the same keys
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    
    # Check if the values for each key are the same
    false_key=[]
    for key in dict1.keys():
        if np.all(dict1[key] != dict2[key]):
            false_key.append(key)
    if len(false_key)>0:
        #print(false_key)
        return False
    return True




def read_pickle(file_path):
    with open(file_path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def write_pickle(data,file_path):
    #maybe add and istance dic?
    with open(file_path, 'wb') as handle:
        #print(filename.replace("hdf5","pickle"))
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

class ModelNotFoundError(Exception):
    def __init__(self, model, available_models):
        message = f"The model '{model}' is not in the lensmodel keys list: {available_models}"
        super().__init__(message)

