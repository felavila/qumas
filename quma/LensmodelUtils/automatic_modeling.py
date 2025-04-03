import os
import numpy as np
from .lensmodel_wrapper import run_lensmodel
from .lensmodel_handler import lensmodel_handler
from .utils import full_modeling_result,compare_dicts,write_pickle,read_pickle
#,get_paths_before #?
#run_lensmodel,save_dict_to_hdf5,full_modeling_result,read_hdf5,compare_dicts,write_pickle
from .lensmodel_result_handler import Result_Handler
from .mcmc_lensmodel import make_mcmc_lensmodel

#maybe use YAML 
def set_up_dir(path_for_models,dir_models,system_name):
    if not os.path.isdir(path_for_models):
        os.mkdir(path_for_models)
    if not os.path.isdir(os.path.join(path_for_models,dir_models)):
        os.mkdir(os.path.join(path_for_models,dir_models))
    if not os.path.isdir(os.path.join(path_for_models,dir_models,system_name)):
        os.mkdir(os.path.join(path_for_models,dir_models,system_name))
    return os.path.join(path_for_models,dir_models,system_name)
# To do change all the save and reading to pickle

def automatic_modeling(system,max_separation=0.01,path_for_models=None,dir_models=None,model_All=False,do_mcmc=False,max_lap=10,relative_flux_error=0.2,use_informed_flux=False,use_real_error_flux=False,do_model_of=None,**kwargs):
    #Check for existing results
    #path_model_result_csv = os.path.join(set_up_dir(system_name,path_for_models,dir_models),"final_result",f"models_{system_name}.csv")
    system_name = system.name.values[0]
    modeling_path = set_up_dir(path_for_models,dir_models,system_name)
    path_save = os.path.join(modeling_path,"model_result.pickle")
    lensmodel_system = lensmodel_handler(modeling_path,system)
    center_mass_error = 0.003
    astrometry_error = 0.003 
    can_end = False
    #quick solution
    #available_lens_models = lensmodel_system.mass_models.keys()
    if len(lensmodel_system.images) == 2:
        available_lens_models_to_model = ['SIS', 'SIE','SIS+shear']
    if len(lensmodel_system.images) > 2:
        available_lens_models_to_model = ['SIS+shear','SIS+shear','SIE','SIE+shear',"POW",'POW+shear']
    if system.total_lens.values[0]>1:
        available_lens_models_to_model = np.array([[model if r==1 else model+f"-{r}G" for r in range(1,system.total_lens.values[0]+1)] for model in available_lens_models_to_model]).ravel()
    if system.z_l.astype(float).values[0] > system.z_s.astype(float).values[0]:
            return print(f"Check redshift zl ({system.z_l.values[0]}) cant be bigger than zs ({system.z_s.values[0]})")
    if os.path.exists(path_save):
        #print(path_save)
        dic_lens_system=read_pickle(path_save)
        System_Class =  Result_Handler(dic_lens_system,max_separation=max_separation)#.get_best_model(,print_best=print_best)
        if System_Class.can_end and not model_All and not do_mcmc:
            print(f"The code found a best model with max(sep)< {max_separation} it is {System_Class.current_best_n_model}")
            return System_Class,modeling_path
        elif System_Class.can_end and do_mcmc:
            #lens_system_results = result_handler(dic_lens_system,max_separation=max_separation,look_for_best_model=True)
            #print(f"We will perform the mcmc using {lens_system_results.best_model_name} and it converge ={can_end}")
            print(f"The code found a best model with max(sep)< {max_separation} it is {System_Class.current_best_n_model}")
            System_Class = make_mcmc_lensmodel(System_Class,System_Class.current_best_n_model,modeling_path,rewrite=False)
            return System_Class,modeling_path
    available_lens_models = do_model_of or available_lens_models_to_model
    print("available_lens_models:",available_lens_models)
    for lap in range(10):
        if can_end and not model_All:
                    break
        for _,mass_model in enumerate(available_lens_models):
            if can_end and not model_All:
                    break
            alredy_model = False
            model_setup=lensmodel_system.writter_model_data(mass_distribution=mass_model,center_mass_error=center_mass_error,\
                                                            relative_flux_error=relative_flux_error,astrometry_error=astrometry_error,\
                                                                use_informed_flux=use_informed_flux,use_real_error_flux=use_real_error_flux)
            for key, value in kwargs.items():
                if key=="print_save" and value:
                    print(path_save)
            if os.path.exists(path_save):
                dic_lens_system=read_pickle(path_save)
                System_Class =  Result_Handler(dic_lens_system,max_separation=max_separation)
                if System_Class.can_end and not model_All:
                    break
                for key,items in dic_lens_system.items():
                    if compare_dicts(items["model_setup"]["model_setup"],model_setup):
                        alredy_model = True
                        print(f"lap {lap} Modeled {system_name} using {mass_model} center_mass_error {center_mass_error} relative_flux_error {relative_flux_error} astrometry_error {astrometry_error}")
            if not alredy_model:
                print(f"lap {lap} Modeling {system_name} using {mass_model} center_mass_error {center_mass_error} relative_flux_error {relative_flux_error} astrometry_error {astrometry_error}")
                run_lensmodel(modeling_path,model_setup['name_run'])
                current_model_result=full_modeling_result(modeling_path,model_setup) #local solution or dictionary with all the results from a model 
                if not os.path.exists(path_save) and "final_step" in current_model_result.keys():
                    write_pickle({"model_0":current_model_result},os.path.join(modeling_path,"model_result.pickle"))
                    dic_lens_system=read_pickle(path_save) #just it is done to be sure we are cheking well
                elif "final_step" in current_model_result.keys():
                    dic_lens_system=read_pickle(path_save)
                    n = max([int(i.replace("model_","")) for i in dic_lens_system.keys()])+1
                    dic_lens_system[f"model_{n}"] = current_model_result
                    write_pickle(dic_lens_system,os.path.join(modeling_path,"model_result.pickle"))
            System_Class =  Result_Handler(dic_lens_system,max_separation=max_separation)
            if System_Class.can_end and not model_All:
                print(f"The code found a best model with max(sep)< {max_separation} it is {System_Class.current_best_n_model}")
                break
        
        can_end = System_Class.can_end 
        
        if lap >=  max_lap:
            break
                #return dic_lens_system,modeling_path
        if lap==0:
            center_mass_error = 0.01
        if lap==1:
            center_mass_error = 0.01
            astrometry_error = 0.01 
        if lap==2:
            relative_flux_error = 0.5
            center_mass_error = 0.01 
            astrometry_error = 0.003
        if lap==3:
            relative_flux_error = 0.5
            center_mass_error = 0.01 
            astrometry_error = 0.01
        if lap == 4:
            print("Looks like any model works")
            break
    if not do_mcmc:
        return System_Class,modeling_path
    else:
        System_Class = make_mcmc_lensmodel(System_Class,System_Class.current_best_n_model,modeling_path,rewrite=False)
    return System_Class,modeling_path  