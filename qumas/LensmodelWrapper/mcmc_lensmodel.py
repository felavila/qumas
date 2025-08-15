import multiprocessing
import pandas as pd
import numpy as np
from .lensmodel_wrapper import run_lensmodel
from .utils import get_RE,write_pickle,get_kappa_gamma,get_kappa_gamma#,get_critical_caustic,get_grid,clean_number
import os 

def make_pandas_from_mcmc(path):
    # Initialize lists to hold data
    f = open(f"{path}/mcmc.dat","r")
    file_content=f.readlines()
    lens_parms = []
    source_parms = []
    chisq_values = []

    # Iterate over the lines and extract data
    for i, line in enumerate(file_content):
        if "LENS PARMS" in line:
            lens_values = list(map(str, file_content[i+1].strip().split()[0:]))
            lens_parms.append(lens_values)
        elif "SOURCE PARMS" in line:
            source_values = list(map(str, file_content[i+1].strip().split()[0:]))
            source_parms.append(source_values)
        elif "CHISQ" in line:
            chisq_value = float(line.split()[1])
            chisq_values.append(chisq_value)

    # Combine into a DataFrame
    data = {
        'Lens_mass': [params[0] for params in lens_parms],
        'p[1]': [params[1] for params in lens_parms],
        'p[2]': [params[2] for params in lens_parms],
        'p[3]': [params[3] for params in lens_parms],
        'p[4]': [params[4] for params in lens_parms],
        'p[5]': [params[5] for params in lens_parms],
        'p[6]': [params[6] for params in lens_parms],
        'p[7]': [params[7] for params in lens_parms],
        'p[8]': [params[8] for params in lens_parms],
        'p[9]': [params[9] for params in lens_parms],
        'p[10]': [params[10] for params in lens_parms],
        'Source': [params[0] for params in source_parms],
        'Source_Param1': [params[1] for params in source_parms],
        'Source_Param2': [params[2] for params in source_parms],
        'Source_Param3': [params[3] for params in source_parms],
        'Source_Param4': [params[4] for params in source_parms],
        'Source_Param5': [params[5] for params in source_parms],
        'Source_Param6': [params[6] for params in source_parms],
        'Source_Param7': [params[6] for params in source_parms],
        'CHISQ': chisq_values
    }

    df = pd.DataFrame(data)
    return df
def modify_alpha_row(path,data, new_values):
    """
    Modify the values in the 'alpha' row of the data list.

    Parameters:
    data (list): The list containing the rows of data as strings.
    new_values (list): A list of new values to replace the current values in the 'alpha' row.

    Returns:
    list: The modified list with updated 'alpha' row values.
    """
    # Find the row containing the 'alpha' keyword
    alpha_row_index = next((i for i, row in enumerate(data) if 'alpha' in row), None)

    if alpha_row_index is not None:
        # Split the alpha row into components
        components = data[alpha_row_index].split()

        # Replace the numerical values with new values
        # Assuming the 'alpha' keyword is the first component
        components[1:len(new_values)+1] = new_values

        # Recombine the components into a single string
        data[alpha_row_index] = ' '.join(components) + '\n'
    with open(f'{path}/start_re_kg.dat', 'w') as file:
        file.writelines(data)

def make_mcmc_lensmodel(system_class,model_to_mcmc,path,rewrite=False):
    model,system_info = system_class.get_necessary_to_mcmc(model_to_mcmc)
    if not rewrite:
        if "mcmc" in model.keys():
            print(f"MCMC alredy done for the model {model_to_mcmc} and rewritte false")
            return system_class
    system_file,start_mcmc = model["model_setup_file"],model["best_model_file"]
    z_l,z_s,name = system_info["zl"],system_info["zs"],system_info["name"]
    print("Starting MCMC")
    with open(f'{path}/{name}.dat', 'w') as file:
        file.writelines(system_file)
    with open(f'{path}/start_mcmc.dat', 'w') as file:
        file.writelines(start_mcmc)
    with open(f'{path}/domcmc.dat', 'w') as file:
        mcmc = [    "#MCMC\n",
                    "set omitcore = 0.05\n",
                    f"data {path}/{name}.dat #filewhitdata\n",
                    "set omega = 0.3\n",
                    "set lambda = 0.7\n",
                    "set hval = 0.7\n",
                    "set hvale = 1000 # Uncertainty in H_0 in units of 100 km/s/Mpc\n",
                    "set chimode = 0 #sourceplane\n",
                    f"set zlens = {z_l} # lens redshift\n",
                    f"fset zsrc = {z_s} # source redshift\n",
                    "set checkparity = 0 # don't worry about parities\n",
                    "set gridflag = 0 # don't need the tiling\n",
                    "#set restart = 2\n",
                    f"setlens {path}/start_mcmc.dat\n",
                    "MCMCset 2 Nchain 10 maxsteps 100000\n",
                    f"MCMCrun {path}/mcmc\n","quit\n"]
        file.writelines(mcmc)
    run_lensmodel(path,"domcmc")
    result_pandas=make_pandas_from_mcmc(path)
    with open(f'{path}/do_re_kg.dat', 'w') as file:
        main = [
                    "set omitcore = 0.05\n",
                    f"data {path}/{name}.dat #filewhitdata\n",
                    "set omega = 0.3\n",
                    "set lambda = 0.7\n",
                    "set hval = 0.7\n",
                    "set hvale = 1000 # Uncertainty in H_0 in units of 100 km/s/Mpc\n",
                    "set chimode = 0 #sourceplane\n",
                    f"set zlens = {z_l} # lens redshift\n",
                    f"fset zsrc = {z_s} # source redshift\n",
                    "set checkparity = 0 # don't worry about parities\n",
                    "set gridflag = 0 # don't need the tiling\n",
                    "#set restart = 2\n",
                    f"setlens {path}/start_re_kg.dat\n",
                    f"calcRein 10 {path}/RE.dat\n",
                    f"kapgam 3 {path}/kappa_gamma.dat\n",
    "quit\n"]
        file.writelines(main)
    RE = []
    data_list = []
    for new_values in result_pandas[[i for i in result_pandas.columns.values if "p[" in i]].values:
        #print(new_values)
        modify_alpha_row(path,start_mcmc, new_values)
        run_lensmodel(path,"do_re_kg")
        RE.append( get_RE(open(os.path.join(path,"RE.dat"), "r").readlines())["RE"])
        data_list.append(get_kappa_gamma(open(os.path.join(path,"kappa_gamma.dat"), "r").readlines())["kappa_gamma"])
    result_pandas["RE"] = RE 
    model.update({"mcmc":{"system_info":system_info,"mcmc_chain":result_pandas,"kappa_gamma_chain": pd.concat([pd.DataFrame(data) for data in data_list], ignore_index=True)}})
    system_class.lensmodel_system[model_to_mcmc] = model
    write_pickle(system_class.lensmodel_system,f"{path}/model_result.pickle")
    [os.remove(os.path.join(path,i) )for i in os.listdir(path) if "model_result.pickle" not in i]
    print("MCMC ENDED")
    return system_class

# def make_mcmc_lensmodel_file(path,model_to_do_mcmc,system,max_separation,converge,rewrite=False):
#     """given the format of my code i prefer use list that contains the init and start file the start file is the best values file"""
#     if rewrite==False and os.path.isfile(f"{path}/mcmc.pickle"):
#         print("MCMC alredy done")
#         return
    
#     name = path.split("/")[-1]
#     model_name = model_to_do_mcmc["model_name"]
#     system_file,start_mcmc = model_to_do_mcmc["model_setup_file"],model_to_do_mcmc["best_model_file"]
#     z_l,z_s = float(clean_number(system.z_l.values[0])),float(clean_number(system.z_s.values[0]))
#     print("Starting mcmc")
#     with open(f'{path}/{name}.dat', 'w') as file:
#         file.writelines(system_file)
#     with open(f'{path}/start_mcmc.dat', 'w') as file:
#         file.writelines(start_mcmc)
#     with open(f'{path}/domcmc.dat', 'w') as file:
#         mcmc = [
#                     "#MCMC\n",
#                     "set omitcore = 0.05\n",
#                     f"data {path}/{name}.dat #filewhitdata\n",
#                     "set omega = 0.3\n",
#                     "set lambda = 0.7\n",
#                     "set hval = 0.7\n",
#                     "set hvale = 1000 # Uncertainty in H_0 in units of 100 km/s/Mpc\n",
#                     "set chimode = 0 #sourceplane\n",
#                     f"set zlens = {z_l} # lens redshift\n",
#                     f"fset zsrc = {z_s} # source redshift\n",
#                     "set checkparity = 0 # don't worry about parities\n",
#                     "set gridflag = 0 # don't need the tiling\n",
#                     "#set restart = 2\n",
#                     f"setlens {path}/start_mcmc.dat\n",
#                     "MCMCset 2 Nchain 10 maxsteps 100000\n",
#                     f"MCMCrun {path}/mcmc\n",
                    
#     "quit\n"
# ]
    
#         file.writelines(mcmc)
    
#     run_lensmodel(path,"domcmc")
#     result_pandas=make_pandas_from_mcmc(path)
    
#     with open(f'{path}/do_re_kg.dat', 'w') as file:
#         main = [
#                     "set omitcore = 0.05\n",
#                     f"data {path}/{name}.dat #filewhitdata\n",
#                     "set omega = 0.3\n",
#                     "set lambda = 0.7\n",
#                     "set hval = 0.7\n",
#                     "set hvale = 1000 # Uncertainty in H_0 in units of 100 km/s/Mpc\n",
#                     "set chimode = 0 #sourceplane\n",
#                     f"set zlens = {z_l} # lens redshift\n",
#                     f"fset zsrc = {z_s} # source redshift\n",
#                     "set checkparity = 0 # don't worry about parities\n",
#                     "set gridflag = 0 # don't need the tiling\n",
#                     "#set restart = 2\n",
#                     f"setlens {path}/start_re_kg.dat\n",
#                     f"calcRein 10 {path}/RE.dat\n",
#                     f"kapgam 3 {path}/kappa_gamma.dat\n",
#     "quit\n"]
#         file.writelines(main)
#     RE = []
#     data_list = []
#     for new_values in result_pandas[[i for i in result_pandas.columns.values if "p[" in i]].values:
#         #print(new_values)
#         modify_alpha_row(path,start_mcmc, new_values)
#         run_lensmodel(path,"do_re_kg")
#         RE.append( get_RE(open(os.path.join(path,"RE.dat"), "r").readlines())["RE"])
#         data_list.append(get_kappa_gamma(open(os.path.join(path,"kappa_gamma.dat"), "r").readlines())["kappa_gamma"])
#     result_pandas["RE"] = RE 
#     model_result = {"mcmc_chain":result_pandas,"system":system,"kappa_gamma_chain": pd.concat([pd.DataFrame(data) for data in data_list], ignore_index=True)}
#     model_result.update(model_to_do_mcmc)
#     #model_result[i for i in model_to_do_mcmc.keys()] = {i:model_to_do_mcmc[i] for i in model_to_do_mcmc.keys()}
#     model_result["max_separation"] = max_separation
#     model_result["converge"] = converge
#     for i in os.listdir(path):
#         #i can remove this some day
#         if i =="final_result":
#             continue
#         if i.endswith(('.pickle', '.hdf5')):
#             continue
#         os.remove(os.path.join(path,i))
#     main_result = {}
#     main_result[model_name+"_"+str(max_separation)+"_"+str(converge)] = model_result
#     write_pickle(main_result,f"{path}/mcmc.pickle")
#     print(f"{path}/mcmc.pickle")
    
    
    