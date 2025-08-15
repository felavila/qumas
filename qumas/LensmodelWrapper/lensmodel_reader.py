
### all of this should be here 
def make_dictionary(list,prefix="p"):
    return {f"{prefix}[{i}]":float(list[i]) for i in range(len(list))}

def get_numeric_values(text):
    numeric_values=re.findall(r'[-+]?\d*\.\d+|\d+|\(?\s*[-+]?\d*\.?\d*e[+-]?\d*\s*\)?', text)
    return numeric_values

def get_images_result(list_lensmodel):
    images_result = {"ra_imput":[],"dec_imput":[],"radec_error":[],"flux_imput":[],"flux_error":[],"time_delay_imput":[],"time_delay_error":[],"ra_output":[],"dec_output":[],"flux_output":[],"time_delay_output":[]}
    for j,imagen in enumerate(list_lensmodel):
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
    result = {}
    for n,line in enumerate(list_of_lines):
        if "LENS PARMS" in line:
            #what happend if i have 2 galaxies ?
            result["LENS PARMS"] ={}
            for i in range(1,10):
                if "alpha" in list_of_lines[n+i]:
                    result["LENS PARMS"][f"alpha{i}"] = make_dictionary([i.replace("\n","") for i in list_of_lines[n+1].split(" ") if (len(i)>0 and i!='alpha')])
        if "SOURCE PARMS" in line:
            result["SOURCE PARMS"] ={}
            for i in range(1,10):
                if "ptsrc" in list_of_lines[n+i]:
                    result["SOURCE PARMS"][f"ptsrc{i}"] = make_dictionary([i.replace("\n","") for i in list_of_lines[n+1].split(" ") if (len(i)>0 and i!='ptsrc')],prefix="s")
        if "CHISQ" in line:
            chi_values,chi_columns = list_of_lines[n].split("#")
            result["CHISQ"]= {v: float(k) for k, v in zip(chi_values.replace("CHISQ:","").split(" ")[1:-1],chi_columns.replace("\n","").split(" ")[1:])}
        if "Source" in line:
            #should be updated for cases with more source
            result["Source"] = {i[0].replace(" ",""):i[1] for i in [list_of_lines[n+i].replace("\n","").split(":") for i in range(1,6)]}
        if "images" in line:
            result["images"] = get_images_result(list_of_lines[n:][1:-2])
    return result

def get_critical_caustic(list_of_lines):
    return {"critical_caustic":{key:np.array([[i for i in i.split(" ")[1:] if len(i)>0] for i in [i.replace("\n","") for i in list_of_lines][1:] if len(i)>0],dtype="object").astype(float)[::,n] for n,key  in enumerate(["x","y","u","v"])}}

def get_grid(list_of_lines):
    return {"grid":{key:np.array([[i for i in i.split(" ") if len(i)>0] for i in [i.replace("\n","") for i in list_of_lines][1:] if len(i)>0],dtype="object").astype(float)[::,n] for n,key  in enumerate(["x","y","u","v"])}}

def get_RE(list_of_lines):
   re = {'RE':float(list_of_lines[0].split(" ")[0])}
   print(re)
   return re

def run_lensmodel(modeling_path,run_name):
    #run_name = "model_run"
    try:
        path_to_run = os.path.join(modeling_path,f"{run_name}.dat")
        where_we_are = os.getcwd()
        path_to_lensmodel = os.path.join(os.getcwd(),"time_scales/lensmodel/lensmodel")
        st = os.stat(path_to_lensmodel)
        # Get permissions and make the file executable
        os.chmod(path_to_lensmodel, st.st_mode | stat.S_IEXEC)
        # Change the current working directory to work_pad
        os.chdir(os.path.join(where_we_are,"time_scales/lensmodel"))
        with open(path_to_run, 'r') as f:
            proc = subprocess.Popen([path_to_lensmodel], stdin = f, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            proc.wait()
        os.chdir(where_we_are)
    except KeyboardInterrupt:
        os.chdir(where_we_are)
        #print("Execution stopped by the user.")







def full_modeling_result(modeling_path,model_setup):
    model_result = {}
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
        [os.remove(os.path.join(modeling_path,i) )for i in os.listdir(modeling_path) if step in i]
    model_result["model_setup"] = {"model_setup":model_setup}
    return model_result

def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as f:
        _save_dict_to_hdf5_recursive(f, '/', dic)

def _save_dict_to_hdf5_recursive(f, path, dic):
    for key, item in dic.items():
        if isinstance(item, dict):
            _save_dict_to_hdf5_recursive(f, path + key + '/', item)
        else:
            f[path + key] = item

def read_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        data = _read_hdf5_recursive(f, '/')
    return data

def _read_hdf5_recursive(f, path):
    data = {}
    for key, item in f[path].items():
        if isinstance(item, h5py.Group):
            data[key] = _read_hdf5_recursive(f, path + key + '/')
        else:
            # Decode byte strings to regular strings
            if isinstance(item, h5py.Dataset) and isinstance(item[()], bytes):
                data[key] = item[()].decode('utf-8')  # Specify the appropriate encoding
            elif isinstance(item, h5py.Dataset) and isinstance(item[()], np.ndarray):
                data[key] = [x if type(x)==np.float64 else x.decode('utf-8')  for x in item[()] ] # Specify the appropriate encoding
            else:
                data[key] = item[()]
    return data
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