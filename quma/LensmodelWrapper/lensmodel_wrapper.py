import os
import stat
import subprocess

module_dir = os.path.dirname(os.path.abspath(__file__))

def run_lensmodel(modeling_path,run_name):
    try:
        path_to_run = os.path.join(modeling_path,f"{run_name}.dat")
        where_we_are = os.getcwd()
        path_to_lensmodel = f"{module_dir}/lensmodel/lensmodel"
        #print(path_to_lensmodel)
        st = os.stat(path_to_lensmodel)
        # Get permissions and make the file executable
        os.chmod(path_to_lensmodel, st.st_mode | stat.S_IEXEC)
        # Change the current working directory to work_pad
        os.chdir(os.path.join(module_dir,f"{module_dir}/lensmodel"))
        with open(path_to_run, 'r') as f:
            proc = subprocess.Popen([path_to_lensmodel], stdin = f, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            proc.wait()
        os.chdir(where_we_are)
    except KeyboardInterrupt:
        os.chdir(where_we_are)
    #just remove the files in lensmodel
    [os.remove(os.path.join(f"{module_dir}/lensmodel",i) )for i in os.listdir(f"{module_dir}/lensmodel") if "lensmodel" not in i]
        #print("Execution stopped by the user.")
