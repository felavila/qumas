import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import itertools

import stat
import shutil
from datetime import datetime
from ast import literal_eval


from astropy.cosmology import FlatLambdaCDM
from  lenstronomy.Util import param_util
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
Cosmo=FlatLambdaCDM(H0=67.36,Om0=0.264+0.051, Ob0=0.051,Neff=3.04)#lcdm no confiable del todo
Cpath=os.getcwd()+"/" 

def models_disp(imagenes,dif):
    #all use SIE
    Models=[]
    if len(imagenes)==1:
        Models=["SIS"]
    if len(imagenes)==2:
        Models=["SIS","SIE","SIS+shear"]
    if len(imagenes)==3:
        Models=["SIS","SIE","SIS+shear"]
    if len(imagenes)==4:
        Models=["SIE","SIE+shear"]
        if len(dif)>1:
            Models.append("SIE-2G")
    if len(imagenes)>=5 and len(dif)>1:
        Models.append("SIE-2G")
    return Models
def copy_file(pad):
    shutil.copyfile(pad+"/critT.dat",pad+"/result/critT.dat")
    shutil.copyfile(pad+"/gridT.dat",pad+"/result/gridT.dat")
    shutil.copyfile(pad+"/third.start",pad+"/result/third.start")

def canbemodeled(p):
    nombre_sistema = p.name.drop_duplicates().values[0]
    band=chooseband(p,print_bandas=False)
    imagenes=p[p['type']=="ima"][["RA","DEC","dDEC","dRA"]].astype(float) #informacion de las imagenes
    dif=p[p['type']=="dif"][[i for i in p[p['type']=="dif"].columns if i not in ["RA","DEC","dDEC","dRA"]]]
    Problem=[]
    tieneproblemas=False
    if len(imagenes)==0:
        problem="ceroimage"
        Problem.append(problem)
    if len(imagenes)==1:
        problem="ring"
        Problem.append(problem)
        print([nombre_sistema,problem])
    if len(imagenes)>5:
        problem="mas5i"
        Problem.append(problem)
        print([nombre_sistema,"mas5i"])
    if len(imagenes)==3:
        problem="triple"
        Problem.append(problem)
        print([nombre_sistema,"triple"])
    if band=="Noneband":
        problem="Noband"
        Problem.append(problem)
        #print([nombre_sistema,"Noband"])
    if len(dif)==0:
        problem="NoG"
        Problem.append(problem)
        #print([nombre_sistema,problem])
    if len(Problem)>0:
           tieneproblemas=True
    return tieneproblemas,np.array([nombre_sistema,Problem],dtype=object)

def sacador(i,L): #i==lista L array con los segmentos a sacar 
    sa=np.array([])
    for II in range(len(L)):
        if II==0:
            continue
        sa=np.append(sa,(i[int(L[II-1]):int(L[II])]).replace(" ", ""))
    return sa
            #print(list_of_lines[I][int(L[II-1]):int(L[II])])
def db(x,y): 
    #x:componente x del vector
    #y:componente y del vector
    D=np.array([])
    for i in range(len(x)):
        for ii in range(len(x)):
            d=np.sqrt((x[i]-x[ii])**2+(y[i]-y[ii])**2)
            if d not in D:
                D=np.append(D,d)
    return D

def correction(magnification,no_corregido): #correccion
    """magnification:magnificacion de la imagen
        no_corregido:flujo del sistema sin corregir"""
    #A=10**(-no_corregido/2.5) 
    #B=abs(magnification)
    return -2.5*np.log10((10**(-no_corregido/2.5))/abs(magnification))

def chooseband(B,print_bandas=True):
    try:
        B=B.loc[B.index.get_level_values('type')=="ima"]
    except:
        B=B.loc[B['type']=="ima"]
    Flujos=['g', 'r',"V","R","I","H",'i','z','Y','F160W','F814W','F475X',"G","5GHz","B","Kp"]
    F=[]
    B=B.fillna("-")
    for i in Flujos:
        if "-" in B[i].values:
            continue
        else:
            F.append(i)
    if print_bandas:
        print(f"Bandas disponibles {F}")
    if len(F)==0:
        f="Noneband"
        #print("Banda a utilizar "+f)
    if len(F)>0:
        if "F814W" in F:
            f="F814W"
            #print("Banda a utilizar "+f)
        elif 'i' in F:
            f='i'
        elif 'I' in F:
            f='I'
            #printabla_qso_lensingB.csvt("Banda a utilizar "+f)
        else:
            f=F[0]
            #print("Banda a utilizar "+f)
    return f
    
def getcurves(pad,crit="critT.dat",grid="gridT.dat"):
    files=os.listdir(pad)
    if crit in files and grid in files:
        files=[crit,grid]
        CG=[]
        for i in files:
            P=np.empty((0,4),float)#array vacia
            f=open(pad+"/"+i,"r")
            lines=f.readlines()
            for L in lines:
                if "Columns" in L or len(L)==1:
                    continue
                else:
                    p_critic=[]
                    for l in range(len(L)):
                        if " "==L[l] and " " !=L[l-1]:
                            p_critic.append(int(l))
                if len(p_critic)<4:
                    p_critic=[0]+p_critic
                p_critic=p_critic+[len(L)]
                p=np.array([float(L[p_critic[o]:p_critic[o+1]]) for o in range(len(p_critic)-1)])
                P=np.vstack([P,p])
            CG.append(P)
        curvas={"xc":CG[0][::,0],"yc":CG[0][::,1],"uc":CG[0][::,2],"vc":CG[0][::,3],
                "xg":CG[1][::,0],"yg":CG[1][::,1],"ug":CG[1][::,2],"vg":CG[1][::,3]}
    else:
        curvas="SAD"
        print("No estan los archivos")
    return curvas

def lensmodel(pad,nombre_run,work_pad=Cpath):
    """pad:directorio donde se quiere operar
    nombre de la run a aplicar"""
    x=pad+"/"+nombre_run
    preprocess_path =work_pad+"lensmodel"
    #print(preprocess_path)
    st = os.stat(preprocess_path)
    os.chmod(preprocess_path, st.st_mode | stat.S_IEXEC)   
    with open(x, 'r') as f:
        proc = subprocess.Popen([preprocess_path], stdin = f, stdout=subprocess.DEVNULL,
  stderr=subprocess.DEVNULL)
        proc.wait()
    for i in ['crit.dat','grid.dat','chitmp.dat','best.sm',"chitmp.dat"]:
        try:
            os.remove(os.path.join(Cpath,i))
            #print(f'The file {file_path} has been successfully removed.')
        except OSError as e:
            pass
    return

def getchis(imagenes,pad,archivo):
    """imagenes:solo para obtener el numero de imagenes del sistema
    pad:directorio donde se buscara el archivo
    archivo:nombre del archivo a utilizar
    """

    n_i=len(imagenes) # numero de imagenes 
    factorcorrection=1.143e-9
    completName = os.path.join(pad,archivo)
    file = open(completName, "r")
    list_of_lines = file.readlines()
    chix=np.array([])
    for I in range(len(list_of_lines)):
        L=np.array([])
        if 'CHISQ:'in list_of_lines[I]:
            for ii in range(len(list_of_lines[I])):
                if " "==list_of_lines[I][ii] and " "!=list_of_lines[I][ii-1]:
                    chix=np.append(chix,ii)
            chi_t=float(list_of_lines[I][int(chix[0]):int(chix[1])])
            chi_p=float(list_of_lines[I][int(chix[1]):int(chix[2])])
            chi_f=float(list_of_lines[I][int(chix[2]):int(chix[3])])
    chi={"chi_t":chi_t,"chi_p":chi_p,"chi_f":chi_f}
    return chi

def getresultados(i,pad,archivo): #nombre de una galaxia dentro de la lista de datos 
    """"i:nombre sistema
        pad:directorio
        archivo que contiene la informacion
     """
    completName = os.path.join(pad,archivo)
    file = open(completName, "r")
    list_of_lines = file.readlines()
    A=np.empty((0,10),float)#array vacia
    for I in range(len(list_of_lines)):
        L=np.array([])
        if 'alpha'in list_of_lines[I]:
            for ii, txt in enumerate(list_of_lines[I]):
                if txt==" " and list_of_lines[I][ii-1] !=" ":
                    L=np.append(L,ii)
            L=np.append(L,len(list_of_lines[I])-1)
            a=sacador(list_of_lines[I],L)[1:]
            A=np.vstack([A,a])
        if "posn:" in list_of_lines[I]:
            for ii, txt in enumerate(list_of_lines[I]):
                if txt==" " and list_of_lines[I][ii-1] !=" ":
                    L=np.append(L,ii)
            L=np.append(L,len(list_of_lines[I])-1)
            S=sacador(list_of_lines[I],L)[1:3]
    A=A.astype(float)   
    Datos={"s_x":float(S[0]),
           "s_y":float(S[1]),"p[1]":A[::,0],"p[2]":A[::,1],"p[3]":A[::,2],
           "p[4]":A[::,3],"p[5]":A[::,4],"p[6]":A[::,5],"p[7]":A[::,6],"p[8]":A[::,7],"p[9]":A[::,8],"p[10]":A[::,9]}
    return Datos
    #corrected=correction(1.143e-9,magnification,no_corregidos)

def getkg(pad,imagenes,band):
    for f in os.listdir(pad):
        if "kappa" in f:
            pad2=f
                #K.append(f)
        #k=dict.fromkeys([*K])
    f=open(pad+"/"+pad2,"r")
    lines=f.readlines()
    S=np.empty([0,6])
    for x in lines:
        if "output" in x or "source" in x:
            continue
        else:
            critic=[]
            for I in range(len(x)):
                if " "==x[I] and " "!=x[I-1]: 
                    critic.append(int(I))
                    if x[I+1]=="n":
                        print("hola")
            critic.append(len(x))
        L=[float(x[critic[i]:critic[i+1]]) for i in range(len(critic)-1)][1:6]
        L.append(1/((1-L[2])**2-L[3]**2))
        S=np.vstack((S,L))
    #correction(magnification,no_corregido)
    keys=["RA","DEC","kappa","gamma","theta","magnification"]
    k=dict.fromkeys(keys)
    for i,txt in enumerate(keys):
        k[txt]=S[::,i]
    dk=pd.DataFrame(k)
    dk["component"]=imagenes["component"].values
    dk["corrected"]=correction(dk["magnification"].values,imagenes[band].values)
    dk=dk.reindex(["component",*keys,"corrected"], axis=1)
    return dk

def dchi(nombre_sistema,pad,archivo,imagenes,dif,model,error_cm,error_flux,kapagamma=True,band="NOINFO",Psystem="NOINFO"): #nombre de una galaxia dentro de la lista de datos 
    """nombre_sistema:nombre galaxia
    pad: direccion en donde esta el archivo a extraer
    archivo:nombre del archivo
    imagenes: para obtener las componentes de cada imagen
    r_f: razon de flujo utilizada en el calculo
    e_d: error utilizado en el defractor
    er_f: error utilizado en la razon de flujos
    l:en esta version corresponde a el run a utlizar"""
    
    result=getresultados(nombre_sistema,pad,archivo)
    dis=np.empty([0,9])
    ################sacar_data_third.dat###############
    completName = os.path.join(pad,archivo)
    file = open(completName, "r")
    list_of_lines = file.readlines()
    nombres_imagenes=imagenes["component"].values
    n_i=len(nombres_imagenes)
    chi_t,chi_pos,chi_f=chi_t,chi_pos,chi_f=np.transpose(np.repeat(np.array(list(getchis(imagenes,pad,archivo).values()))[None,:],n_i, axis=0))
    fecha=str(datetime.now().year)+str(datetime.now().month)+str(datetime.now().day)+str(datetime.now().hour)+str(datetime.now().minute)
    for I in range(len(list_of_lines)):
        L=np.array([])
        if 'alpha'in list_of_lines[I]:
            for ii, txt in enumerate(list_of_lines[I]):
                if txt==" " and list_of_lines[I][ii-1] !=" ":
                    L=np.append(L,ii)
            L=np.append(L,len(list_of_lines[I])-1)
            A=sacador(list_of_lines[I],L)[1:]
        if "posn:" in list_of_lines[I]:
            for ii, txt in enumerate(list_of_lines[I]):
                if txt==" " and list_of_lines[I][ii-1] !=" ":
                    L=np.append(L,ii)
            L=np.append(L,len(list_of_lines[I])-1)
            S=sacador(list_of_lines[I],L)[1:3] #posicion de la fuente
        if ">" in list_of_lines[I]:
            for ii, txt in enumerate(list_of_lines[I]):
                if txt==" " and list_of_lines[I][ii-1] !=" ":
                    L=np.append(L,ii)
            L=np.append(L,len(list_of_lines[I])-1)
            X=sacador(list_of_lines[I],L)
            dis=np.vstack((dis,np.array([X[0].replace(" ", ""),X[1].replace(" ", ""),X[3],X[6],X[9],X[10],X[11],X[12],X[2].replace("(", "").replace(")", "")])))##reemplazar algo en un str
    
    A=A.astype(float)
    dis=dis.astype(float)
    
    d_im={"name":np.repeat(nombre_sistema,n_i),'image':nombres_imagenes,"RA":dis[:,0],'~RA':dis[:,4],
    "deltaRA":dis[:,0]-dis[:,4],"e_RA":dis[:,8],
    "DEC":dis[:,1],'~DEC':dis[:,5],"deltaDEC":dis[:,1]-dis[:,5],
    "e_DEC":dis[:,8],'F':np.abs(dis[:,2]),'~F':np.abs(dis[:,6]),"deltaF":dis[:,2]-dis[:,6],
    "error_flux":[error_flux for i in range(n_i)],"error_cm":[error_cm for i in range(n_i)],"chis_t":chi_t,"chi_f":chi_f,"chi_p":chi_pos,
    "fecha":np.repeat(fecha,n_i),"model":np.repeat(model,n_i),"banda":np.repeat(band,n_i),"Psystem":np.repeat(Psystem,n_i)}
    for i in result:
        if "s" not in i :#and len(result[i])>1
            d_im[i]=[[result[i][XD] for XD in range(len(result[i]))] for xd in range(n_i)]
        else:
            d_im[i]=np.repeat(result[i],n_i) 
    ###faltaria definir para en caso de tener mas de un G########
    for f in os.listdir(pad):#listdir(pad) para hacer lista de los objetos en un directorio
        if "kappa" in f:
            pad2=f
                #K.append(f)
        #k=dict.fromkeys([*K])
    f=open(pad+"/"+pad2,"r")
    lines=f.readlines()
    S=np.empty([0,6])
    for x in lines:
        if "output" in x or "source" in x:
            continue
        else:
            critic=[]
            for I in range(len(x)):
                if " "==x[I] and " "!=x[I-1]: 
                    critic.append(int(I))
                    if x[I+1]=="n":
                        print("hola")
            critic.append(len(x))
        L=[float(x[critic[i]:critic[i+1]]) for i in range(len(critic)-1)][1:6]
        L.append(1/((1-L[2])**2-L[3]**2))
        S=np.vstack((S,L))
    #correction(magnification,no_corregido)
    keys=["RA","DEC","kappa","gamma","theta","magnification"]
    k=dict.fromkeys(keys)
    for i,txt in enumerate(keys):
        k[txt]=S[::,i]
    dk=pd.DataFrame(k)
    
    dk["image"]=imagenes["component"].values
    dk["corrected"]=correction(dk["magnification"].values,imagenes[band].values)
    dk=dk.reindex(["image",*keys,"corrected"], axis=1)
    dk=dk.drop(['RA', 'DEC'], axis=1)
    #d_dif={"name":nombre_sistema,'image':"G","RA":0,'~RA':A[1],"deltaRA":0-A[1],"e_RA":0,
    #"DEC":0,'~DEC':A[2],"deltaDEC":0-A[2],"e_DEC":0,
    #'F':np.nan,'~F':np.nan,"deltaF":0,"e_F":0,"chis_t":chi_t[0],"chi_f":chi_f[0],"chi_p":chi_pos[0],
    #"fecha":fecha,"L":l}
    d=pd.DataFrame(d_im)
    Distancias=[np.sqrt(np.sum(np.power(i,2))) for i in d[["deltaRA","deltaDEC"]].values]
    dk["d_max"] = np.repeat(np.max(Distancias),len(d))
    C=pd.merge(d,dk, on ='image')
    #D = d.append(d_dif, ignore_index=True)
    dchi= pd.DataFrame(C)
    
    return dchi

def lenstronomy_obj(nombre_sistema,pad,archivo,imagenes,dif,pandas,zl,zs):
    if type(pandas["p[1]"].iloc[0])==str:
        d_r={i:pandas[i].iloc[[0]].values[0] for i in ["s_x","s_y"]}
        b={i:literal_eval(pandas[i].iloc[[0]].values[0]) for i in ["p[1]","p[2]","p[3]","p[4]","p[5]","p[6]","p[7]","p[8]","p[9]","p[10]"]}
        d_r.update(b)
    else:
        d_r={i:pandas[i].iloc[[0]].values[0] for i in ["s_x","s_y","p[1]","p[2]","p[3]","p[4]","p[5]","p[6]","p[7]","p[8]","p[9]","p[10]"]}
    #getresultados(nombre_sistema,pad,archivo)  
    kwargs_lens=[]
    lens_model_list=[]
    #print(d_r)
    for i in range(len(d_r["p[5]"])):
        model='PEMD'
        angulo=d_r["p[5]"][i]-90
        q=1-d_r["p[4]"][i]
        center_x,center_y=d_r["p[2]"][i],d_r["p[3]"][i]
        theta_E=np.sqrt((1+q**2)/(2*q))*d_r["p[1]"][i]#theta_E_gravlens
        e1,e2 = param_util.phi_q2_ellipticity((angulo)*np.pi/180.,q=q)
        #print(e1,e2,"exentricitis")
        kwargs_l={'theta_E': theta_E, 'e1': e1, 'e2': e2, 'center_x': center_x, 'center_y': center_y, 'gamma': 2.0}
        lens_model_list.append(model)
        kwargs_lens.append(kwargs_l)
        if len(imagenes)==4 and i==0 and d_r["p[7]"][i] !=0:
            model='SHEAR'
            angulo=d_r["p[7]"][i]+180
            gamma=d_r["p[6]"][i]
            gamma1,gamma2=param_util.shear_polar2cartesian(phi=(angulo)*np.pi/180., gamma=gamma)
            #e1,e2 = param_util.phi_q2_ellipticity((angulo)*np.pi/180.,q=q)
            kwargs_l = {'gamma1': gamma1, 'gamma2': gamma2}
            lens_model_list.append(model)
            kwargs_lens.append(kwargs_l)
    #lensModel=LensModel(lens_model_list=['PEMD'])
    source_x,source_y=d_r["s_x"],d_r["s_y"]
    # data imagen
    numPix = 500  #  cutout pixel size
    deltaPix = 0.05  #  pixel size in arcsec (area per pixel = deltaPix**2)
    ##################
    """print(source_x, source_y,"sx,sy modelo lente")
    print(lens_model_list)"""
    """print(kwargs_lens)
    print(lens_model_list)"""
    
    lens_model_class = LensModel(lens_model_list=lens_model_list, z_lens=zl, z_source=zs, cosmo=Cosmo) #primer cambio
    lensModel = LensModel(lens_model_list=lens_model_list)
    lensModelExtensions = LensModelExtensions(lensModel=lensModel)
    lensEquationSolver = LensEquationSolver(lens_model_class)
    xima, yima = lensEquationSolver.findBrightImage(source_x, source_y, kwargs_lens,
                                                        min_distance=deltaPix, search_window=numPix * deltaPix, numImages=len(imagenes))
    r_imagenes=[xima,yima]
    """DC=np.array([center_x,center_y])
    print("Diferencia entre centros",DC)
    if np.max(abs(DC))>0.05:
        print("Se aumento el error del centro")"""
    coord=imagenes[["RA","DEC"]].values-dif[["RA","DEC"]].values[0]
    d=np.array([])
    for i in range(len(xima)):
        D=np.power(np.power(coord[::,0]-xima[i],2)+np.power(coord[::,1]-yima[i],2),0.5)
        d=np.append(d,min(D))
    distancia_imagenes=d
    """print(d, "Distancia entre imagenes obtenidos e ingresadas")
    if np.max(abs(d))>0.05:
        print("Distancia maxima mayor a 0.05")
    Lista=[lensModel,kwargs_lens,source_x,source_y,imagenes,dif]"""
    return d_r,r_imagenes,lensModel,kwargs_lens,source_x,source_y,r_imagenes,distancia_imagenes

def graf_lensmodel(pandas,dif,imagenes,pad,NS=False,name_model=None,nombre_sistema=None):
    if type(pandas["p[1]"].iloc[0])==str:
        d_r={i:pandas[i].iloc[[0]].values[0] for i in ["s_x","s_y"]}
        b={i:literal_eval(pandas[i].iloc[[0]].values[0]) for i in ["p[1]","p[2]","p[3]","p[4]","p[5]","p[6]","p[7]","p[8]","p[9]","p[10]"]}
        d_r.update(b)
    else:
        d_r={i:pandas[i].iloc[[0]].values[0] for i in ["s_x","s_y","p[1]","p[2]","p[3]","p[4]","p[5]","p[6]","p[7]","p[8]","p[9]","p[10]"]}
    P1=[]
    T=[]
    for i in range(len(d_r["p[5]"])):
        angulo=d_r["p[5]"][i]-90
        q=1-d_r["p[4]"][i]
        theta_E=np.sqrt((1+q**2)/(2*q))*d_r["p[1]"][i]#theta_E_gravlens
        T.append(round(theta_E,4))
        P1.append(round(d_r["p[1]"][i],4))
    d2=np.power(np.power(pandas["RA"].values-pandas["~RA"].values,2)+np.power(pandas["DEC"].values-pandas["~DEC"].values,2),0.5)
    if NS==False:
        plt.subplots(1, 1, figsize=(10,10), sharex=False, sharey=False)
        plt.scatter(pandas["RA"].values,pandas["DEC"],c="g",marker="D",label="Posición observada imagenes")
        for ii in range(len(pandas["RA"])):
            plt.gca().text(pandas["RA"].values[ii]-0.02, pandas["DEC"].values[ii]-0.02, pandas["image"].values[ii], rotation=0,fontsize=20,zorder=10)
        plt.scatter(pandas["~RA"].values,pandas["~DEC"],marker="D", facecolors='none', edgecolors='r',label="Posición estimada imagenes")
        for i in range(len(d_r["p[5]"])):
            plt.scatter(dif["RA"].values[i]-dif["RA"].values[0],dif["DEC"].values[i]-dif["DEC"].values[0],c="r",label="Posición observada lente")
            plt.scatter(d_r["p[2]"][i],d_r["p[3]"][i],facecolors='none', edgecolors='k',label="Posición estimada de la lente")
        C=getcurves(pad)
        u=C["xc"]
        v=C["yc"]
        s=np.array([])

        for i in range(len(u)-1):
            D=np.power(np.power(u[i]-u[i+1],2)+np.power(v[i]-v[i+1],2),0.5)
            s=np.append(s,D)
        S=np.argmax(s)+1
        #plt.plot(C["xc"][S::],C["yc"][S::],label="criticaB")
        plt.plot(C["uc"][0:S],C["vc"][0:S], alpha=0.5,label="Curva caustica")
        plt.plot(C["xc"][0:S],C["yc"][0:S], alpha=0.5,label="Curva critica")
        left, width = .1, .0
        bottom, height = .1, .0
        right = left + width
        top = bottom + height
        ax = plt.gca()
        #p = plt.Rectangle((left, bottom), width, height, fill=False)
        #p.set_transform(ax.transAxes)
        #p.set_clip_on(False)
        #plt.gca().add_patch(p)
        #print(str(nombre_sistema)+" chi_t= "
         #               +str(pandas['chis_t'].values[0][0])+" model "+str(pandas['model'].values[0])
          #              + " p[1]="+str(P1)+" error_cm="+str(pandas["error_cm"].values[0][0]))
        print(f"{nombre_sistema} model={pandas['model'].values[0]} p[1]={P1}")
        print(f"chi_t = {pandas['chis_t'].values[0]}  error_cm= {pandas['error_cm'].values[0]} ")
        #print(f"chi_t={pandas['chis_t'].values[0]}, error_cm={} ")
        #pandas["error_cm"].values[0]
        print(f"separacion_maxima={np.max(d2)} y separacion_minima={np.min(d2)}") 
        
        
        # plt.gca().text(left, bottom, nombre_sistema+" chi_t= "
        #                +str(pandas['chis_t'].values[0])+" model "+str(pandas['model'].values[0])
        #                + " p[1]="+str(P1)+" error_cm="+str(pandas["error_cm"].values[0]),
        #         horizontalalignment='left',
        #         verticalalignment='top',
        #         transform=ax.transAxes)
        #plt.plot(C["uc"][S::],C["vc"][S::],label="causticaB")
        plt.scatter(pandas["s_x"],pandas["s_y"],c="y", alpha=1,marker="*",label="Posición estimada de la fuente")
        #print(d2)
        plt.ylabel(r'$\alpha$'+"[\"]", fontsize=20)
        plt.xlabel(r'$\delta$'+"[\"]", fontsize=20)
        plt.gca().invert_xaxis()
        plt.legend(bbox_to_anchor =(1.5,1))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(pad+f"/model_{nombre_sistema}_{name_model}.png", bbox_inches='tight', dpi=400)
        plt.show()
    return d2
def Escritor(nombre_sistema,pad,imagenes,banda,dif,z_lens,z_source
             ,error_cm=[0.003],error_flux=0.2,Ngal=1,
             CM=False,noC=False,error_image=0.003,model="SIE"
             ,nombre_run="run_general",R_chi=None,use_real_error_flux=False):
    """"
    nombre_sistema
    pad: directorio donde se realizara todo el proceso
    imagenes: pandas_con informacion
    dif: pandas con informacion lentes
    error_cm list error asociado al centro de masa
    mG:bool si True se modelara las multiples galaxias del sistema lente
    CM: bool si es True se aumentara el error asociado a la galaxia lente
    Ngal: numero de galaxias que se ajustaran
    noC: bool si es True significa que se intentara modelar un sistema sin centro de masa conocido(work in progress)
    error_flux: error maximo en el flujo
    error_image: error en la pos de las imagenes
    """
    
    #calculate first approx p[1]
    alfa_0=np.array([])
    for i in range(len(imagenes)):
        ii=np.power(np.power(imagenes["RA"].values-imagenes["RA"].values[i],2)+np.power(imagenes["DEC"].values-imagenes["DEC"].values[i],2),0.5)
        alfa_0=np.append(alfa_0,ii/2) 
    alfa_0=round(np.max(alfa_0),2)
    #create de directory for the system an the results
    if os.path.isdir(pad)==False:
        os.mkdir(pad)
    if os.path.isdir(pad+"/result")==False:
        os.mkdir(pad+"/result")
    #########Writte data file###############
    name_of_file =str(nombre_sistema)+".dat"
    completeName = os.path.join(pad,name_of_file)
    f = open(completeName, 'w')
    #f.read
    f.write("#"+str(nombre_sistema)+'\n')
    ##########Numero de galaxias#########
    Ngal=1
    error_cm=error_cm#Error general
    if model=="SIE-2G":#multiples galaxias
        Ngal=len(dif)
        if len(error_cm)==1:
            error_cm=[0.05]*Ngal 
        if CM==True:
            error_cm=[0.0501]*Ngal
        print(error_cm)
    f.write(str(Ngal)+"  # number of lens galaxy"+'\n') # cambiar
    #################Error de G#####################
    if CM==True and  model!="SIE-2G":
        if len(imagenes)==2:
            x1,x2=imagenes["RA"].values-dif["RA"].values[0]
            y1,y2=imagenes["DEC"].values-dif["DEC"].values[0]
            p1=np.array([x1,y1])
            p2=np.array([x2,y2])
            p3=np.array([0,0])
            error_cm=[np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)]
        if len(imagenes)>2:
        #   x,y=zip(*dif[["RA","DEC"]].values)#para asignar
            error_cm=[0.05] 
            if np.max(dif[["dRA","dDEC"]].values)>=0.003 and np.max(dif[["dRA","dDEC"]].values)<0.05:
                error_cm=[np.max(dif[["dRA","dDEC"]].values)]
            if np.max(dif[["dRA","dDEC"]].values)==0.003:
                error_cm=[0.003*2]
    if noC==True:#no centro ingresado
        error_cm=[1]
    for ii in range (Ngal):
        ra,dec=dif["RA"].values[ii]-dif["RA"].values[0],dif["DEC"].values[ii]-dif["DEC"].values[0]
        f.write(str(round(ra,3))+" "+str(round(dec,3))+" "+str(error_cm[ii])+" "+str(error_cm[ii])+" # position"+'\n') #arreg
        f.write("0.0 1000. # R_eff observed: 0.59 +/- 0.06"+'\n')
        f.write("0.0 1000. # PA unconstrained in observations"+'\n')
        f.write("0.0 1000. # observed e < 0.07 at 1sigma"+'\n')
    f.write('\n')
    f.write("1 # nºsource"+'\n')
    f.write(str(len(imagenes))+" # n° images of the source"+'\n')
    f.write("#flujo de "+str(banda)+'\n') 
    f.write("#Ra Dec flux sigmax sigmaflux tdel sigma(tdel) part "+'\n') 
    cord_imagenes=imagenes[["RA","DEC"]].values-dif[["RA","DEC"]].values[0]
    flujos=imagenes[banda].values
    error_cord=imagenes[["dRA","dDEC"]].values
    for ii, txt in enumerate(imagenes["component"].values): 
        a=cord_imagenes[:,0][ii]#ra
        b=cord_imagenes[:,1][ii] #Rdec
        c=((-1)**(ii))*10**(((-flujos[ii])+min(flujos))/2.5) #flujo divido
        if np.max(flujos)<=5:
            c=((1)**(ii))*flujos[ii]
        #r_f=np.append(r_f,abs(c))
        #d=eI[ii]
        #if len(error_cord[ii])>1:
        if np.max(flujos)>30:
            c=((-1)**(ii))*flujos[ii]/np.max(flujos)
        d=np.max(error_cord.flatten())
        d=error_image
        #np.sqrt(D[D["image"]!="G"]["dRA"].values[ii]**2+D[D["image"]!="G"]["dDEC"].values[ii]**2) #mejora el chi_p podria seguir poniendo 0 
        if use_real_error_flux==True:
            e=imagenes["d"+band].values[ii]
        g=0 #time delay
        h=1000 #deltatime delay
        f.write(str(round(a,10))+" "+str(round(b,10))+" "
                +str(round(c,10))+" "+str(round(d,10))+" "
                +str(round(e,10))+" "+str(round(g,10))+" "+str(round(h,10))+" "+str(txt)+'\n')
    f.close()
    """
    p[1] = M = mass scale
    (p[2],p[3]) = (x0,y0) = galaxy position
    (p[4],p[5]) = (e,θe) = ellipticity parameters
    (p[6],p[7]) = (γ,θγ) = external shear parameters
    (p[8],p[9]) = (s,a) = misc., often scale radii
     p[10] = α = misc., often a power law index"""
    alfa_0=np.array([])
    for i in range(len(imagenes)):
        ii=np.power(np.power(imagenes["RA"].values-imagenes["RA"].values[i],2)+np.power(imagenes["DEC"].values-imagenes["DEC"].values[i],2),0.5)
        alfa_0=np.append(alfa_0,ii/2) 
    alfa_0=round(np.max(alfa_0),2)
    ###creandocarpetas
    df=open(pad+"/"+nombre_run,'w')
    gf,sr="#set gridflag = 0 # don't need the tiling","#set restart = 2"
    chm="set chimode = 0 #sourceplane"
    if len(imagenes)==2:
        gf="#set gridflag = 0 # don't need the tiling"
        chm="set chimode = 0 #sourceplane"
    if len(imagenes)==4 and model=="SIE+shear":
        sr="set restart = 2"
    inicio=["#Modelado","set omitcore = 0.05","data "+pad+"/"+nombre_sistema+".dat #filewhitdata"
        ,"set omega = 0.3","set lambda = 0.7","set hval = 0.7"
        ,"set hvale = 1000  # Uncertainty in H_0 in units of 100 km/s/Mpc",chm
        ,"set zlens = "+str(z_lens)+"# lens redshift","set zsrc ="+ str(z_source)+"# lens redshift",
       "set checkparity = 0    # don't worry about parities",gf,sr,
       "#Start modeling","startup "+str(Ngal)+" 1 #startup Ngal 1"]
    third="third"#ultimo archivo para realizarle el mcmc
    if len(imagenes)==2:
        if model=="SIS":
            L3=["alpha "+str(alfa_0)+" 0 0 0 0.0 0 0 0 0 1","1 1 1 0 0 0 0 0 0 0"
                ,"optimize "+pad+"/third"]
            #L2=["#secondpart","setlens "+pad+"/first.start","changevary 1","1 1 1 0 0 0 0 0 0 0  #second","optimize "+pad+"/second"]
            #L3=["#thirdpart","setlens "+pad+"/first.start","changevary 1"," 1 1 1 0 0 0 0 0 0 0 # third","optimize "+pad+"/third"]
            L = itertools.chain(L3)
        if model=="SIE":
            L1=["alpha "+str(alfa_0)+" 0 0 0.1 10.0 0 0 0 0 1","1 0 0 1 1 0 0 0 0 0"
                ,"varyone 1 5 -90.0 90.0 19 "+pad+"/first"]
            #L2=["#secondpart","setlens "+pad+"/first.start","changevary 1","1 1 1 0 0 0 0 0 0 0  #second","optimize "+pad+"/second"]
            L3=["#thirdpart","setlens "+pad+"/first.start","changevary 1"," 1 0 0 1 1 0 0 0 0 0 # third","optimize "+pad+"/third"]
            L = itertools.chain(L1,L3)
        if model=="SIS+shear":
            L1=["alpha "+str(alfa_0)+" 0 0 0. 0.0 0,1 10 0 0 1","1 0 0 0 0 1 1 0 0 0"
                ,"varyone 1 7 -90.0 90.0 19 "+pad+"/first"]
            #L2=["#secondpart","setlens "+pad+"/first.start","changevary 1","1 1 1 0 0 0 0 0 0 0  #second","optimize "+pad+"/second"]
            L3=["#thirdpart","setlens "+pad+"/first.start","changevary 1"," 1 0 0 1 1 0 0 0 0 0 # third","optimize "+pad+"/third"]
            L = itertools.chain(L1,L3)
    if len(imagenes)>2: #always whit shear
        L1=["alpha "+str(alfa_0)+" 0 0 0.03 10.0 0 0.0 0 0 1","1 0 0 1 1 0 0 0 0 0"
            ,"varyone 1 5 -90.0 90.0 19 "+pad+"/first"]
        L2=["#secondpart","setlens "+pad+"/first.start","changevary 1","1 0 0 1 1 0 0 0 0 0 #second","optimize "+pad+"/second"]

        L3=["#thirdpart","set chimode= 1","setlens "+pad+"/second.start","changevary 1"," 1 1 1 1 1 0 0 0 0 0 # third","optimize "+pad+"/third"]
        L = itertools.chain(L1,L2,L3)
        if model=="SIE+shear": #"set restart = 2",
            L1=["alpha "+str(alfa_0)+" 0 0 0.1 10.0 0.1 10.0 0 0 1","1 0 0 1 1 0 0 0 0 0"
            ,"varytwo 1 5 -90.0 90.0 19 1 7 -90.0 90.0 19 "+pad+"/first"]
            L2=["#secondpart","setlens "+pad+"/first.start","changevary 1","1 1 1 1 1 0 0 0 0 0  #second","optimize "+pad+"/second"]
            L3=["#thirdpart","setlens "+pad+"/second.start","changevary 1"," 1 1 1 1 1 1 1 0 0 0 # third","optimize "+pad+"/third"]
            L = itertools.chain(L1,L2,L3)
        # if model=="POW" and len(imagenes)==4:
        #     L1=["alpha "+str(alfa_0)+" 0 0 0.03 10.0 0 0.0 0 0 1","1 0 0 1 1 0 0 0 0 0"
        #     ,"varyone 1 5 -90.0 90.0 19 "+pad+"/first"]
        #     L2=["#secondpart","setlens "+pad+"/first.start","changevary 1","1 0 0 1 1 0 0 0 0 1 #second","optimize "+pad+"/second"]

        #     L3=["#thirdpart","set chimode= 1","setlens "+pad+"/second.start","changevary 1"," 1 1 1 1 1 0 0 0 0 1 # third","optimize "+pad+"/third"]
    
    if model=="SIE-2G":
        print("este ajuste solo esta habilitado para sistemas con 2G")
        ra,dec=dif["RA"].values-dif["RA"].values[0],dif["DEC"].values-dif["DEC"].values[0]
        L1=[]
        for i in range(len(ra)):
            L1.append("alpha "+str(alfa_0/2)+" "+str(ra[i])+" "+str(dec[i]) +" 0.1 10.0 0 0.0 0 0 1")
        for i in range(len(ra)):
            L1.append("1 0 0 1 1 0 0 0 0 0")
        L1.append("varytwo 1 5 -90.0 90.0 19 2 5 -90.0 90.0 19 "+pad+"/first")   
        L2=["#secondpart","setlens "+pad+"/first.start","changevary 1","1 1 1 0 0 0 0 0 0 0 #second","1 1 1 0 0 0 0 0 0 0","optimize "+pad+"/second"]
        L3=["#thirdpart","setlens "+pad+"/second.start","changevary 1"," 1 0 0 1 1 0 0 0 0 0 # third","1 0 0 1 1 0 0 0 0 0","optimize "+pad+"/third"]
        L = itertools.chain(L1,L2,L3)
    final=["set plotmode    = 2","kapgam 3 "+pad+"/kappa_gamma.dat","plotcrit "+pad+"/critT.dat",
          "plotgrid "+pad+"/gridT.dat",
           "calcRein 0 "+pad+"/RE.dat","quit"]
    if model=="SIE-2G" and CM==True:
        ra,dec=dif["RA"].values-dif["RA"].values[0],dif["DEC"].values-dif["DEC"].values[0]
        #values=R_chi[R_model["model"]=="SIE+shear"]
        L1=[]
        for i in range(len(ra)):
            if i==0:
                L1.append("alpha "+str(alfa_0)+" "+str(ra[i])+" "+str(dec[i]) +" 0.1 10.0 0 0.0 0 0 1")
            else:
                L1.append("alpha "+str(alfa_0)+" "+str(ra[i])+" "+str(dec[i]) +" 0.1 10.0 0 0.0 0 0 1")
        for i in range(len(ra)):
            L1.append("1 0 0 1 1 0 0 0 0 0")
        L1.append("varytwo 1 5 -90.0 90.0 19 2 5 -90.0 90.0 19 "+pad+"/first")   
        L2=["#secondpart","setlens "+pad+"/first.start","changevary 1","1 1 1 0 0 0 0 0 0 0 #second","1 1 1 0 0 0 0 0 0 0","optimize "+pad+"/second"]
        L3=["#thirdpart","setlens "+pad+"/second.start","changevary 1"," 1 1 1 1 1 0 0 0 0 0 # third","1 1 1 1 1 0 0 0 0 0","optimize "+pad+"/third"]
        L = itertools.chain(L1,L2,L3)
    if model=="SIE-2G" and CM==True and R_chi!=None and len(imagenes)>1000:
        ra,dec=dif["RA"].values-dif["RA"].values[0],dif["DEC"].values-dif["DEC"].values[0]
        values=R_chi[R_model["model"]=="SIE+shear"]
        L1=[]
        for i in range(len(ra)):
            L1.append("alpha "+str(alfa_0/2)+" "+str(ra[i])+" "+str(dec[i]) +" 0.1 10.0 0 0.0 0 0 1")
        for i in range(len(ra)):
            L1.append("1 0 0 1 1 0 0 0 0 0")
        L1.append("varytwo 1 5 -90.0 90.0 19 2 5 -90.0 90.0 19 "+pad+"/first")   
        L2=["#secondpart","setlens "+pad+"/first.start","changevary 1","1 1 1 0 0 0 0 0 0 0 #second","1 1 1 0 0 0 0 0 0 0","optimize "+pad+"/second"]
        L3=["#thirdpart","setlens "+pad+"/second.start","changevary 1"," 1 1 1 1 1 0 0 0 0 0 # third","1 1 1 1 1 0 0 0 0 0","optimize "+pad+"/third"]
        L = itertools.chain(L1,L2,L3)
    if nombre_run=="mcmc":    
        gf="set gridflag = 0 # don't need the tiling"
        chm="set chimode = 0 #sourceplane"
        sr="#set restart = 2"
        inicio=["#Modelado","set omitcore = 0.05","data "+pad+"/"+nombre_sistema+".dat #filewhitdata"
        ,"set omega = 0.3","set lambda = 0.7","set hval = 0.7"
        ,"set hvale = 1000  # Uncertainty in H_0 in units of 100 km/s/Mpc",chm
        ,"set zlens = "+str(z_lens)+"# lens redshift","set zsrc ="+ str(z_source)+"# lens redshift",
       "set checkparity = 0    # don't worry about parities",gf,sr]
        L=["setlens  "+pad+"/result/"+third+".start","MCMCset 2 Nchain 10 maxsteps 10000","MCMCrun "+pad+"/mcmc"]
        final=["quit"]
    #una nueva run ,"setlens  "+pad+"/"+third+".start","MCMCset 2 Nchain 10","MCMCrun "+pad+"/mcmc"
    for i in inicio:
        df.write(i)
        df.write('\n')
    for i in L:
        df.write(i)
        df.write('\n')
    for i in final:
        df.write(i)
        df.write('\n')
    df.close()
    if use_real_error_flux==True:
        error_flux=imagenes["d"+band].values
    return error_cm,error_flux

def mcmc(pad):
    if os.path.isfile(pad+"/"+"mcmc.dat")==False:
        print("realizando el MCMC (aqui esta lensmodel)")
        lensmodel(pad,nombre_run)
    if os.path.isfile(pad+"/"+"mcmc_r.csv")==False:
        print("creando CSV")
        import multiprocessing
        f=open(pad+"/mcmc.dat","r")
        lines=f.readlines()
        global alfas #para que solucione el error picke variable
        def alfas(l):
            A,B,C=None,None,None
            if "alpha" in l:
                #Parameters.append(l)
                critic=[]
                for i,text in enumerate(l):
                    if text==" " and l[i-1]!=" ": 
                        critic.append(int(i))
                critic.append(len(l))
                A=[l[critic[i]:critic[i+1]] for i in range(len(critic)-1)]
                #L=L.append("alfa")
            if "CHISQ:" in l:
                B=float(l.replace("CHISQ:",""))
            if "ptsrc" in l:
                critic=[]
                for i,text in enumerate(l):
                    if text==" " and l[i-1]!=" ": 
                        critic.append(int(i))
                critic.append(len(l))
                C=[l[critic[i]:critic[i+1]] for i in range(len(critic)-1)]
            return A,B,C
        ncpu = multiprocessing.cpu_count()
        #print(ncpu)
        if __name__ == "__main__":
            pool = multiprocessing.Pool(ncpu)
            out1, out2, out3 = zip(*pool.map(alfas, lines))
        Alfas,Chis,Ptsrc  = list(filter(None,out1)),list(filter(None,out2)),list(filter(None,out3))
        LP=np.array(Alfas)
        Chi=np.array(Chis)
        SP=np.array(Ptsrc)
        if len(LP)==2*len(Chi):
            LP=LP[::2]
        Dic={"Chi":0,"alpha":LP[::,0],"p[1]":0,"p[2]":0,"p[3]":0,"p[4]":0,"p[5]":0,"p[6]":0
        ,"p[7]":0,"p[8]":0,"p[9]":0,"p[10]":0,"ptsrc":SP[::,0],"s[1]":0,"s[2]":0,"s[3]":0,"s[4]":0,"s[5]":0,"s[6]":0,"s[7]":0}
        for i,txt in enumerate([*Dic]):
            if i==0:
                Dic[txt]=Chi
                continue
            if i>0 and i<12:
                if txt=="alpha":
                    continue
                Dic[txt]=LP[::,i-1].astype(float)
                limit=i
                continue
            else:
                if txt=="ptsrc":
                    continue
                i=i-limit
                Dic[txt]=SP[::,i]
        if len(Dic["alpha"])/len(Dic['s[1]']) >1:
            print("sistema con multiples galaxias lentes solo se guardara los parametros de la principal")
            F=int(len(Dic["alpha"])/len(Dic['s[1]']))
            for k in ['alpha', 'p[1]', 'p[2]', 'p[3]', 'p[4]', 'p[5]', 'p[6]', 'p[7]', 'p[8]', 'p[9]', 'p[10]']:
                Dic[k]=Dic[k][::F]
        Dic=pd.DataFrame(Dic)
        Dic.to_csv(pad+"/"+"mcmc_r.csv",index=False)
    return print("MCMC listo") 