from astropy.cosmology import FlatLambdaCDM
#,FlatwCDM,Flatw0waCDM,FlatJBP,FlatBA,FlatINTR,FlatCARD
from astropy import constants as const
from scipy import interpolate
from astropy import units as u
import sympy as sp
from sympy import symbols, diff
import statsmodels.api as sm
import numpy as np
import pandas as pd

def noleter(L):
    if type(L) != str:
        l=[]
        for ii in L:
            for i in [chr(i) for i in range(ord('A'),ord('Z')+1)]:
                    ii=ii.replace(i,"").replace("†","")
            l.append(ii)
        L=l
    else:
        for i in [chr(i) for i in range(ord('A'),ord('Z')+1)]:
            L=L.replace(i,"").replace("†","")
    return L
def codigoxd(txt):
    if txt=="None":
        return None
    xd=[]
    for i in range(len(txt)):
        if ","==txt[i] or "["==txt[i] or "]"==txt[i]:
            xd.append(i)
            #print(i)
    return np.array([float(txt[xd[ii]:xd[ii+1]].replace("[","").replace(",","")) for ii in range(0,len(xd)-1)])

class microlensingtimescale(object):
    cms = u.cm / u.s#cm/s
    G=const.G.to("cm3/s2solMass").value#cm/(s2SM)
    c=const.c.to("cm/s").value
    def __init__(self,cosmologia):
#         self.target = target   #may not need this
        self.cosmologia = cosmologia
    
    def dsig_pec_t(self,z):
        import statsmodels.api as sm
        import numpy as np
        x = np.array([0.,.5,1.0,2.0,3.0])
        y = np.array([179.544800,162.386536,144.020065,114.992302,95.906265])
        # include constant in ols models, which is not done by default
        x = sm.add_constant(x)
        model = sm.OLS(y,x)
        results = model.fit()
        y0,m=results.params
        dy0,dm=results.bse#
        y01,m1=results.params-results.bse
        y02,m2=results.params+results.bse
        yb=z*m+y0
        ymax,ymin=z*m1+y01,z*m2+y02
        yerror=abs((ymax-ymin)/2)
        return yb,yerror

    def dRe(self,zl,zs,M=0.3,dM=0.03,c=c,G=G): #radio einstein
        #zl lente
        #zs source
        #M solar mass es 0.3 segun paper profe error de esto?
        cos=self.cosmologia
        L={"Da_s":cos.angular_diameter_distance(zs).to(u.cm).value,
        "Da_l":cos.angular_diameter_distance(zl).to(u.cm).value,
        "Da_ls":cos.angular_diameter_distance_z1z2(zl,zs).to(u.cm).value,"M":M,"G":G,"c":c}
        [Da_s,Da_l,Da_ls,M,G,c]=symbols([*L], real=True)
        v=[Da_s,Da_l,Da_ls,M,G,c]
        f=Da_s*(4*G*M*Da_ls/(Da_s*Da_l*c**2))**(0.5)
        F=f.evalf(subs={v[i]:L[str(v[i])] for i in range(len(v))})
        df=abs(diff(f, M).evalf(subs={v[i]:L[str(v[i])] for i in range(len(v))}))*dM
        return float(F),float(df)#en cm
    
    def dRs(self,m,dm,z,zpt,lam,c=c,G=G,i=np.pi/3):
        #ya esta haciendo el cambio de variable 
        cos=self.cosmologia
        lamI=0.8140#micrometros
        L={"h":cos.H0.value/100,"r_h":c/cos.H0.to("cm/Mpc s").value,
        "Da":cos.angular_diameter_distance(z).value,"i":np.pi/3,"lamda":lam,"zpt":zpt,"m":m}
        [h,r_h,Da,i,lamda,zpt,m]=symbols([*L], real=True)
        v=[h,r_h,Da,i,lamda,zpt,m]
        f=((3.4*10**(15))/sp.sqrt(sp.cos(i)))*(Da/r_h)*((lamda)**(3/2))*((zpt/3631)**(0.5))*(10**(-0.2*(m-19)))*(h**(-1))
        df=abs(diff(f, m).evalf(subs={v[i]:L[str(v[i])] for i in range(len(v))}))*dm*(lamI/ L["lamda"])**(4/3)
        #https://iopscience.iop.org/article/10.3847/1538-4357/aac9bb/pdf equation (3)
        F=f.evalf(subs={v[i]:L[str(v[i])] for i in range(len(v))})*(lamI/ L["lamda"])**(4/3)
        return float(F),float(df)
    
    def dsig_d_s(self,teta_e,dteta_e,zl,zs,c=c):#velocity dispersion of the stars in the lens galaxy
        #teta_e radio de einstein en arcsecons
        #zl,zs redshif fuente y lente
        #teta_e en arcsec(")
        cos=self.cosmologia
        L={"teta_e":teta_e*u.arcsec.to("rad"),"c":(c*u.cm/u.s).to("km/s").value,
        "Da_s":cos.angular_diameter_distance(zs).value,
        "Da_l":cos.angular_diameter_distance(zl).value,
        "Da_ls":cos.angular_diameter_distance_z1z2(zl,zs).value,"pi":np.pi}
        [teta_e,c,Da_s,Da_l,Da_ls,pi]=symbols([*L], real=True)
        v=[teta_e,c,Da_s,Da_l,Da_ls,pi]
        f=0.5*c*sp.sqrt(teta_e*Da_s/(pi*Da_ls))
        dteta_e=dteta_e*u.arcsec.to("rad")
        df=abs(diff(f, teta_e).evalf(subs={v[i]:L[str(v[i])] for i in range(len(v))}))*dteta_e
        F=f.evalf(subs={v[i]:L[str(v[i])] for i in range(len(v))})
        return  float(F),float(df)
    
    def dVe(self,teta_e,dteta_e,zl,zs,v_CMB=369,dvcmb=0.9): # separation redshift lens redshift source peculiar velocity lens peculiar velocity source
        #teta_e radio de einsteint en arcsec(") y dteta_e error
        #v_cmb km/s#buscar esto en el paper de hinshaw 2009 #f 369.0 ± 0.9 km s−1   
        cos=self.cosmologia
        p_l,dp_l=self.dsig_pec_t(zl)
        p_s,dp_s=self.dsig_pec_t(zs)
        v_d,dv_d=self.dsig_d_s(teta_e,dteta_e,zl,zs)
        L={"Da_s":cos.angular_diameter_distance(zs).to(u.km).value,
        "Da_l":cos.angular_diameter_distance(zl).to(u.km).value,
        "Da_ls":cos.angular_diameter_distance_z1z2(zl,zs).to(u.km).value,
        "p_l":p_l,"p_s":p_l,"v_d":v_d,"v_CMB":v_CMB,"zl":zl,"zs":zs}
        #p_l velocidad peculiar lente
        #p_s velocidad peculiar fuente
        #v_d velocidad de dispercion
        [Da_s,Da_l,Da_ls,p_l,p_s,v_d,v_CMB,zl,zs]=symbols([*L], real=True)
        v=[Da_s,Da_l,Da_ls,p_l,p_s,v_d,v_CMB,zl,zs]
        f=sp.sqrt(((p_l*Da_s/((1+zl)*(Da_l)))**2)+((p_s/(1+zs))**2)+((v_CMB*Da_ls/((1+zl)*Da_l))**2)+(2*(v_d*Da_s/((1+zl)*Da_l))**2))
        df=abs(diff(f, p_l).evalf(subs={v[i]:L[str(v[i])] for i in range(len(v))}))*dp_l
        +abs(diff(f, p_s).evalf(subs={v[i]:L[str(v[i])] for i in range(len(v))}))*dp_s
        +abs(diff(f, v_CMB).evalf(subs={v[i]:L[str(v[i])] for i in range(len(v))}))*dvcmb
        +abs(diff(f, v_d).evalf(subs={v[i]:L[str(v[i])] for i in range(len(v))}))*dv_d
        F=f.evalf(subs={v[i]:L[str(v[i])] for i in range(len(v))})
        return float(F),float(df) 
def teta(nombre_sistema,pandas,columna="noleter"):
    #b valor de p[0]
    #e valor de p[4]
    #db error de p[0]
    #de error de p[4]
    #pad=work_pad+models+"/"+str(nombre_sistema)
    #p=pd.read_csv(pad+"/result/"+'finalparameters_'+str(nombre_sistema)+".csv")
    try:
        p=pandas[pandas[columna]==nombre_sistema]
    except:
        print("revisar columnas")
        return
    #print(p["p[1]"+"_f"].values[0],((p["up_p[1]"+"_f"])+abs(p["down_p[1]"+"_f"]))/2)
    try:
        b=p["p[1]"+"_f"].values[0]
        db=((p["up_p[1]"+"_f"])+abs(p["down_p[1]"+"_f"]))/2
        db=db.values[0]
    except:
        b=p["p[1]"+"_f"]
        db=((p["up_p[1]"+"_f"])+abs(p["down_p[1]"+"_f"]))/2
        db=db[0]
    try:
        e=p["p[4]"+"_f"].values[0]
        de=((p["up_p[4]"+"_f"])+abs(p["down_p[4]"+"_f"]))/2
        de=de.values[0]
    except:
        e=0
        de=0
    theta_E=np.sqrt((1+(1-e)**2)/(2*(1-e)))*b#theta_E_gravlens
    etheta_E=np.sqrt((1+(1-e)**2)/(2*(1-e)))*db+b*((((1-e)**2+1)/((1-e)**2))-2)/(2*np.sqrt(2)*np.sqrt((((1-e)**2+1)/((1-e)))))*de
    return float(theta_E),float(etheta_E),float(e)
def zpt_lam(nombre_sistema,pandas,columna="noleter"): 
    #['g', 'r',"V","R","I","H",'i','z','Y','F160W','F814W','F475X',"G","5GHz","B","Kp"]
    try:
        p=pandas[pandas[columna]==nombre_sistema]
    except:
        print("revisar columnas")
        return
    System,band=p[["Psystem","banda"]].values[0]
    #i,f814,I  (λ = 0.814 µm and zpt= 2409) mosquera
    if System=="AB":
        zpt=3631
    if System=="vega":
        if band in ["Kp"]:
            zpt=676.31
        if band in ["i","I","F814W"]:
            zpt=2409
        if band in ["R","F160W","5GHz","r"]:
            zpt=1132
        if band=="G":
            zpt=2847.56
        if band=="R":
            zpt=3989.43
    if band in ["i","I","F814W"]:
        lam=0.814#micrometros
    if band in ["Kp"]:
        lam=21125.25/10000
    if band in ["G"]:
        lam=0.672
    if band in ["R"]:
        lam=4750.64/10000
    if band in ["R","F160W","5GHz","r"]:
        lam=15435/10000
    if band in ["B"]:
        lam=4336.33/10000
    if band in ['F475X']:
        lam=4937.39/10000
    return zpt,lam
def Result_pandas(cosmologia,nombres=[],GPanda=[],RPanda=[]):
    #Rλ ∝ λ4/3
    Cms=microlensingtimescale(cosmologia)#calculos basados en una cosmologia especifica
    D=pd.DataFrame()
    D["name"]=nombres
    D['model']=np.array([RPanda[RPanda["name"]==i]['model'].values[0] for i in nombres])
    D[["zl","zs"]]=np.array([GPanda[GPanda["name"]==i][["z_l","z_s"]].values[0] for i in nombres])
    D["zl"][D["zl"]=="-"]=0.5*np.ones(len(D["zl"]=="-"))
    D["zs"][D["zs"]=="-"]=2*np.ones(len(D["zs"]=="-"))
    D["zl"]=[i.replace("?","") if type(i)==str else i for i in D["zl"].values]
    D["zs"]=[i.replace("?","") if type(i)==str else i for i in D["zs"].values]
    D[["Rex16cm","dRex16cm"]]=np.around(np.array([Cms.dRe(zl,zs) for zl,zs in D[["zl","zs"]].astype(float).values])*10**(-16),4)
    D[["m","dm"]]=[[np.mean(RPanda[RPanda["name"]==i]["corrected"]),1.2*np.std(RPanda[RPanda["name"]==i]["corrected"])] for i in nombres]
    #suponiendo 20% de error
    #D[["m","dm"]]=[[np.mean(RPanda[RPanda["name"]==i]["corrected"]),0.1*np.mean(RPanda[RPanda["name"]==i]["corrected"])] for i in nombres]
    D[["zpt","lam"]]=[zpt_lam(i,RPanda) for i in nombres]
    D[["Rsx15cm","dRsx15cm"]]=np.round(np.array([Cms.dRs(m,dm,z,zpt,lam) for m,dm,z,zpt,lam in D[["m","dm","zs","zpt","lam"]].astype(float).values])*10**(-15),4)
    D[["teta_e","dteta_e","e"]]=[teta(nombre_sistema,RPanda) for nombre_sistema in nombres]
    D[["Vexkm/s","dVexkm/s"]]=np.around(np.array([Cms.dVe(teta_e,dteta_e,zl,zs) for teta_e,dteta_e,zl,zs in D[["teta_e","dteta_e","zl","zs"]].astype(float).values]),4)
    D["te[yr]"]=np.around((D["Rex16cm"]*10**(16)/(D["Vexkm/s"]*u.km.to("cm")))*u.s.to("yr"),4)
    D["dte[yr]"]=np.around((abs((D["Rex16cm"]*10**(16)/((D["Vexkm/s"]*u.km.to("cm"))**2)))*D["dVexkm/s"]*u.km.to("cm")
    +abs(1/(D["Vexkm/s"]*u.km.to("cm")))*D["dRex16cm"]*10**(16))*u.s.to("yr"),4)
    D["ts[yr]"]=np.around((D["Rsx15cm"]*10**(15)/(D["Vexkm/s"]*u.km.to("cm")))*u.s.to("yr"),4)
    D["dts[yr]"]=np.around((abs((D["Rsx15cm"]*10**(15)/((D["Vexkm/s"]*u.km.to("cm"))**2)))*D["dVexkm/s"]*u.km.to("cm")
    +abs(1/(D["Vexkm/s"]*u.km.to("cm")))*D["dRsx15cm"]*10**(15))*u.s.to("yr"),4)
    return D
#cosM=FlatLambdaCDM(H0=72,Om0=0.3, Ob0=0.,Neff=3.04) #Mosquera
#Microlensing_Thomas=Result_pandas(cosM,nombres=resultadosThomas.name.drop_duplicates(),GPanda=datosThomas,RPanda=resultadosThomas)
def Result_pandasv2(cosmologia,nombres=[],GPanda=[],RPanda=[]):
    #Rλ ∝ λ4/3
    Cms=microlensingtimescale(cosmologia)#calculos basados en una cosmologia especifica
    D=pd.DataFrame()
    D["name"]=nombres
    nombres_sinletra=noleter(nombres)
    RPanda["noleter"]=noleter(RPanda.name)
    GPanda["noleter"]=noleter(GPanda.name)
    D['model']=np.array([RPanda[RPanda["noleter"]==i]['model'].values[0] for i in nombres_sinletra])
    D["zl"] = [float(str(GPanda[GPanda["noleter"]==name]["z_l"].drop_duplicates().values[0]).replace("-","0.5").replace("?","")) for name in nombres_sinletra]
    D["zs"] = [float(str(GPanda[GPanda["noleter"]==name]["z_s"].drop_duplicates().values[0]).replace("-","2.0").replace("?","")) for name in nombres_sinletra]
    D[["Rex16cm","dRex16cm"]]=np.around(np.array([Cms.dRe(zl,zs) for zl,zs in D[["zl","zs"]].astype(float).values])*10**(-16),4)
    D[["m","dm"]]=[[np.mean(RPanda[RPanda["noleter"]==i]["corrected"]),1.2*np.std(RPanda[RPanda["noleter"]==i]["corrected"])] for i in nombres_sinletra]
    #suponiendo 20% de error
    #D[["m","dm"]]=[[np.mean(RPanda[RPanda["name"]==i]["corrected"]),0.1*np.mean(RPanda[RPanda["name"]==i]["corrected"])] for i in nombres]
    D[["zpt","lam"]]=[zpt_lam(i,RPanda) for i in nombres_sinletra]
    D[["Rsx15cm","dRsx15cm"]]=np.round(np.array([Cms.dRs(m,dm,z,zpt,lam) for m,dm,z,zpt,lam in D[["m","dm","zs","zpt","lam"]].astype(float).values])*10**(-15),4)
    D[["teta_e","dteta_e","e"]]=[teta(nombre_sistema,RPanda) for nombre_sistema in nombres_sinletra]
    D[["Vexkm/s","dVexkm/s"]]=np.around(np.array([Cms.dVe(teta_e,dteta_e,zl,zs) for teta_e,dteta_e,zl,zs in D[["teta_e","dteta_e","zl","zs"]].astype(float).values]),4)
    D["te[yr]"]=np.around((D["Rex16cm"]*10**(16)/(D["Vexkm/s"]*u.km.to("cm")))*u.s.to("yr"),4)
    D["dte[yr]"]=np.around((abs((D["Rex16cm"]*10**(16)/((D["Vexkm/s"]*u.km.to("cm"))**2)))*D["dVexkm/s"]*u.km.to("cm")
    +abs(1/(D["Vexkm/s"]*u.km.to("cm")))*D["dRex16cm"]*10**(16))*u.s.to("yr"),4)
    D["ts[yr]"]=np.around((D["Rsx15cm"]*10**(15)/(D["Vexkm/s"]*u.km.to("cm")))*u.s.to("yr"),4)
    D["dts[yr]"]=np.around((abs((D["Rsx15cm"]*10**(15)/((D["Vexkm/s"]*u.km.to("cm"))**2)))*D["dVexkm/s"]*u.km.to("cm")
    +abs(1/(D["Vexkm/s"]*u.km.to("cm")))*D["dRsx15cm"]*10**(15))*u.s.to("yr"),4)
    return D
def mag_to_flux(mag):
    return 10 ** (-0.4 * mag)

def flux_to_mag(flux):
    return -2.5 * np.log10(flux)

def mean_magnitude_with_error(magnitudes, errors):
    # Convert magnitudes to fluxes
    fluxes = [mag_to_flux(mag) for mag in magnitudes]
    # Convert errors in magnitudes to errors in fluxes
    flux_errors = [(flux) * np.log(10) * 0.4 * (err+1e-5) for flux, err in zip(fluxes, errors)]

    # Calculate the weighted mean flux and its error
    weights = [1 / (err ** 2) for err in flux_errors]
    weighted_mean_flux = np.sum([flux * 1 for flux, weight in zip(fluxes, weights)]) / np.sum(1)
    weighted_mean_flux_error = np.sqrt(1 / np.sum(weights))

    # Convert the weighted mean flux and its error back to magnitude and error in magnitude
    mean_mag = flux_to_mag(weighted_mean_flux)
    mean_mag_error = weighted_mean_flux_error / (weighted_mean_flux * np.log(10) * 0.4)

    return mean_mag, mean_mag_error