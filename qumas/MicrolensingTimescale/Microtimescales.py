from astropy import units as u
from astropy import constants as const
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LinearModel
from lmfit import Model
import uncertainties
from uncertainties import unumpy,ufloat
from uncertainties.core import AffineScalarFunc
from .v_cmb import calculate_projections


class MicrolensingTimescale:
    #?
    #Not sure about the errors maybe can think about it again in the future 
    #cm_s = u.cm / u.s# cm/s
    # Global constants with default values
    G = 1.3271244e+26  # cm³/s²/solar mass
    c = 2.998e+10  # Speed of light in cm/s
    gamma = 0.55  # Growth index for LambdaCDM
    #sigma_pec(0)=439+/-69 (Hess & Kitaura, 2016)
    sigma_pec_0 = 439  
    dsigma_pec_0 = 69  # Uncertainty in peculiar velocity dispersion
    ######################
    M = 0.3  # Lens mass in solar masses
    p_dM = 0.1  # Percentage error in lens mass
    i = np.pi / 3  # Inclination angle in radians
    #################################
    v0 = 369.82  # CMB velocity in km/s
    dv0 = 0.11  # Uncertainty in CMB velocity
    ##################################
    z_pec,v_pec = np.array([0.,.5,1.0,2.0,3.0]), np.array([179.544800,162.386536,144.020065,114.992302,95.906265])
    fit_model = Model(LinearModel().func, independent_vars=['x'])
    model_v_pec = fit_model.fit(v_pec, x=z_pec)
    def __init__(self,cosmology=None,H=70,om=0.3):
        self.H = H
        self.om = om
        self.gamma = 0.55
        self.c = 2.998e+10  # Speed of light in cm/s
        self.G =  1.3271244e+26  # cm³/(s²*solar mass)
        self.cosmology = cosmology or FlatLambdaCDM(self.H,self.om)    
    # def sigma_pec(self, z):
    #     """
    #     Calculate the peculiar velocity dispersion at redshift z.
    #     """
    #     f = (unumpy.uarray(self.sigma_pec_0, self.dsigma_pec_0) * self.cosmology.Om(z)**self.gamma) / \
    #         (((1+z)**0.5) * self.cosmology.Om(0)**self.gamma)
        
    #     return self._convert_uncertainties(f)
    def enclosed_mass(self,zl,zs,teta_e,dteta_e,G=None,c=None,Da=None,Da_ls=None,return_all=False):
        G = G or self.G
        c = c or self.c
        Da = Da or self.cosmology.angular_diameter_distance([zl,zs]).to(u.cm).value
        Da_ls = Da_ls or self.cosmology.angular_diameter_distance_z1z2(zl,zs).to(u.cm).value
        Re = unumpy.uarray(teta_e, dteta_e) * 4.84813681109536e-06 #* Da[0]
        M_e = (Re**2*Da[0] * Da[1] * c**2)/(4* G* Da_ls)
        if return_all:
            return np.concatenate([MicrolensingTimescale.to_return(M_e),MicrolensingTimescale.to_return(Re*Da[0]*3.2407792894443653e-22)],axis=1)
        return MicrolensingTimescale.to_return(M_e)
    def cosmological_(self,zl,zs,G=None,c=None,Da=None,Da_ls=None):
        #just get the ratio between the angular diameter distance
        G = G or self.G
        c = c or self.c
        Da = Da or self.cosmology.angular_diameter_distance([zl,zs]).to(u.cm).value
        Da_ls = Da_ls or self.cosmology.angular_diameter_distance_z1z2(zl,zs).to(u.cm).value
        return (Da[0] * Da[1])/Da_ls
    def sigma_pec(self,z, gamma=None,sigma_pec_0=None,dsigma_pec_0=None):
        """
        for vp eq (9) from https://iopscience.iop.org/article/10.3847/0004-637X/832/1/46
        for f eq (8) from  https://ui.adsabs.harvard.edu/abs/2022MNRAS.509.2994A/abstract
        
        gamma -- Growth index, default is 0.55 for LambdaCDM
        return values of peculiar velocity in km/s
        235 evencio
        """
        gamma = gamma or self.gamma
        sigma_pec_0 = sigma_pec_0 or self.sigma_pec_0
        dsigma_pec_0 = dsigma_pec_0  or self.dsigma_pec_0
        f_z = self.cosmology.Om(z)**gamma
        f_0 = self.cosmology.Om(0)**gamma
        vp = (unumpy.uarray(sigma_pec_0, dsigma_pec_0) * f_z)/(((1+z)**0.5)* f_0)

        return MicrolensingTimescale.to_return(vp)
    def Re(self,zl,zs,M=None,c=c,G=G,p_dM=p_dM,Da=None,Da_ls=None):
        """Einstein radius of the lens 
        for Re eq 1 from https://iopscience.iop.org/article/10.1088/0004-637X/738/1/96
        """
        M = M or self.M
        p_dM = p_dM
        dM = p_dM * M
        #if zl>zs:
        #   return "zl should be minor than zs"
        #Estos podrian estar como variables globales ?
        if Da is None:
            Da = self.cosmology.angular_diameter_distance([zl,zs]).to(u.cm).value
        if Da_ls is None:
            Da_ls = self.cosmology.angular_diameter_distance_z1z2(zl,zs).to(u.cm).value
        Re = Da[1]*(4* G *unumpy.uarray(M, dM) *Da_ls/ (Da[0] * Da[1] * c**2))**(0.5)
        return MicrolensingTimescale.to_return(Re)
        
    def Rs(self,z,m,dm,zpt,lamda,c=c,i=None,Da=None,**kwards):
        """Source radius 
        for Re eq 3 from https://iopscience.iop.org/article/10.1088/0004-637X/738/1/96
        and also re scale the radius to lamda 0.814 micro meter 
        """
        i = i or self.i
        h = self.H / 100
        r_h = self.c / self.cosmology.H(0).to("cm/cm s").value
        if Da is None:
            Da = self.cosmology.angular_diameter_distance(z).to(u.cm).value
        re_scale_factor = (0.814/(lamda/(1+z)))**(4/3)
        f= re_scale_factor * 3.4e15*(Da/r_h)*(lamda**(3/2))*((zpt/3631)**(0.5))*(10**(-0.2*(unumpy.uarray(m,dm)-19)))*(h**(-1))*(unumpy.sqrt(np.cos(i))**-1)
        return MicrolensingTimescale.to_return(f)
        #if isinstance(f,AffineScalarFunc):
         #   return np.array([f.nominal_value,f.s])
        #return np.array([[i.nominal_value,i.s] for i in f])
    
    def sigma_dv(self,zl,zs,teta_e,dteta_e,c=c,Da=None, Da_ls=None):#velocity dispersion of the stars in the lens galaxy
        """_summary_
        for sigma_dv eq 1 from https://www.aanda.org/articles/aa/abs/2008/02/aa7534-07/aa7534-07.html
        Args:
            zl (_type_): _description_
            zs (_type_): _description_
            teta_e (_type_): _description_ arcsec(")
            dteta_e (_type_): _description_ arcsec(")
            c (_type_, optional): _description_. Defaults to c.
            Da (_type_, optional): _description_. Defaults to None.
            Da_ls (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_ return velocidad en km/s
        """
        #
        teta_e = teta_e * 4.84814e-6 #to - rad
        dteta_e = dteta_e * 4.84814e-6 #to - rad
        if Da is None:
            Da = self.cosmology.angular_diameter_distance([zl]).to(u.cm).value
        if Da_ls is None:
            Da_ls = self.cosmology.angular_diameter_distance_z1z2(zl,zs).to(u.cm).value
        
        f=1e-5*0.5*c*unumpy.sqrt(unumpy.uarray(teta_e, dteta_e)*Da/(np.pi*Da_ls))
        
        return MicrolensingTimescale.to_return(f)
    def v_proyected(self,ra,dec, v0=None, dv0=None,verbose=False):
        """_summary_
        for v_proyected eq 1 from https://www.aanda.org/articles/aa/abs/2008/02/aa7534-07/aa7534-07.html
        Args:
            ra (_type_): _description_
            dec (_type_): _description_
            v0 (_type_, optional): _description_. Defaults to None.
            dv0 (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        #value and error from https://arxiv.org/pdf/1807.06205
        v0 = v0 or self.v0
        dv0 = dv0 or self.dv0
        v_north,v_east = calculate_projections(ra,dec)
        V0 = unumpy.uarray(v0, dv0)
        f=1*unumpy.sqrt((v_north*V0)**2+(v_east*V0)**2)
        return MicrolensingTimescale.to_return(f)
    
    def Ve(self,zl,zs,ra,dec,teta_e,dteta_e,v0=None,dv0=None,gamma=gamma,sigma_pec_0=sigma_pec_0 \
        ,c=c,dsigma_pec_0=dsigma_pec_0,Da=None, Da_ls=None,return_all=True,verbose=False): # separation redshift lens redshift source peculiar velocity lens peculiar velocity source
        """_summary_
        folowing equation 8 from https://iopscience.iop.org/article/10.3847/0004-637X/832/1/46
        Args:
            zl (_type_): _description_
            zs (_type_): _description_
            ra (_type_): _description_
            dec (_type_): _description_
            teta_e (_type_): _description_
            dteta_e (_type_): _description_
            v0 (_type_, optional): _description_. Defaults to None.
            dv0 (_type_, optional): _description_. Defaults to None.
            gamma (_type_, optional): _description_. Defaults to gamma.
            sigma_pec_0 (_type_, optional): _description_. Defaults to sigma_pec_0
            c (_type_, optional): _description_. Defaults to c.
            dsigma_pec_0 (_type_, optional): _description_. Defaults to dsigma_pec_0.
            Da (_type_, optional): _description_. Defaults to None.
            Da_ls (_type_, optional): _description_. Defaults to None.
            return_all (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        gamma = gamma or self.gamma
        sigma_pec_0 = sigma_pec_0 or self.sigma_pec_0
        dsigma_pec_0 = dsigma_pec_0 or self.dsigma_pec_0
        v0 = v0 or self.v0
        dv0 = dv0 or self.dv0
        if Da is None:
            Da_l,Da_s = self.cosmology.angular_diameter_distance([zl,zs]).to(u.cm).value
        else:
            Da_l,Da_s = Da
        if Da_ls is None:
            Da_ls = self.cosmology.angular_diameter_distance_z1z2(zl,zs).to(u.cm).value
        
        p_l=unumpy.uarray(*self.sigma_pec(zl,gamma=gamma,sigma_pec_0=sigma_pec_0,dsigma_pec_0=dsigma_pec_0).T)
        p_s=unumpy.uarray(*self.sigma_pec(zs,gamma=gamma,sigma_pec_0=sigma_pec_0,dsigma_pec_0=dsigma_pec_0).T)
        v_d=unumpy.uarray(*self.sigma_dv(zl,zs,teta_e,dteta_e,c=c,Da=Da_l, Da_ls=Da_ls).T)
        v_cmb = unumpy.uarray(*self.v_proyected(ra,dec,v0=v0,dv0=dv0).T)
        f=unumpy.sqrt(2*((p_l*Da_s/((1+zl)*(Da_l)))**2)+2*((p_s/(1+zs))**2)+((v_cmb*Da_ls/((1+zl)*Da_l))**2)+2*((v_d*Da_s/((1+zl)*Da_l))**2))
        if verbose:
            print(f"p_l:{p_l},p_s:{p_s},v_d:{v_d},v_cmb:{v_cmb}")
        return   [MicrolensingTimescale.to_return(i) for i in [f,v_cmb,v_d,p_s,p_l]]
        # if isinstance(f,AffineScalarFunc):
        #     return np.array([f.nominal_value,f.s])
        # elif len(f.shape)>1:
        #     return np.array([[i.nominal_value,i.s] for i in f[0].T])
        # elif len(f.shape)==1:
        #     return np.array([[i.nominal_value,i.s] for i in f])
        # else:
        #     f = f.reshape(1,)[0]
        #     return np.array([f.nominal_value,f.s])
    def time_scales(self,zl,zs,ra,dec,m,dm,zpt,lamda,teta_e,dteta_e,
                    v0=None,dv0=None,c=c,M=None,G=G,p_dM=None,i=None,
                    gamma=gamma,sigma_pec_0=None,dsigma_pec_0=None,return_all=False):
        """time scales in years
        zl,zs,m,dm,zpt,lamda,teta_e,dteta_e"""
        gamma = gamma or self.gamma
        sigma_pec_0 = sigma_pec_0 or self.sigma_pec_0
        dsigma_pec_0 = dsigma_pec_0 or self.dsigma_pec_0
        v0 = v0 or self.v0
        dv0 = dv0 or self.dv0
        p_dM = p_dM or self.p_dM
        M = M or self.M
        i = i or self.i
        Da = self.cosmology.angular_diameter_distance([zl,zs]).to(u.cm).value
        Da_ls = self.cosmology.angular_diameter_distance_z1z2(zl,zs).to(u.cm).value
        V =self.Ve(zl,zs,ra,dec,teta_e,dteta_e,v0=v0,dv0=dv0,c=c,Da=Da, Da_ls=Da_ls,\
            sigma_pec_0=sigma_pec_0,dsigma_pec_0=dsigma_pec_0)
        v = unumpy.uarray(*V[0].T*1e5)
        re = unumpy.uarray(*self.Re(zl,zs,M=M,c=c,G=G,p_dM=p_dM,Da=Da,Da_ls=Da_ls).T)
        rs= unumpy.uarray(*self.Rs(zs,m,dm,zpt,lamda,c=c,G=G,i=i,Da=Da[1]).T)
        te = 3.17098e-8 *(re/v)
        ts = 3.17098e-8 *(rs/v)
        if return_all:
            A = [np.squeeze(MicrolensingTimescale.to_return(i)) for i in [te,ts,rs,re]]+[np.squeeze(i) for i in V]
            if (A[0]).shape ==(2,):
                return np.concatenate(A)
            return  np.concatenate(A, axis=1)
        return MicrolensingTimescale.to_return(te),MicrolensingTimescale.to_return(ts)
        # if isinstance(te, AffineScalarFunc):
        #     tes = np.array([te.nominal_value,te.s]),np.array([ts.nominal_value,ts.s])
        # elif len(te.shape)>1:
        #     tes = np.array([[i.nominal_value,i.s] for i in te[0].T]),np.array([[i.nominal_value,i.s] for i in ts[0].T])
        # elif len(te.shape)==1:
        #     tes = np.array([[i.nominal_value,i.s] for i in te]),np.array([[i.nominal_value,i.s] for i in ts])
        # else:
        #     te = te.reshape(1,)[0]
        #     ts = ts.reshape(1,)[0]
        #     tes = np.array([te.nominal_value,te.s]),np.array([ts.nominal_value,ts.s])
        # if return_all:
        #     return tes,
        # return tes
    
        #return pandas zl,zs,m,dm,zpt,lambda,teta_e,dteta_e,vcmb,vcm,c,m,g,p_dm,i,p_l,ep_l,p_s,ep_s,v_d,ev_d,V,ve,rs,ers,re,ere,te,ete,ts,ets
    def get_supertabla(pandas:pd.DataFrame):
        return     
    def get_distance(self,zl,zs):
        return self.cosmology.angular_diameter_distance([zl,zs]).to(u.cm).value
    
    @staticmethod
    def to_return(f):
        if isinstance(f,AffineScalarFunc):
            f= np.array([f.nominal_value,f.s])
        elif len(f.shape)>1:
            f= np.array([[i.nominal_value,i.s] for i in f[0].T])
        elif len(f.shape)==1:
            f= np.array([[i.nominal_value,i.s] for i in f])
        else:
            f = f.reshape(1,)[0]
            f= np.array([f.nominal_value,f.s])
        return f