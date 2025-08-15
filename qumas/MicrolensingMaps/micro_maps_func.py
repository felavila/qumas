import numpy as np
from scipy.ndimage import gaussian_filter
from .maps_utils import map2d_to_stats,renormalize_mean


dict_maps = {"alpha_0.1":  "/home/felipe/work/microlensing_timescale_qso_lensing/analysis/microlensing_maps/Mapas_magnification",
             "alpha_0.2":  "/home/felipe/work/microlensing_timescale_qso_lensing/analysis/microlensing_maps/Mapas_02"}


def maps_to_stats(name,component,rs_lday,pix,mappix,num_bins=100,bins_limit=3,rs_times = [0.3,1,2],dict_maps=dict_maps):
   """_summary_

   Args:
       name (_type_): _description_
       component (_type_): _description_
       rs_lday (_type_): _description_
       pix (_type_): _description_
       mappix (_type_): _description_
       num_bins (int, optional): _description_. Defaults to 100.
       bins_limit (int, optional): _description_. Defaults to 3.
       factors (list, optional): _description_. Defaults to [0.3,1,2].
       dict_maps (_type_, optional): _description_. Defaults to dict_maps.

   Returns:
       _type_: _description_
   """
   mag_maps = {}
   for key,path in dict_maps.items():
      map_1d_path = f'{path}/{name}{component}/magmap.dat'
      map_1d = np.loadtxt(map_1d_path)
      map_2d = np.reshape(map_1d, (int(mappix), int(mappix)))
      n_factor_results = {}
      for rs_time in rs_times:
         result = map2d_to_stats(map_2d,pix,rs_lday,num_bins=num_bins,bins_limit = bins_limit,rs_times = rs_time)
         n_factor_results[str(rs_time)] = result
      mag_maps[key] = n_factor_results
   return mag_maps
      
# def map_to_stat_full(name,component,rs_lday,pix,mappix):
#    print(f'Doing {name} {component}')
#    maps_alpha_01 = "/home/felipe/work/microlensing_timescale_qso_lensing/analysis/microlensing_maps/Mapas_magnification"
#    maps_alpha_02 = "/home/felipe/work/microlensing_timescale_qso_lensing/analysis/microlensing_maps/Mapas_02"
   
#    #magnification_map_path = f'microlensing_maps/{path_mapas}/{name}{component}/magmap.dat'
#    mag_maps = {}
#    for key,pats in zip(["alpha_0.1","alpha_0.2"],[maps_alpha_01,maps_alpha_02]):
#       map_1d_path = f'{pats}/{name}{component}/magmap.dat'
#       map_1d = np.loadtxt(map_1d_path)
#       map_2d = np.reshape(map_1d, (int(mappix), int(mappix)))
#       map_2d_norm = map_2d/np.mean(map_2d)
#       mag_map_2d_norm = -2.5*np.log10(map_2d_norm)
#       S = {}
#       for factor in [0.3,1,2]:
#          mag_map_2d_norm_conv = gaussian_filter(mag_map_2d_norm, [pix*rs_lday*factor,pix*rs_lday*factor],mode='mirror')
#          num_bins = 100
#          bins = np.linspace(-3, 3, num_bins + 1)
#          counts, bin_edges = np.histogram(mag_map_2d_norm_conv.ravel(), bins=bins)
#          pmf = counts / counts.sum()
#          bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#          pdf = counts / (counts.sum() * np.diff(bin_edges))
#          expected_pmf = sum(pmf * bin_centers)
#          mean = np.sum(bin_edges * np.r_[pmf, pmf[-1]])
#          abs_mean = np.sum(abs(bin_edges) * np.r_[pmf, pmf[-1]])
         
#          #cumulative = np.cumsum(pmf)
#          # Find the median index (the first outcome where cumulative probability is at least 0.5)
#          #median = outcomes[np.where(cumulative >= 0.5)[0][0]]
         
#          var_pmf = sum(pmf * bin_centers**2) - expected_pmf**2
#          expected_pmf_abs = sum(pmf * abs(bin_centers))
#          var_pmf_abs = sum(pmf * abs(bin_centers)**2) - expected_pmf_abs**2
#          mean_pos_mag = renormalize_mean(pmf,bin_centers,bin_centers>0)
#          mean_neg_mag = renormalize_mean(pmf,bin_centers,bin_centers<0)
#          prob_mag_less_than_one = sum(pmf[bin_centers<-1])
#          prob_mag_less_than_m_032 = sum(pmf[bin_centers<-0.32])
#          prob_mag_more_than_05 = sum(pmf[bin_centers>0.5])
#          prob_mag_more_than_m_032 = sum(pmf[bin_centers>0.32])
         
#          #renormalize_mean(system_info[image]['pmf'],system_info[image]['bin_centers'],system_info[image]['bin_centers']>0)
#          small = {
#             "bin_centers": bin_centers, "pmf": pmf, "pdf": pdf,
#             "width": np.diff(bin_edges), "dx": np.diff(bin_edges)[0],
#             "expected_pmf": expected_pmf, "var_pmf": var_pmf,
#             "expected_pmf_abs": expected_pmf_abs, "var_pmf_abs": var_pmf_abs,
#             "bin_edges": bin_edges,"mag_map_2d_norm":mag_map_2d_norm,"mag_map_2d_norm_conv": mag_map_2d_norm_conv,
#             "factor":factor,'rs_lday':rs_lday,"mean_pos_mag":mean_pos_mag,"mean_neg_mag":mean_neg_mag,"prob_mag_less_than_one":prob_mag_less_than_one,"prob_mag_less_than_m_032":prob_mag_less_than_m_032,
#             "prob_mag_more_than_05":prob_mag_more_than_05,"prob_mag_more_than_m_032":prob_mag_more_than_m_032,"mean":mean}
#          S[factor] = small
#       mag_maps[key] = S
#    return mag_maps