
import numpy as np
from scipy.ndimage import gaussian_filter


def renormalize_mean(pmf,bin_centers,mask):
    """_summary_

    Args:
        pmf (_type_): _description_
        bin_centers (_type_): _description_
        mask (_type_): _description_

    Returns:
        _type_: _description_
    """
    normalized_probabilities = pmf[mask] / pmf[mask].sum()
    return np.sum(bin_centers[mask] * normalized_probabilities)

def pmf_stats(pmf,bins):
    assert pmf.shape == bins.shape, "should have same dimension the pmf and the bins"
    mean = np.sum(pmf * bins)
    mean_abs = sum(pmf * abs(bins))
    mean_positive = renormalize_mean(pmf,bins,bins>0)
    mean_negative = renormalize_mean(pmf,bins,bins<0)
    prob_mag_less_than_n032 = sum(pmf[bins<-0.32])
    prob_mag_more_than_05 = sum(pmf[bins>0.5])
    prob_mag_more_than_p032 = sum(pmf[bins>0.32])
    prob_mag_less_than_none = sum(pmf[bins<-1])
    var = sum(pmf * bins**2) - mean**2
    var_pmf_abs = sum(pmf * abs(bins)**2) - mean_abs**2
    return mean,mean_abs,mean_positive,mean_negative,prob_mag_less_than_n032,prob_mag_more_than_05,prob_mag_more_than_p032,prob_mag_less_than_none,var,var_pmf_abs
    


def map2d_to_stats(map_2d,pix,rs_lday,num_bins=100,bins_limit = 3,rs_times = 1,factor_scale_to_sigma= 2.44/1.18):
    """_summary_

    Args:
        map_2d (_type_): _description_
        pix (_type_):  lt-day/pix 
        rs_lday (_type_): _description_
        num_bins (int, optional): _description_. Defaults to 100.
        bins_limit (int, optional): _description_. Defaults to 3.
        factor (int, optional): _description_. Defaults to 1.
        factor_scale_to_sigma= when you pass from rs to sigma you have to re-scale again 2.44/1.18
    Returns:
        _type_: _description_
    """
    map_2d_norm = map_2d/np.mean(map_2d)
    mag_map_2d_norm = -2.5*np.log10(map_2d_norm)
    mag_map_2d_norm_conv = gaussian_filter(mag_map_2d_norm, [factor_scale_to_sigma*rs_lday*rs_times/pix,factor_scale_to_sigma*rs_lday*rs_times/pix],mode='mirror')
    bins = np.linspace(-bins_limit, bins_limit, num_bins + 1)
    counts, bin_edges = np.histogram(mag_map_2d_norm_conv.ravel(), bins=bins)
    pmf = counts / counts.sum()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    pmf_for_plots = np.r_[pmf, pmf[-1]]
    pdf = counts / (counts.sum() * np.diff(bin_edges))
    result = {"map_2d":map_2d,"mag_map_2d_norm":mag_map_2d_norm,"mag_map_2d_norm_conv": mag_map_2d_norm_conv,"rs_lday":rs_lday,"pix":pix,"num_bins":num_bins,"rs_times":rs_times,
            "bin_edges":bin_edges,"bin_centers":bin_centers,"counts":counts,"pmf":pmf,"pmf_for_plots":pmf_for_plots,"width": np.diff(bin_edges), "dx": np.diff(bin_edges)[0],"pdf":pdf
            }
    keys = ["mean","mean_abs","mean_positive","mean_negative","prob_mag_less_than_n032","prob_mag_more_than_05","prob_mag_more_than_p032","prob_mag_less_than_none","var","var_pmf_abs"]
    result.update({keys[i]+"_center":value for i,value in enumerate(pmf_stats(pmf,bin_centers))})
    result.update({keys[i]+"_edges":value for i,value in enumerate(pmf_stats(pmf_for_plots,bin_edges))})
    return result 
    # mean_p = np.sum(bin_edges * pmf_plos)
    # mean_pos_mag_p = renormalize_mean(pmf_plos,bin_edges,bin_edges>0)
    # mean_neg_mag_p = renormalize_mean(pmf_plos,bin_edges,bin_edges<0)
    # prob_mag_less_than_one_p = sum(pmf_plos[bin_edges<-1])
    # prob_mag_less_than_m_032_p = sum(pmf_plos[bin_edges<-0.32])
    # prob_mag_more_than_05_p = sum(pmf_plos[bin_edges>0.5])
    # prob_mag_more_than_m_032_p = sum(pmf_plos[bin_edges>0.32])
    
    
    # expected_pmf = sum(pmf * bin_centers)
    # var_pmf = sum(pmf * bin_centers**2) - expected_pmf**2
    # expected_pmf_abs = sum(pmf * abs(bin_centers))
    # var_pmf_abs = sum(pmf * abs(bin_centers)**2) - expected_pmf_abs**2
    
    # mean_pos_mag = renormalize_mean(pmf,bin_centers,bin_centers>0)
    # mean_neg_mag = renormalize_mean(pmf,bin_centers,bin_centers<0)
    # prob_mag_less_than_one = sum(pmf[bin_centers<-1])
    # prob_mag_less_than_m_032 = sum(pmf[bin_centers<-0.32])
    # prob_mag_more_than_05 = sum(pmf[bin_centers>0.5])
    # prob_mag_more_than_m_032 = sum(pmf[bin_centers>0.32])
    
    
    # result = {
    #         "bin_centers": bin_centers, "pmf": pmf, "pdf": pdf,
    #         "width": np.diff(bin_edges), "dx": np.diff(bin_edges)[0],
    #         "expected_pmf": expected_pmf, "var_pmf": var_pmf,
    #         "expected_pmf_abs": expected_pmf_abs, "var_pmf_abs": var_pmf_abs,
    #         "bin_edges": bin_edges,"mag_map_2d_norm":mag_map_2d_norm,"mag_map_2d_norm_conv": mag_map_2d_norm_conv,
    #         "factor": factor,'rs_lday':rs_lday,"mean_pos_mag":mean_pos_mag,"mean_neg_mag":mean_neg_mag,"prob_mag_less_than_one":prob_mag_less_than_one,"prob_mag_less_than_m_032":prob_mag_less_than_m_032,
    #         "prob_mag_more_than_05":prob_mag_more_than_05,"prob_mag_more_than_m_032":prob_mag_more_than_m_032,"mean_p":mean_p,"mean_pos_mag_p":mean_pos_mag_p,"mean_neg_mag_p":mean_neg_mag_p,
    #         "prob_mag_less_than_one_p":prob_mag_less_than_one_p,"prob_mag_less_than_m_032_p":prob_mag_less_than_m_032_p,"prob_mag_more_than_05_p":prob_mag_more_than_05_p,"prob_mag_more_than_m_032_p":prob_mag_more_than_m_032_p}
    return result 
