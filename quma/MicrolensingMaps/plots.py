from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Polygon

def map_plot(mag_map_2d,cmap='RdYlBu',vmin=-1.5,vmax=0.5,cbar_side="right",remove_labels=True,cbar_label=r"$-2.5 \, \log(\mu/\langle \mu \rangle)$",save="",**kwargs):
    
    text = kwargs.get("text")
    fig, ax = plt.subplots(1, 1, figsize=(20*1, 10*1))

    im = ax.imshow(mag_map_2d,cmap=cmap, extent=[-2, 2, -2, 2], vmin=vmin, vmax=vmax)
    
    #im.cmap.reversed()
    #divider = make_axes_locatable(ax)
    #side = "right"
    #cax = divider.append_axes(side, size="5%", pad=0.2)  # Adjust size and padding
    
    #ax.set_title(titles[i], fontsize=30)
    cbar = plt.colorbar(im)
    
    #x_min, x_max, y_min, y_max = im.get_extent()
    cbar.ax.set_position([0.33, 0.15, 0.05, 0.73])
    if cbar_side=="right":
        cbar.ax.set_position([0.75, 0.15, 0.05, 0.73]) 
        
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label(cbar_label, fontsize=30,position=cbar_side)
    cbar.ax.yaxis.set_label_position(cbar_side)
    cbar.ax.yaxis.set_ticks_position(cbar_side)
        # Adjust tight_layout to leave space for the suptitle
    if text:
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))    
    #height, width = mag_map_2d.shape

    bluest_color = im.cmap(im.norm(vmin))  # Normalize and map vmin to color
    #if side=="right":
    trixy = np.array([[0, 0], [1, 0], [0.5, -0.05]])
    p = Polygon(trixy, transform=cbar.ax.transAxes, 
                        clip_on=False, edgecolor='k', linewidth=0.7, 
                        facecolor=bluest_color, zorder=4, snap=True)

    cbar.ax.add_patch(p)
    if remove_labels:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel("x [pixels]", fontsize=30)
        ax.set_ylabel("y [pixels]", fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=10)
        side = "right" if cbar_side=="left" else "left"
        ax.yaxis.set_label_position(side)
        ax.yaxis.set_ticks_position(side)
    if save:
        plt.savefig(f"{save}.jpg", bbox_inches='tight')
    plt.show()

def map_pmf_plot(mag_map,label="",label_color_bar =r"$-2.5 \, \log(\mu/\langle \mu \rangle)$", vmin=-1.5, vmax=0.5,text_right_plot="Right plot",bins_limit=3,
            num_bins=100,cmap='RdYlBu',**kwargs):
    #rf"{name} {component} $\ast$ {factor} rs $\alpha = {alpha.split('_')[1]}$ mean={mean:.3f}"
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(27, 10))
    bins = np.linspace(-bins_limit, bins_limit, num_bins + 1)
    counts, bin_edges = np.histogram(mag_map.ravel(), bins=bins)
    pmf = counts / counts.sum()
    pmf_plos = np.r_[pmf, pmf[-1]]
    mean_p = np.sum(bin_edges * pmf_plos)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    expected_pmf = sum(pmf * bin_centers)
    
    im1 = ax1.step(bin_edges,pmf_plos, alpha=0.7, linestyle="-", label=label,c="C0")
    ax1.axvline(expected_pmf,c="k",ls="--",label="Expected PMF")
    ax1.axvline(mean_p,c="r",ls="--",label="Mean")
    
    ax1.legend(loc="best", fontsize=12)
    ax1.legend(loc="best", fontsize=12)
    ax1.set_xlabel(r"$-2.5 \, \log(\mu/\langle \mu \rangle)$", fontsize=30)
    ax1.set_ylabel('Density(PMF)', fontsize=30)
    
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.set_xlim(-bins_limit,bins_limit)
    
    im2 = ax2.imshow(mag_map, cmap=cmap,extent=[-2, 2, -2, 2], vmin=vmin, vmax=vmax)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(0.05, 0.95, text_right_plot, transform=ax2.transAxes, fontsize=30, fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    # Create a colorbar for the mirrored plot (positioned on the left side)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=30)
    cbar2.ax.set_position([ax2.get_position().x0-0.1*ax2.get_position().x0, 0.15, 0.02,  0.73]) 
    cbar2.ax.yaxis.set_ticks_position('left')
    cbar2.ax.yaxis.set_label_position('left')
    cbar2.set_label(label_color_bar, fontsize=30, labelpad=20)
    trixy = np.array([[0, 0], [1, 0], [0.5, -0.05]])
    bluest_color = im2.cmap(im2.norm(vmin))
    patch2 = Polygon(trixy, transform=cbar2.ax.transAxes, clip_on=False,edgecolor='k', linewidth=0.7, facecolor=bluest_color,zorder=4, snap=True)
    cbar2.ax.add_patch(patch2)
    plt.show()

def compare_maps_plot(mag_map_left,mag_map_right
                , vmin=-1.5, vmax=0.5,label_color_bar =r"$-2.5 \, \log(\mu/\langle \mu \rangle)$",cmap='RdYlBu',**kwargs):
    """The code assume that the two maps are normalize  but if this is not the case you can change the label_color_bar
    Args:
        mag_map_left (_type_): _description_
        mag_map_right (_type_): _description_
        text_left_plot (str, optional): _description_. Defaults to "Left plot".
        text_right_plot (str, optional): _description_. Defaults to "Right plot".
        vmin (float, optional): _description_. Defaults to -1.5.
        vmax (float, optional): _description_. Defaults to 0.5.
        label_color_bar (regexp, optional): _description_. Defaults to r"$-2.5 \, \log(\mu/\langle \mu \rangle)$".
        RdYlBu_r or RdYlBu
    """
    
    
    text_left_plot = kwargs.get("text_left_plot", "Left plot")
    text_right_plot = kwargs.get("text_right_plot","Right plot")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(27, 10))

    im1 = ax1.imshow(mag_map_left, cmap=cmap, 
                    extent=[-2, 2, -2, 2], vmin=vmin, vmax=vmax)
    vmin,vmax = im1.get_clim()
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(0.05, 0.95,text_left_plot, transform=ax1.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
    
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=30)
    cbar1.ax.set_position([ax1.get_position().x0-0.1*ax1.get_position().x0, 0.15, 0.02,  0.73])
    cbar1.ax.yaxis.set_ticks_position('left')
    cbar1.ax.yaxis.set_label_position('left')
    cbar1.set_label(label_color_bar, fontsize=30, labelpad=20)
    
    # Add a triangle patch to the colorbar
    bluest_color = im1.cmap(im1.norm(vmin))
    trixy = np.array([[0, 0], [1, 0], [0.5, -0.05]])
    patch1 = Polygon(trixy, transform=cbar1.ax.transAxes, clip_on=False,
                     edgecolor='k', linewidth=0.7, facecolor=bluest_color,
                     zorder=4, snap=True)
    cbar1.ax.add_patch(patch1)
    im2 = ax2.imshow(mag_map_right, cmap=cmap, 
                     extent=[-2, 2, -2, 2], vmin=vmin, vmax=vmax)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(0.05, 0.95, text_right_plot, transform=ax2.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=30)
    cbar2.ax.set_position([ax2.get_position().x1+0.005*ax2.get_position().x1, 0.15, 0.05, 0.73]) 
    cbar2.ax.yaxis.set_ticks_position('right')
    cbar2.ax.yaxis.set_label_position('right')
    cbar2.set_label(label_color_bar, fontsize=30, labelpad=20)
    
    # Add a similar patch to the second colorbar
    patch2 = Polygon(trixy, transform=cbar2.ax.transAxes, clip_on=False,
                      edgecolor='k', linewidth=0.7, facecolor=bluest_color,
                      zorder=4, snap=True)
    cbar2.ax.add_patch(patch2)
    plt.show()
    
    
def compare_maps_pmf(mag_map_left,mag_map_right
                , vmin=-1.5, vmax=0.5,label_color_bar =r"$-2.5 \, \log(\mu/\langle \mu \rangle)$",bins_limit=3,num_bins=100,cmap="RdYlBu",**kwargs):
    """The code assume that the two maps are normalize  but if this is not the case you can change the label_color_bar
    Args:
        mag_map_left (_type_): _description_
        mag_map_right (_type_): _description_
        text_left_plot (str, optional): _description_. Defaults to "Left plot".
        text_right_plot (str, optional): _description_. Defaults to "Right plot".
        vmin (float, optional): _description_. Defaults to -1.5.
        vmax (float, optional): _description_. Defaults to 0.5.
        label_color_bar (regexp, optional): _description_. Defaults to r"$-2.5 \, \log(\mu/\langle \mu \rangle)$".
        'RdYlBu_r' or 'RdYlBu_r'
    """
    #, text_left_plot = "Left plot", text_right_plot = "Right plot"
    text_left_plot = kwargs.get("text_left_plot", "Left plot")
    text_right_plot = kwargs.get("text_right_plot","Right plot")
    fig, (ax1,ax3,ax2) = plt.subplots(1, 3, figsize=(47, 10))

    im1 = ax1.imshow(mag_map_left, cmap=cmap, 
                    extent=[-2, 2, -2, 2], vmin=vmin, vmax=vmax)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(0.05, 0.95,text_left_plot, transform=ax1.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    
  
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=20)
    cbar1.ax.set_position([ax1.get_position().x0-0.1*ax1.get_position().x0, 0.15, 0.02,  0.73])
    cbar1.ax.yaxis.set_ticks_position('left')
    cbar1.ax.yaxis.set_label_position('left')
    cbar1.set_label(label_color_bar, fontsize=30, labelpad=20)
    
    # Add a triangle patch to the colorbar
    bluest_color = im1.cmap(im1.norm(vmin))
    trixy = np.array([[0, 0], [1, 0], [0.5, -0.05]])
    patch1 = Polygon(trixy, transform=cbar1.ax.transAxes, clip_on=False,
                     edgecolor='k', linewidth=0.7, facecolor=bluest_color,
                     zorder=4, snap=True)
    cbar1.ax.add_patch(patch1)
    # ---------------------------
    # Right: Mirrored version
    # ---------------------------
    # Create the horizontally mirrored data using np.fliplr
    #mag_map_2d_norm_mirror = np.fliplr(mag_map_2d_norm)
    im2 = ax2.imshow(mag_map_right, cmap=cmap, 
                     extent=[-2, 2, -2, 2], vmin=vmin, vmax=vmax)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(0.05, 0.95, text_right_plot, transform=ax2.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='left',bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))
    # Create a colorbar for the mirrored plot (positioned on the left side)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=20)
    cbar2.ax.set_position([ax2.get_position().x1+0.005*ax2.get_position().x1, 0.15, 0.05, 0.73]) 
    cbar2.ax.yaxis.set_ticks_position('right')
    cbar2.ax.yaxis.set_label_position('right')
    cbar2.set_label(label_color_bar, fontsize=30, labelpad=20)
    
    # Add a similar patch to the second colorbar
    patch2 = Polygon(trixy, transform=cbar2.ax.transAxes, clip_on=False,
                      edgecolor='k', linewidth=0.7, facecolor=bluest_color,
                      zorder=4, snap=True)
    cbar2.ax.add_patch(patch2)
    colors_a = {text_left_plot:"C0",text_right_plot:"C1"}
    colors_b = {text_left_plot:"k",text_right_plot:"r"}
    for key,map in {text_left_plot:mag_map_left,text_right_plot:mag_map_right}.items():
        bins = np.linspace(-bins_limit, bins_limit, num_bins + 1)
        counts, bin_edges = np.histogram(map.ravel(), bins=bins)
        pmf = counts / counts.sum()
        pmf_plos = np.r_[pmf, pmf[-1]]
        mean_p = np.sum(bin_edges * pmf_plos)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        expected_pmf = sum(pmf * bin_centers)
        ax3.step(bin_edges,np.r_[pmf, pmf[-1]], alpha=0.7, linestyle="-", label=f"PMF ({key})")
        #ax3.axvline(expected_pmf,ls="--",label=f"Expected pmf {key}",alpha=0.7,c=colors_a[key])
        #ax3.axvline(mean_p,ls="-",label=f"mean_p {key}",alpha=0.7,c=colors_b[key])
    
    ax3.legend(loc="best", fontsize=21)
    ax3.set_xlabel(r"$-2.5 \, \log(\mu/\langle \mu \rangle)$", fontsize=30)
    ax3.set_ylabel('Density(PMF)', fontsize=30)
    ax3.tick_params(axis='both', which='major', labelsize=20)
    ax3.set_xlim(-bins_limit, bins_limit)
    
    plt.show()
    
# def plot_mag_map_pmf(mag_map_dic,key_map='mag_map_2d_norm_conv', title="", save="", cbar_label="", show=False, text=None):
#     image = ""  # Image name (for labeling)
    
#     # Define GridSpec with custom subplot sizes
#     fig = plt.figure(figsize=(30, 15))
#     if save:
#         plt.ioff()
#     gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])  # Add space for text
#     # Set general title for the figure
#     fig.suptitle(f"{title}", fontsize=25)
    
#     # First plot (n == 0): Image with colorbar
#     ax0 = fig.add_subplot(gs[:, 0])  # Span both rows for larger size
#     im = ax0.imshow(mag_map_dic[key_map], cmap='RdYlBu')
#     divider = make_axes_locatable(ax0)
#     cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and padding
#     ax0.set_xlabel("x [pixels]", fontsize=20)
#     ax0.set_ylabel("y [pixels]", fontsize=20)
#     ax0.tick_params(axis='both', which='major', labelsize=20)
#     cbar = plt.colorbar(im, cax=cax)
#     cbar.ax.tick_params(labelsize=20)
#     cbar.set_label(cbar_label, fontsize=20)
    
#     # Second plot (n == 1): Histogram with PMF and PDF
#     ax1 = fig.add_subplot(gs[1, 1])  # Small, symmetric plot
    
#     ax1.step(
#         mag_map_dic["bin_edges"], np.r_[mag_map_dic["pmf"], mag_map_dic["pmf"][-1]],
#         alpha=0.5, color='blue') 
#         #label=rf"$|mean| = {mag_map_dic['expected_pmf_abs']:.2f}, \quad |std| = {np.sqrt(mag_map_dic['var_pmf_abs']):.2f}$"
#     #)
#     ax1.set_xlabel(r"$-2.5 \, \log(\mu/\langle \mu \rangle)$", fontsize=16)
#     ax1.set_ylabel('Density (PMF)', fontsize=16)
#     ax1.set_xlim(-3, 3)
#     ax1.legend(loc="best", fontsize=14)
#     ax1.tick_params(axis='both', which='major', labelsize=16)
    
#     # Move y-axis of the second plot to the right
#     ax1.yaxis.set_label_position("right")
#     ax1.yaxis.tick_right()
    
#     # Text area below the small plot (if text is provided)
#     if text:
#         ax_text = fig.add_subplot(gs[1, 1])  # Use the bottom-right quadrant
#         ax_text.axis('off')  # Turn off the axes
#         ax_text.text(0.5, 0.5, text, fontsize=16, ha='center', va='center', wrap=True)
    
#     # Adjust layout
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
    
#     if save:
#         plt.savefig(f"{save}.png")
#         plt.close()
#     else:
#         plt.show()
    
#     return



        
        
# def mag_map_plot(map,main_title=None,titles=None,vmax=None,vmin=None,color_bar_label =r"$-2.5 \, \log(\mu/\langle \mu \rangle)$",save=""):
#     #-0.32 is a value that Evencio askme for 
#     #mag_map_2d_norm_conv
#     if isinstance(map,list):
#         map = np.stack(map)
#         n_plots=map.shape[0]
#     else:
#         map = np.stack([map])
#         n_plots=map.shape[0]
#     titles = titles or [""]*n_plots
#     main_title = main_title or ""
#     fig, axes = plt.subplots(1, n_plots, figsize=(20*n_plots, 10*n_plots))
#     axes = np.atleast_1d(axes)
#     axes = axes.flat
#     for i,ax in enumerate(axes):
#         im = ax.imshow(map[i],vmax=vmax,cmap='RdYlBu')
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and padding
#         ax.set_xlabel("x [pixels]", fontsize=30)
#         ax.set_ylabel("y [pixels]", fontsize=30)
#         ax.tick_params(axis='both', which='major', labelsize=30)
#         ax.set_title(titles[i], fontsize=30)
#         cbar = plt.colorbar(im, cax=cax)
#         cbar.ax.tick_params(labelsize=30)
#         cbar.set_label(color_bar_label, fontsize=30)
    
#     # Adjust tight_layout to leave space for the suptitle
#     plt.tight_layout()  # Reserve top 5% for the suptitle
#     #fig.suptitle(main_title, fontsize=25,y=0.7)
#     if save:
#         plt.savefig(f"plot_{save}.jpg")
#         plt.close()
#     else:
#         plt.show()
        


 
    


    
    
# def hist_n_image(dic,name,save="",images: list=None):
#     #name = "2M1310-1714"
#     title = name 
#     system = dic[name]
#     if not images:
#         images = system.keys()
#     fig, axes = plt.subplots(1, 1, figsize=(20, 10))
#     #plt.ioff()
#         # Plot the three bar series
#     yvalue = "pmf"
#     color = ["C0","C1","C7","C3","C4","C5"]
#     [axes.step(system[image]["bin_edges"],np.r_[system[image][yvalue], system[image][yvalue][-1]], alpha=0.5,color=color[n], label=rf"${image} \quad |mean| = {system[image]['expected_pmf_abs']:.2f}, \quad |std| = {np.sqrt(system[image]['var_pmf_abs']):.2f}$") for n,image in enumerate(images)]
#     #[ax.text(ax.get_xlim()[0]+ax.get_xlim()[1]*0.2, ax.get_ylim()[0]+ax.get_ylim()[1]*(0.8-n/10), rf"$m_1(\mu) = {images_[title][image]['mean_magnification']:.2f}, \quad m_2(\sigma^2) = {images_[title][image]['variance']:.2f}$", color='k', fontsize=10)  for n,image in enumerate(images_[title].keys())] #, bbox=dict(facecolor='white', alpha=0.7)]
#     axes.set_title(f'{title}({results[results.name==title].mass_models.values[0]})', fontsize=14)
#     # Optional: set x-label, y-label for each subplot
#     axes.set_xlabel(r"$-2.5 \, \log(\mu/\langle \mu \rangle)$", fontsize=12)
#     axes.set_ylabel(f'Density({yvalue})', fontsize=12)

#     # Add legend
#     axes.legend(loc="best", fontsize=12)

#     # Make sure tick labels have the same size
#     axes.tick_params(axis='both', which='major', labelsize=20)
#     axes.set_xlim(-3,3)
#     if save:
#         plt.savefig(f"plot_{name}.jpg")
#         plt.close()
#     else:
#         plt.show()


# def plot_mag_map(array, save="", show=False,cbar_label="",title =""):
#     fig, ax = plt.subplots(figsize=(20, 20))
#     im = ax.imshow(array,cmap='RdYlBu')
#     # Create a divider for the axes to append the colorbar
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and padding here
#     ax.set_xlabel("x [pixels]", fontsize=20)
#     ax.set_ylabel("y [pixels]", fontsize=20)
#     ax.tick_params(axis='both', which='major', labelsize=20)
#     ax.set_title(title, fontsize=30)
#     # Add the color bar
#     # Add the color bar with fontsize
#     cbar = plt.colorbar(im, cax=cax)
#     cbar.ax.tick_params(labelsize=20)
#     if cbar_label:
#         cbar.set_label(cbar_label, fontsize=20)
#     plt.tight_layout()
#     if save:
#         plt.savefig(f"{save}.png")
#     if show:
#         plt.show()
#     plt.close()


# Function to plot magnification map and PMF