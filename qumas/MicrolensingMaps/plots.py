from typing import List, Tuple, Optional, Dict, Any
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from matplotlib.patches import Polygon, Rectangle, ConnectionPatch


from qumas.MicrolensingMaps.micro_maps_tracks_func import _sample_profile_along_line_pixels,_sample_profile_along_line_bilinear_pixels


def map_plot(mag_map_2d, cmap='RdYlBu', vmin=-1.5, vmax=0.5,
            cbar_side="right",
            remove_labels=True,
            cbar_label=r"$-2.5 \, \log(\mu/\langle \mu \rangle)$",
             save="", **kwargs):
    
    text = kwargs.get("text")
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Plot image
    im = ax.imshow(mag_map_2d, cmap=cmap, extent=[-2, 2, -2, 2],
                vmin=vmin, vmax=vmax)

    # Create divider attached to the main axis
    divider = make_axes_locatable(ax)

    # Append colorbar axis on chosen side
    cax = divider.append_axes(cbar_side, size="6%", pad=0.35)

    # Create colorbar with extend="min"
    cbar = fig.colorbar(im, cax=cax, extend="min")

    # Tick + label placement consistent with side
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label(cbar_label, fontsize=30)
    cbar.ax.yaxis.set_label_position(cbar_side)
    cbar.ax.yaxis.set_ticks_position(cbar_side)

    # Optional annotation inside main axis
    if text:
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
                fontsize=30, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Axis labels/ticks
    if remove_labels:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel("x [pixels]", fontsize=30)
        ax.set_ylabel("y [pixels]", fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=10)

    if save:
        plt.savefig(f"{save}.jpg", bbox_inches='tight')

    #plt.show()



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


def compare_maps_plot(
    mag_map_left, mag_map_right,
    vmin=-1.5, vmax=0.5,
    label_color_bar=r"$-2.5 \, \log(\mu/\langle \mu \rangle)$",
    cmap='RdYlBu',
    plot_sep=0.05,  # <--- NEW hyperparameter for horizontal spacing
    **kwargs
):
    """
    Compare two magnification maps side by side with auto-sized colorbars,
    placed using make_axes_locatable and showing an 'under' triangle via extend='min'.

    Parameters
    ----------
    plot_sep : float, default=0.05
        Horizontal separation between the left and right plots (wspace in subplots_adjust).

    Notes
    -----
    The code assumes that the two maps are normalized, but if this is not
    the case, you can change `label_color_bar`.
    """
    text_left_plot  = kwargs.get("text_left_plot",  "Left plot")
    text_right_plot = kwargs.get("text_right_plot", "Right plot")
    name_file       = kwargs.get("name_file", None)

    # Optional tweaks for the colorbar layout
    cbar_size = kwargs.get("cbar_size", "6%")  # width of the colorbar relative to axes bbox
    cbar_pad  = kwargs.get("cbar_pad", 0.25)   # gap between axes and colorbar, in inches

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plt.subplots_adjust(wspace=plot_sep)  # <-- use hyperparameter here

    # --- Left image ---
    im1 = ax1.imshow(mag_map_left, cmap=cmap, extent=[-2, 2, -2, 2],
                     vmin=vmin, vmax=vmax)
    vmin_locked, vmax_locked = im1.get_clim()
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.text(0.05, 0.95, text_left_plot, transform=ax1.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='left',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Colorbar on the LEFT of ax1
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("left", size=cbar_size, pad=cbar_pad)
    cbar1 = fig.colorbar(im1, cax=cax1, extend="min", orientation="vertical")
    cbar1.ax.tick_params(labelsize=30)
    cbar1.set_label(label_color_bar, fontsize=30)
    cbar1.ax.yaxis.set_ticks_position("left")
    cbar1.ax.yaxis.set_label_position("left")

    # --- Right image ---
    im2 = ax2.imshow(mag_map_right, cmap=cmap, extent=[-2, 2, -2, 2],
                     vmin=vmin_locked, vmax=vmax_locked)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(0.05, 0.95, text_right_plot, transform=ax2.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='left',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # Colorbar on the RIGHT of ax2
    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size=cbar_size, pad=cbar_pad)
    cbar2 = fig.colorbar(im2, cax=cax2, extend="min", orientation="vertical")
    cbar2.ax.tick_params(labelsize=30)
    cbar2.set_label(label_color_bar, fontsize=30)
    cbar2.ax.yaxis.set_ticks_position("right")
    cbar2.ax.yaxis.set_label_position("right")

    if name_file:
        plt.savefig(f"{name_file}.pdf", bbox_inches='tight')
    plt.show()




def map_plot_with_zoom(
    mag_map_2d,
    cmap='RdYlBu',
    vmin=-1.5, vmax=0.5,
    cbar_side="right",
    remove_labels=True,
    cbar_label=r"$-2.5 \, \log(\mu/\langle \mu \rangle)$",
    save="",
    *,
    # Zoom controls
    extent=(-2, 2, -2, 2),
    zoom_box=None,
    zoom_center=None,
    zoom_size=(1.0, 1.0),
    zoom_axes_position="right",
    share_colorbar=True,
    connect_boxes=True,
    zoom_label="Zoom",
    main_label=None,
    label_fontsize=30,
    fig_size=(20, 10),
    wspace=0.08, hspace=0.1,
    interpolation="nearest"
):
    """
    Show the full map and an enlarged view of a selected box.
    Colorbars are placed with make_axes_locatable so their height
    matches the axes exactly.
    """

    if zoom_box is not None:
        (x0, x1), (y0, y1) = zoom_box
    else:
        xc, yc = zoom_center if zoom_center is not None else (0.5*(extent[0]+extent[1])+0.75, 0.5*(extent[2]+extent[3]))
        w, h = zoom_size
        x0, x1 = xc - 0.5*w, xc + 0.5*w
        y0, y1 = yc - 0.5*h, yc + 0.5*h


    if zoom_axes_position == "bottom":
        fig, (ax_main, ax_zoom) = plt.subplots(2, 1, figsize=fig_size)
        plt.subplots_adjust(hspace=hspace)
    else:
        fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=fig_size)
        plt.subplots_adjust(wspace=wspace)


    im_main = ax_main.imshow(
        mag_map_2d, cmap=cmap, extent=extent,
        vmin=vmin, vmax=vmax,
        interpolation=interpolation, origin="upper"
    )
    rect = Rectangle((x0, y0), x1-x0, y1-y0,fill=False, edgecolor="k", linewidth=2)
    ax_main.add_patch(rect)

    if remove_labels:
        ax_main.set_xticks([]); ax_main.set_yticks([])
    else:
        ax_main.set_xlabel("x [pixels]", fontsize=label_fontsize)
        ax_main.set_ylabel("y [pixels]", fontsize=label_fontsize)

    if main_label:
        ax_main.text(0.05, 0.95, main_label, transform=ax_main.transAxes,
                    fontsize=label_fontsize, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))


    im_zoom = ax_zoom.imshow(mag_map_2d, cmap=cmap, extent=extent,vmin=vmin, vmax=vmax,interpolation=interpolation, origin="upper",aspect="auto")
    
    ax_zoom.set_xlim(x0, x1); ax_zoom.set_ylim(y0, y1)

    if remove_labels:
        ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
    else:
        ax_zoom.set_xlabel("x [pixels]", fontsize=label_fontsize)
        ax_zoom.set_ylabel("y [pixels]", fontsize=label_fontsize)

    if zoom_label:
        ax_zoom.text(0.05, 0.95, zoom_label, transform=ax_zoom.transAxes,
                    fontsize=label_fontsize, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))


    if connect_boxes and zoom_axes_position == "right":
        for (cx, cy) in [(x0,y0),(x0,y1),(x1,y0),(x1,y1)]:
            con = ConnectionPatch(xyA=(0,0), coordsA=ax_zoom.transAxes,
                                xyB=(cx,cy), coordsB=ax_main.transData,
                                linewidth=1.5, color="k", alpha=0.6)
            fig.add_artist(con)

    if share_colorbar:
        div = make_axes_locatable(ax_zoom)
        cax = div.append_axes(cbar_side, size="6%", pad=0.25)
        cbar = fig.colorbar(im_zoom, cax=cax, extend="min", orientation="vertical")
        cbar.ax.tick_params(labelsize=label_fontsize)
        cbar.set_label(cbar_label, fontsize=label_fontsize)
        cbar.ax.yaxis.set_ticks_position(cbar_side)
        cbar.ax.yaxis.set_label_position(cbar_side)
    else:
        for ax, im in [(ax_main, im_main), (ax_zoom, im_zoom)]:
            div = make_axes_locatable(ax)
            cax = div.append_axes(cbar_side, size="6%", pad=0.25)
            cbar = fig.colorbar(im, cax=cax, extend="min", orientation="vertical")
            cbar.ax.tick_params(labelsize=label_fontsize)
            cbar.set_label(cbar_label, fontsize=label_fontsize)
            cbar.ax.yaxis.set_ticks_position(cbar_side)
            cbar.ax.yaxis.set_label_position(cbar_side)
            
        

    if save:
        plt.savefig(f"{save}.pdf", bbox_inches='tight')
    plt.show()










def compare_maps_pmf(
    mag_map_left, mag_map_right,
    vmin=-1.5, vmax=0.5,
    label_color_bar=r"$-2.5 \, \log(\mu/\langle \mu \rangle)$",
    bins_limit=3, num_bins=100,
    cmap="RdYlBu",
    plot_sep=0.2,
    **kwargs
):
    """
    Compare two magnification maps side by side with their Probability Mass Functions (PMF).

    Parameters
    ----------
    mag_map_left, mag_map_right : 2D arrays
        Magnification maps to compare.

    vmin, vmax : float, optional
        Color scale limits for both maps. Defaults are -1.5 and 0.5.

    label_color_bar : str, optional
        Label for the colorbars.

    bins_limit : float, optional
        Range for PMF histogram edges (in log μ). Default = 3.

    num_bins : int, optional
        Number of histogram bins for PMF calculation. Default = 100.

    cmap : str, optional
        Colormap to use for the images. Default = 'RdYlBu'.

    plot_sep : float, optional
        Horizontal separation between panels (same as wspace in subplots_adjust).

    Other Parameters
    ----------------
    text_left_plot, text_right_plot : str, optional
        Titles for each map panel.

    cbar_size, cbar_pad : str or float, optional
        Size and padding of the colorbars (used in make_axes_locatable).

    save : str, optional
        If provided, saves the figure to this path (with .pdf extension if not included).
    """
    text_left_plot  = kwargs.get("text_left_plot",  "Left plot")
    text_right_plot = kwargs.get("text_right_plot", "Right plot")
    name_file       = kwargs.get("save", None)
    cbar_size       = kwargs.get("cbar_size", "6%")
    cbar_pad        = kwargs.get("cbar_pad", 0.25)

    # --- Create figure: 3 panels of equal width ---
    fig, (ax1, ax3, ax2) = plt.subplots(
        1, 3, figsize=(30, 10),
        gridspec_kw={'width_ratios': [1, 1, 1]}
    )
    plt.subplots_adjust(wspace=plot_sep)

    # --- Left map ---
    im1 = ax1.imshow(mag_map_left, cmap=cmap, extent=[-2, 2, -2, 2],
                     vmin=vmin, vmax=vmax)
    vmin_locked, vmax_locked = im1.get_clim()
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.text(0.05, 0.95, text_left_plot, transform=ax1.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='left',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("left", size=cbar_size, pad=cbar_pad)
    cbar1 = fig.colorbar(im1, cax=cax1, extend="min", orientation="vertical")
    cbar1.ax.tick_params(labelsize=25)
    cbar1.set_label(label_color_bar, fontsize=28)
    cbar1.ax.yaxis.set_ticks_position("left")
    cbar1.ax.yaxis.set_label_position("left")

    # --- Right map ---
    im2 = ax2.imshow(mag_map_right, cmap=cmap, extent=[-2, 2, -2, 2],
                     vmin=vmin_locked, vmax=vmax_locked)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.text(0.05, 0.95, text_right_plot, transform=ax2.transAxes,
             fontsize=30, fontweight='bold', va='top', ha='left',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size=cbar_size, pad=cbar_pad)
    cbar2 = fig.colorbar(im2, cax=cax2, extend="min", orientation="vertical")
    cbar2.ax.tick_params(labelsize=25)
    cbar2.set_label(label_color_bar, fontsize=28)
    cbar2.ax.yaxis.set_ticks_position("right")
    cbar2.ax.yaxis.set_label_position("right")

    # --- Middle PMF (equal width to maps) ---
    for label, data, color in zip(
        [text_left_plot, text_right_plot],
        [mag_map_left, mag_map_right],
        ["C0", "C1"]
    ):
        bins = np.linspace(-bins_limit, bins_limit, num_bins + 1)
        counts, _ = np.histogram(data.ravel(), bins=bins)
        pmf = counts / counts.sum()
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax3.step(bin_centers, pmf, where='mid', lw=3,
                 alpha=0.9, label=f"PMF ({label})", color=color)

    ax3.legend(loc="best", fontsize=22)
    ax3.set_xlabel(r"$-2.5 \, \log(\mu/\langle \mu \rangle)$", fontsize=30)
    ax3.set_ylabel("Density (PMF)", fontsize=30)
    ax3.tick_params(axis='both', which='major', labelsize=22)
    ax3.set_xlim(-bins_limit, bins_limit)
    ax3.grid(True, alpha=0.3)

    # --- Optional save ---
    if name_file:
        #if not name_file.endswith(".pdf"):
         #   name_file += ".pdf"
        plt.savefig(name_file, bbox_inches="tight")

    plt.show()




def map_plot_with_profiles(
    mag_map_2d: np.ndarray,
    p0: Optional[Tuple[float, float]] = None,
    p1: Optional[Tuple[float, float]] = None,
    *,
    lines: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    labels: Optional[List[str]] = None,
    line_kwargs: Optional[Dict[str, Any]] = None,
    per_line_kwargs: Optional[List[Dict[str, Any]]] = None,
    profile_mode: str = "pixels",   # <- default now uses pixel-index x-axis
    cmap='RdYlBu',
    vmin=-1.5, vmax=0.5,
    cbar_side="left",
    remove_labels=True,
    cbar_label=r"$-2.5 \, \log(\mu/\langle \mu \rangle)$",
    pix_L = None,
    veff = None,
    extent=(-2, 2, -2, 2),
    lines_pix = None,
    save: str = ""
):
    """
    Plot a 2D map and multiple line profiles on the right.

    profile_mode:
      - "bilinear_pixels": bilinear sampling, exactly one sample per crossed pixel; x = pixel index
      - "pixels":          discrete Bresenham pixels (nearest); x = pixel index
    """
    if lines is None:
        if p0 is None or p1 is None:
            raise ValueError("Provide either (p0, p1) or lines=[(p0,p1), ...].")
        lines = [(p0, p1)]

    n_lines = len(lines)
    if labels is not None and len(labels) != n_lines:
        raise ValueError("labels length must match number of lines.")
    if per_line_kwargs is not None and len(per_line_kwargs) != n_lines:
        raise ValueError("per_line_kwargs length must match number of lines.")

    if line_kwargs is None:
        line_kwargs = dict(lw=4, alpha=0.95)

    fig = plt.figure(figsize=(28, 10))
    gs = GridSpec(nrows=1, ncols=2, width_ratios=[1.0, 2.0], wspace=0.01, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    ax.text(
    0.0, 1.05, fr"{mag_map_2d.shape[0]}$\times${mag_map_2d.shape[0]}pix",          # (x,y) in axes fraction
    transform=ax.transAxes,         # <-- use axes coords, not data coords
    ha='left', va='top',            # align to the right/top
    fontsize=30, color='black', fontweight='bold'
)
    ax_prof = fig.add_subplot(gs[0, 1])

    # Image
    im = ax.imshow(
        mag_map_2d, cmap=cmap, vmin=vmin, vmax=vmax,
        extent=[extent[0], extent[1], extent[2], extent[3]],
        origin="upper", aspect="auto"
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(cbar_side, size="6%", pad=0.18)
    cbar = fig.colorbar(im, cax=cax, extend="min")
    cbar.ax.tick_params(labelsize=30)
    cbar.ax.yaxis.set_label_position(cbar_side)
    cbar.ax.yaxis.set_ticks_position(cbar_side)
    cbar.ax.invert_yaxis()
    if remove_labels:
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.set_xlabel("x", fontsize=30)
        ax.set_ylabel("y", fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=18)

    # Lines & profiles
    profiles = []
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    exclude = {'#1f77b4', '#d62728'}
    filtered_cycle = [c for c in color_cycle if c.lower() not in exclude]
    T = []
    xlabel = r"t [days]"
    if not pix_L or not veff:
        xlabel = "Number of pixels"
        pix_L = 1
        veff = 1
    for idx, (q0, q1) in enumerate(lines):
        kw = dict(line_kwargs)
        if per_line_kwargs and per_line_kwargs[idx]:
            kw.update(per_line_kwargs[idx])
        if 'color' not in kw and filtered_cycle:
            kw['color'] = filtered_cycle[idx % len(filtered_cycle)]

        ax.plot([q0[0], q1[0]], [q0[1], q1[1]], **kw)
        ax.scatter([q0[0], q1[0]], [q0[1], q1[1]], s=60, color=kw.get("color", "k"), zorder=3)
        rr = None
        cc = None 
        center_pix = None 
        if profile_mode == "pixels":
            xaxis, vals, rr, cc = _sample_profile_along_line_pixels(mag_map_2d, q0, q1, extent)
            center_pix = lines_pix[idx]
        elif profile_mode == "bilinear_pixels":
            xaxis, vals, Npixels = _sample_profile_along_line_bilinear_pixels(mag_map_2d, q0, q1, extent)
        # elif profile_mode == "other":
        #     xaxis, vals, Npixels = _sample_from_index(mag_map_2d, q0, q1)
        else:
            raise ValueError("profile_mode must be 'bilinear_pixels' or 'pixels'.")
        #print(xaxis.shape)
        label = (labels[idx] if labels else f"Line {idx+1}")
        ax_prof.plot(xaxis*pix_L/veff, vals, label=label, **{k:v for k,v in kw.items() if k != 'alpha'})
        T.append(xaxis*pix_L/veff)
        # add a second x-axis at the top for pixels
        profiles.append({'p0': q0, 'p1': q1, 'x_pix': xaxis, 'values': vals, 'label': label,"rr":rr,"cc":cc,"center_pix":center_pix})


    ax_prof.set_xlabel(xlabel, fontsize=30)
    ax_prof.yaxis.set_label_position("right")
    ax_prof.yaxis.set_ticks_position("right")
    ax_prof.invert_yaxis()
    ax_prof.set_xlim(0,np.max(T))
    ax_prof.set_ylabel(cbar_label, fontsize=40)
    ax_prof.tick_params(axis='both', which='major', labelsize=30)
    ax_prof.grid(True, alpha=0.3)
    def ld_to_pix(ld):
            return ld * veff / pix_L       # inverse of what you used in plot

    def pix_to_ld(pix):
        return pix * pix_L / veff

    secax = ax_prof.secondary_xaxis(
        'top', functions=(ld_to_pix, pix_to_ld)
    )
    secax.set_xlabel("Number of pixels", fontsize=30)
    secax.set_xlim(ax_prof.get_xlim())  # keep limits consistent
    secax.tick_params(axis='both', which='major', labelsize=30)
    
    # ax_prof.legend(loc="best", fontsize=16, frameon=True)

    fig.subplots_adjust(left=0.04, right=0.985, top=0.98, bottom=0.08, wspace=0.04)
    if save:
        fig.savefig(f"{save}.pdf", dpi=200, bbox_inches="tight")

    return fig, (ax, ax_prof), profiles

def map_plot_with_profiles_pixels(
    mag_map_2d: np.ndarray,
    *,
    lines_pix: Optional[List[Tuple[Tuple[int,int], Tuple[int,int]]]] = None,
    p0: Optional[Tuple[float,float]] = None,
    p1: Optional[Tuple[float,float]] = None,
    extent=(-2,2,-2,2),
    profile_mode: str = "pixels",   # default stays pixel-index x-axis
    **kwargs
):
    """If lines_pix is given, convert to data coords then call map_plot_with_profiles."""
    H, W = mag_map_2d.shape
    xmin, xmax, ymin, ymax = extent

    if lines_pix:
        lines = []
        for (r0,c0), (r1,c1) in lines_pix:
            x0 = xmin + (c0 / (W-1)) * (xmax - xmin)
            x1 = xmin + (c1 / (W-1)) * (xmax - xmin)
            y0 = ymax - (r0 / (H-1)) * (ymax - ymin)
            y1 = ymax - (r1 / (H-1)) * (ymax - ymin)
            lines.append(((x0, y0), (x1, y1)))
        return map_plot_with_profiles(
            mag_map_2d, lines=lines, extent=extent, profile_mode=profile_mode,lines_pix=lines_pix, **kwargs
        )

    return map_plot_with_profiles(
        mag_map_2d, p0=p0, p1=p1, extent=extent, profile_mode=profile_mode, **kwargs
    )