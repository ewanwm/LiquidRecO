import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

import typing

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from liquidreco.hit import Hit, Hit2D, Hit3D




def make_corner_plot(
        fig,
        axs, 
        hits:typing.List['Hit3D'],
        charge_cutoff=200.0,
        cmap=plt.get_cmap("coolwarm"),
        marker=".",
        plot_cbar = True,
        include_fibers = False,
        colour_override:str = None):

    c= colour_override

    if c is None:
        c = [cmap(ev.weight / charge_cutoff) for ev in hits]
    
    s = [0.1*(min(ev.weight,charge_cutoff) / charge_cutoff) for ev in hits]

    axs[0,0].scatter(
        [hit.x for hit in hits],
        [hit.z for hit in hits],
        c = c,
        s = s,
        alpha = 1.0,
        marker = marker
    )
    axs[0,0].set_ylabel("z [mm]")

    if include_fibers:
        x_lims = axs[0,0].get_xlim()
        y_lims = axs[0,0].get_ylim()

        for hit in hits:
            axs[0,0].plot(
                [hit.y_fiber_hit.fiber_x, hit.y_fiber_hit.fiber_x],
                y_lims, 
                c="k",
                lw = 0.05)
            
            axs[0,0].plot(
                x_lims, 
                [hit.y_fiber_hit.fiber_z, hit.y_fiber_hit.fiber_z],
                c="k",
                lw = 0.05)
            
    

    axs[1,0].scatter(
        [hit.x for hit in hits],
        [hit.y for hit in hits],
        c = c,
        s = s,
        alpha = 1.0,
        marker=marker
    )
    axs[1,0].set_xlabel("x [mm]")
    axs[1,0].set_ylabel("y [mm]")
    
    if include_fibers:
        x_lims = axs[1,0].get_xlim()
        y_lims = axs[1,0].get_ylim()

        for hit in hits:
            axs[1,0].plot(
                [hit.z_fiber_hit.fiber_x, hit.z_fiber_hit.fiber_x],
                y_lims, 
                c="k",
                lw = 0.05)
            
            axs[1,0].plot(
                x_lims, 
                [hit.z_fiber_hit.fiber_y, hit.z_fiber_hit.fiber_y],
                c="k",
                lw = 0.05)


    mappable = axs[1,1].scatter(
        [hit.z for hit in hits],
        [hit.y for hit in hits],
        c = c,
        s = s,
        alpha = 1.0,
        marker=marker
    )
    axs[1,1].set_xlabel("z [mm]")

    if include_fibers:
        x_lims = axs[1,1].get_xlim()
        y_lims = axs[1,1].get_ylim()

        for hit in hits:
            axs[1,1].plot(
                [hit.x_fiber_hit.fiber_z, hit.x_fiber_hit.fiber_z],
                y_lims, 
                c="k",
                lw = 0.05)
            
            axs[1,1].plot(
                x_lims, 
                [hit.x_fiber_hit.fiber_y, hit.x_fiber_hit.fiber_y],
                c="k",
                lw = 0.05)
    
    if plot_cbar:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=charge_cutoff)
        fig.colorbar(
            mappable = cm.ScalarMappable(norm, cmap=cmap), 
            ax=axs, label="N Hits", location = "right"
        )


def make_corner_plot_fiber_hits(
        fig,
        axs, 
        x_fiber_hits,
        y_fiber_hits,
        z_fiber_hits,
        charge_cutoff=100.0,
        cmap=plt.get_cmap("coolwarm"),
        colour_override:str = None,
    ):

    c = colour_override
    
    if colour_override is None:
        c = [cmap(ev.weight / charge_cutoff) for ev in y_fiber_hits]

    axs[0,0].scatter(
        [hit.x for hit in y_fiber_hits],
        [hit.z for hit in y_fiber_hits],
        s = [0.1 * min(hit.weight, charge_cutoff) / charge_cutoff for hit in y_fiber_hits], 
        c = c
    )
    axs[0,0].set_ylabel("z [mm]")
    
    if colour_override is None:
        c = [cmap(ev.weight / charge_cutoff) for ev in z_fiber_hits]

    axs[1,0].scatter(
        [hit.x for hit in z_fiber_hits],
        [hit.y for hit in z_fiber_hits],
        s = [0.1 * min(hit.weight, charge_cutoff) / charge_cutoff for hit in z_fiber_hits], 
        c = c
    )
    axs[1,0].set_xlabel("x [mm]")
    axs[1,0].set_ylabel("y [mm]")

    if colour_override is None:
        c = [cmap(ev.weight / charge_cutoff) for ev in x_fiber_hits]
    
    mappable = axs[1,1].scatter(
        [hit.z for hit in x_fiber_hits],
        [hit.y for hit in x_fiber_hits],
        s = [0.1 * min(hit.weight, charge_cutoff) / charge_cutoff for hit in x_fiber_hits], 
        c = c
    )
    axs[1,1].set_xlabel("z [mm]")
    
    #fig.colorbar(mappable[3], location="right", ax=axs, label = "N Hits")

def make_rotating_gif(fig, ax, gif_file_name):

    def animate(i):
        # Rotate the axes and update
        angle = range(0, 360 + 1)[i]
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (angle + 180) % 360 - 180

        # Cycle through a full rotation of elevation, then azimuth, roll, and all
        elev = azim = roll = 0
        azim = angle_norm
        
        # Update the axis view and title
        ax.view_init(elev, azim, roll)


    ani = animation.FuncAnimation(fig, animate, repeat=True, frames=361, interval=50) #360*4 + 1, interval=50)

    writer = animation.PillowWriter(fps=10,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    
    ani.save(gif_file_name, writer=writer)

