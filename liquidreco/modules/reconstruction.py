"""Module to be used to reconstruct tracks from hits.

"""

import numpy as np
from matplotlib import pyplot as plt
from liquidreco.event import Event

from linscan import LINSCAN
from hough3d.basic_hough import Hough3D

from sklearn.cluster import DBSCAN

import matplotlib
import typing

from scipy.sparse import csr_array
from scipy.sparse import csgraph

from sklearn.neighbors import NearestNeighbors


from liquidreco.plotting import make_corner_plot, make_rotating_gif, make_corner_plot_fiber_hits
from liquidreco.modules.module_base import ModuleBase

class HoughTransform(ModuleBase):
    """ Performs simple Hough line transform
    """

    def __init__(self):
        
        super().__init__()

        self.requirements = ["3d_hits"]

    def _setup_cli_options(self, parser):

        parser.add_argument(
            "--min-charge-thresh", 
            help="Minimum charge that a hit must have to be included in the track fitting", 
            required = False, default = 80.0, type = float,
        )
        parser.add_argument(
            "--max-plot-charge", 
            help="Maximum charge for plots", 
            required = False, default = 100.0, type = float,
        )
        parser.add_argument(
            "--make-gifs", 
            help="Whether to make animated gifs of events (slooow)", 
            action='store_true'
        )
        
    def _initialise(self) -> None:
        
        self.min_charge_thresh = self.args.min_charge_thresh
        self.max_plot_charge = self.args.max_plot_charge
        self.make_gifs = self.args.make_gifs

        self.hough_finder = Hough3D(
            neighbour_dist=20.0, min_points_per_line=5, lattice_step_size=10.0
        )

        self._pdf = matplotlib.backends.backend_pdf.PdfPages("Hough-event-examples.pdf")
        self._corner_pdf = matplotlib.backends.backend_pdf.PdfPages("Hough-event-examples-corner.pdf")

        self.clusterer = DBSCAN(40.0)

    def _process(self, event:Event):
        """ Perform Hough transform on an event and save the result to a given file

        :param event: Object describing the hits in an event
        :type event: Event
        """

        hits_3d = event["3d_hits"]

        fig = plt.figure(dpi=400)
        ax = fig.add_subplot(projection='3d')

        data = np.array(
            [[hit.x, hit.y, hit.z] for hit in hits_3d if hit.weight > self.min_charge_thresh]
        )

        if data.shape[0] == 0:
            # maybe there are no hits above threshold
            
            plt.tight_layout()

            self._pdf.savefig(fig)
            fig.clf()

            return

        cluster_ids = self.clusterer.fit_predict(data)

        cmap = plt.get_cmap("coolwarm")
        c = [cmap(ev.weight / self.max_plot_charge) for ev in hits_3d]
        s = [0.1*(min(ev.weight, self.max_plot_charge) / self.max_plot_charge) for ev in hits_3d]
        ax.scatter(
            [hit.x for hit in hits_3d],
            [hit.y for hit in hits_3d],
            [hit.z for hit in hits_3d],
            s = 0.1,
            c = c,
            alpha = 0.5
        )

        corner_fig, corner_axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(5, 5))
        corner_fig.suptitle("Hough Transform Tracks")
        make_corner_plot(corner_fig, corner_axs, hits_3d, charge_cutoff=self.min_charge_thresh, plot_cbar=False)

        for cluster_id in np.unique(cluster_ids):
            if cluster_id == -1:
                # -1 is id for "noise"
                continue

            linePoints = self.hough_finder(data[cluster_ids == cluster_id, :])
        
            for i in range(linePoints.shape[0]):
                ax.plot(linePoints[i, :, 0], linePoints[i, :, 1], linePoints[i, :, 2])

                corner_axs[0,0].plot(linePoints[i, :, 0], linePoints[i, :, 2])
                corner_axs[1,0].plot(linePoints[i, :, 0], linePoints[i, :, 1])
                corner_axs[1,1].plot(linePoints[i, :, 2], linePoints[i, :, 1])

        plt.tight_layout()

        self._pdf.savefig(fig)
        self._corner_pdf.savefig(corner_fig)

        if self.make_gifs:
            make_rotating_gif(fig, ax, f"Hough_3D.gif")

        plt.close(fig)
    
    def _finalise(self):
        self._pdf.close()
        self._corner_pdf.close()
    
class MinimumSpanningTree2D(ModuleBase):

    def __init__(self):
        super().__init__()

        self.requirements = ["x_fiber_hits", "y_fiber_hits", "z_fiber_hits"]

    def _initialise(self):
    
        self._make_plots = self.args.make_plots

        self._pdf = None
        if self._make_plots:
            self._pdf = matplotlib.backends.backend_pdf.PdfPages(self.args.plot_file_name)
    
    def _finalise(self):
        
        if self._pdf is not None:
            self._pdf.close()

    def _setup_cli_options(self, parser):
        
        parser.add_argument(
            "--make-plots", 
            help="Whether to make basic plots", 
            action='store_true'
        )
        parser.add_argument(
            "--plot-file-name", 
            help="Where to put the plots if --make-plots is specified", 
            type=str, default="MST-examples.pdf", required=False
        )

    def _process(self, event):

        x_fiber_hits = event["x_fiber_hits"]
        y_fiber_hits = event["y_fiber_hits"]
        z_fiber_hits = event["z_fiber_hits"]

        fig, axs = None, None
        if self._make_plots:
            fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(5, 5))
            fig.suptitle("Minimal Spanning Tree")

        for fiber_hits, u_name, v_name, ax_ids in zip([
            x_fiber_hits,
            y_fiber_hits,
            z_fiber_hits
        ],
        ["z", "x", "x"],
        ["y", "z", "y"],
        [[1,1], [0,0], [1,0]]
        ):
            
            n_hits = len(fiber_hits)

            graph_array = csr_array((n_hits, n_hits), dtype=float)

            for i1, hit1 in enumerate(fiber_hits):
                for i2, hit2 in enumerate(fiber_hits):

                    pos1 = np.array(hit1.pos)
                    pos2 = np.array(hit2.pos)

                    pos1[pos1 == None] = 0.0
                    pos2[pos2 == None] = 0.0

                    dir1 = np.array(hit1.dir)
                    dir2 = np.array(hit2.dir)

                    dir1[dir1 == None] = 0.0
                    dir2[dir2 == None] = 0.0

                    if np.linalg.norm(pos1 - pos2) > 25.0:
                        continue

                    else:
                        dot = np.abs(np.dot(dir1, dir2))

                        graph_array[i1, i2] = dot + np.abs(np.dot(dir1, pos2 - pos1)) / 2.0 + np.abs(np.dot(dir2, pos2 - pos1)) / 2.0 + np.linalg.norm(pos1 - pos2) / 25.0

            min_spanning_tree = csgraph.minimum_spanning_tree(graph_array)
        
            if self._make_plots:
            
                mst_array = min_spanning_tree.toarray()

                for i1, hit1 in enumerate(fiber_hits):
                    for i2, hit2 in enumerate(fiber_hits):

                        if mst_array[i1, i2] != 0.0:

                            axs[ax_ids[0], ax_ids[1]].plot(
                                [getattr(hit1, u_name), getattr(hit2, u_name)],
                                [getattr(hit1, v_name), getattr(hit2, v_name)],
                                c = "r", linewidth = 1.0
                            )

        if self._make_plots:
        
            make_corner_plot_fiber_hits(
                fig,
                axs, 
                x_fiber_hits,
                y_fiber_hits,
                z_fiber_hits,
                plot_directions=True
            )

            self._pdf.savefig(fig)
            plt.close(fig)

class LocalMeanDBSCAN(ModuleBase):

    def __init__(
            self,
            mean_ball_radius = 25.0, ## 2 fibers in each direction
            mean_iterations = 1, 
            dbscan_eps = 9.0,
            min_charge_thresh = 0.0
    ):
        
        raise NotImplementedError("This thing is BUSTED")
        # TODO: Fix this shit
        
        super().__init__()

        ## the algorithm used to find nearest neighbours
        self.neighbour_algo_ = NearestNeighbors(radius=mean_ball_radius)
        
        ## the dbscan algorithm to use for clustering distributions
        self._clusterer = LINSCAN(n_dims=3, ecc_pts = 10, eps = dbscan_eps, dbscan_eps=1.0, min_samples=25) #DBSCAN(dbscan_eps, min_samples=dbscan_min_samples)

        self._mean_iterations = mean_iterations
        self._min_charge_thresh = min_charge_thresh

        self._pdf = matplotlib.backends.backend_pdf.PdfPages("LocalMeanDBSCAN-reconstructed-event-examples.pdf")

        self.requirements = ["3d_hits"]

    def _process(self, event:Event):

        hits_3d = event["3d_hits"]

        weights = np.array(
            [hit.weight for hit in hits_3d if hit.weight > self._min_charge_thresh]
        )

        data = np.array(
            [[hit.x, hit.y, hit.z] for hit in hits_3d if hit.weight > self._min_charge_thresh]
        )

        for _ in range(self._mean_iterations):

            distances, indices = self.neighbour_algo_.fit(data).radius_neighbors(data)
            
            for point_id in range(data.shape[0]):

                neighbourhood_weights = weights[indices[point_id]]

                neighbourhood_points = data[indices[point_id]]

                data[point_id, :] = np.average(neighbourhood_points, weights = neighbourhood_weights, axis = 0)


        cluster_ids = self._clusterer.fit_predict(data) #, weights)

        fig = plt.figure()
        plt.scatter(data[:,0], data[:,2], s = 0.01 * weights, c = cluster_ids)
        self._pdf.savefig(fig)
        plt.close(fig)
        
    def _finalise(self):
        self._pdf.close()
