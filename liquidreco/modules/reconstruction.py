"""Module to be used to reconstruct tracks from hits.

"""

import numpy as np
from matplotlib import pyplot as plt
from liquidreco.event import Event

from linscan import LINSCAN
from hough3d.basic_hough import Hough3D
from hough3d.utils import genIcosahedron

from sklearn.cluster import DBSCAN

import matplotlib
import typing

from sklearn.neighbors import NearestNeighbors


from liquidreco.plotting import make_corner_plot, make_rotating_gif
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
            required = False, default = False, type = bool,
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
