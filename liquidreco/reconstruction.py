import numpy as np
from matplotlib import pyplot as plt
from liquidreco.hit import Event

from linscan import LINSCAN
from hough3d.basic_hough import Hough3D
from hough3d.utils import genIcosahedron

from sklearn.cluster import DBSCAN
import tqdm

import matplotlib
from liquidreco.plotting import make_corner_plot, make_rotating_gif
import typing

from sklearn.neighbors import NearestNeighbors

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.linalg import eig

class ReconstructionAlgorithm:

    def __init__(
            self,
            events:list[Event]
    ):
        """_summary_
        """

        self.events = events

    def process_events(self):

        for event_id, event in tqdm(enumerate (self.events)):
            
            self._process(event)

        if hasattr(self, "_finalise") and callable(getattr(self, "_finalise")):
            self._finalise()

    def _process(self, event:Event):
        raise NotImplementedError("Shouldn't be directly using the base RaconstructionAlgorithm class!\nPlease implement your own derived algorithm.")

    def finalise(self):
        if hasattr(self, "_finalise") and callable(getattr(self, "_finalise")):
            self._finalise()


class HoughTransform(ReconstructionAlgorithm):
    """ Performs simple Hough line transform
    """

    def __init__(
            self, 
            events:list[Event],
            min_charge_thresh = 80.0,
            max_plot_charge = 100.0,
            make_gifs = False
        ):

            self.min_charge_thresh = min_charge_thresh
            self.max_plot_charge = max_plot_charge
            self.make_gifs = make_gifs

            self.hough_finder = Hough3D(
                neighbour_dist=20.0, min_points_per_line=5, lattice_step_size=10.0
            )

            super().__init__(events)

            self._pdf = matplotlib.backends.backend_pdf.PdfPages("Hough-event-examples.pdf")
            self._corner_pdf = matplotlib.backends.backend_pdf.PdfPages("Hough-event-examples-corner.pdf")

            self.clusterer = DBSCAN(40.0)

    def _process(self, event:Event):
        """ Perform Hough transform on an event and save the result to a given file

        :param event: Object describing the hits in an event
        :type event: Event
        """

        hits_3d = event.hits_3d

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

        #self.clusterer.fit(data)

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

        #ax.scatter(data[:, 0], data[:, 1], data[:, 2], s = 0.1, c = cluster_ids, cmap=plt.get_cmap("rainbow"))

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

        fig.clf()
    
    def _finalise(self):
        self._pdf.close()
        self._corner_pdf.close()
    
class LocalMeanDBSCAN(ReconstructionAlgorithm):

    def __init__(
            self,
            events:list[Event],
            mean_ball_radius = 25.0, ## 2 fibers in each direction
            mean_iterations = 1, 
            dbscan_eps = 9.0,
            dbscan_min_samples = 25,
            min_charge_thresh = 0.0
    ):
        
        ## the algorithm used to find nearest neighbours
        self.neighbour_algo_ = NearestNeighbors(radius=mean_ball_radius)
        
        ## the dbscan algorithm to use for clustering distributions
        self._clusterer = LINSCAN(n_dims=3, ecc_pts = 10, eps = dbscan_eps, dbscan_eps=1.0, min_samples=25) #DBSCAN(dbscan_eps, min_samples=dbscan_min_samples)

        self._mean_iterations = mean_iterations
        self._min_charge_thresh = min_charge_thresh

        self._pdf = matplotlib.backends.backend_pdf.PdfPages("LocalMeanDBSCAN-reconstructed-event-examples.pdf")

    def _process(self, event:Event):

        hits_3d = event.hits_3d

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
        
    def _finalise(self):
        self._pdf.close()



class HesseRidgeDetection(ReconstructionAlgorithm):
    """ Performs simple Hough line transform
    """

    def __init__(
            self, 
            events:list[Event],
            x_bounds:typing.Tuple[float], 
            y_bounds:typing.Tuple[float], 
            z_bounds:typing.Tuple[float],
            plot_orthogonal_dirs = False
        ):

        super().__init__(events)

        ## set up binning of the detector
        self._x_bins = np.arange(start=x_bounds[0], stop=x_bounds[1], step=5.0)
        self._y_bins = np.arange(start=y_bounds[0], stop=y_bounds[1], step=5.0)
        self._z_bins = np.arange(start=z_bounds[0], stop=z_bounds[1], step=5.0)

        self._plot_orthogonal_dirs = plot_orthogonal_dirs

        self._pdf = matplotlib.backends.backend_pdf.PdfPages("Hesse-event-examples.pdf")
        self._hesse_reco_pdf = matplotlib.backends.backend_pdf.PdfPages("Hesse-reconstructed-event-examples.pdf")

    def _process(self, event:Event):
        """ Perform Hough transform on an event and save the result to a given file

        :param event: Object describing the hits in an event
        :type event: Event
        """

        hits_3d = event.hits_3d
        
        hist, _, _ = np.histogram2d(
            [hit.x for hit in hits_3d], [hit.y for hit in hits_3d],
            bins = (self._x_bins, self._y_bins), weights=[hit.weight for hit in hits_3d]
        )

        hist = np.clip(hist, 0.0, 400.0)

        hist = np.astype(hist, np.uint8)
        
        ## Hessian eigenvalues
        hxx, hxy, hyy = hessian_matrix(hist, 0.0, order="xy", use_gaussian_derivatives=False)
        
        hess_eigenvals = np.ndarray((2, *hxx.shape))
        hess_eigenvecs = np.ndarray((2, 2, *hxx.shape))

        print(hxx.shape)
        for dim0 in range(0, hxx.shape[-2]):
            for dim1 in range(0, hxx.shape[-1]):

                if(hist[dim0, dim1] == 0.0):
                    hess_eigenvals[:, dim0, dim1] = 0.0
                    hess_eigenvecs[:, :, dim0, dim1] = 0.0
                
                else:
                    evals, evecs = eig(
                        np.array(
                            [[hxx[dim0, dim1], hxy[dim0, dim1]],
                                [hxy[dim0, dim1], hyy[dim0, dim1]]]
                        )
                    )

                    hess_eigenvals[:, dim0, dim1] = evals[:]
                    hess_eigenvecs[:, :, dim0, dim1] = evecs[:, :]


        fig, ax = plt.subplots(1, 5, figsize=(50, 10))
        ax[0].imshow(hist, cmap=plt.get_cmap("coolwarm"))
        ax[0].set_title("Original Event")

        ax[1].imshow(hxx, cmap=plt.get_cmap("gray"))
        ax[1].set_title("H_xx")
        ax[2].imshow(hxy, cmap=plt.get_cmap("gray"))
        ax[2].set_title("H_xy")
        ax[3].imshow(hyy, cmap=plt.get_cmap("gray"))
        ax[3].set_title("H_yy")

        ridgeness = np.clip(-np.min(hess_eigenvals, axis=0), 0.0, 99999.9)
        ridgeness /= np.max(ridgeness)
        ax[4].imshow(ridgeness, cmap=plt.get_cmap("gray"))
        ax[4].set_title("Hessian Filter")

        self._pdf.savefig(fig)
        plt.clf()

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(ridgeness, cmap=plt.get_cmap("gray"))
        plt.colorbar()
        plt.title("Hessian Filter")

        for dim0 in range(0, hess_eigenvecs.shape[-1]):
            for dim1 in range(0, hess_eigenvecs.shape[-2]):

                if np.any(hess_eigenvals[:, dim1, dim0] != 0.0):
                    max_eval_id = np.argmax(hess_eigenvals[:, dim1, dim0])

                    plt.plot(
                        (
                            dim0 - 0.5 * hess_eigenvecs[0, max_eval_id, dim1, dim0], 
                            dim0 + 0.5 * hess_eigenvecs[0, max_eval_id, dim1, dim0]
                        ), 
                        (
                            dim1 - 0.5 * hess_eigenvecs[1, max_eval_id, dim1, dim0],
                            dim1 + 0.5 * hess_eigenvecs[1, max_eval_id, dim1, dim0]
                        ),
                        c = "r"
                    )

                    if self._plot_orthogonal_dirs:
                        plt.plot(
                            (
                                dim0 - 0.5 * hess_eigenvecs[0, 1 - max_eval_id, dim1, dim0], 
                                dim0 + 0.5 * hess_eigenvecs[0, 1 - max_eval_id, dim1, dim0]
                            ), 
                            (
                                dim1 - 0.5 * hess_eigenvecs[1, 1 - max_eval_id, dim1, dim0],
                                dim1 + 0.5 * hess_eigenvecs[1, 1 - max_eval_id, dim1, dim0]
                            ),
                            c = "b",
                            linewidth = 0.2
                        )

        self._hesse_reco_pdf.savefig(fig)
    
    def _finalise(self):
        self._pdf.close()
        self._hesse_reco_pdf.close()
    