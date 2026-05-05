import math as m
import itertools
import typing

import scipy
from scipy.stats import laplace
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import json
from sklearn.cluster import DBSCAN

import torch
from torch import Tensor, tensor
from torch.nn.functional import conv2d, pad
from torch.nn import PoissonNLLLoss, MSELoss, L1Loss
from torch.optim import Adam

from liquidreco.hit import Hit, Hit2D, Hit3D
from liquidreco.modules.module_base import ModuleBase
from liquidreco.geometry import GeometryManager
from liquidreco.plotting import make_corner_plot, make_corner_plot_fiber_hits

class LaplaceFitter(ModuleBase):
    """Performs a fit of an ensemble of laplace distributions to observed collected charges in fibers in order to try to determine which 
    """

    def __init__(
        self,
    ):
        """Initialiser
        """

        self.requirements = ["x_fiber_hits", "y_fiber_hits", "z_fiber_hits"]

        self.outputs = [
            "laplace_x_peaks", "laplace_y_peaks", "laplace_z_peaks"
        ]

    def _initialise(self):

        self.DBSCAN_args = json.loads(self.args.DBSCAN_args)
        self._clusterer = DBSCAN(**self.DBSCAN_args)
        
        self._neighbourhood_dist = self.args.neighbourhood_dist 
        self._peak_candidate_threshold = self.args.peak_candidate_threshold
        self._amplitude_threshold = self.args.amplitude_threshold
        self._fix_width = self.args.fix_width
        self._min_n_neighbours = self.args.min_n_neighbours
        self._make_plots = self.args.make_plots

        ## use this to define neighbourhood around fibers
        self._neighbour_algo = NearestNeighbors(radius = self._neighbourhood_dist)

        self._pdf = matplotlib.backends.backend_pdf.PdfPages("LaplaceFitter-plots.pdf")
        self._cluster_pdf = matplotlib.backends.backend_pdf.PdfPages("LaplaceFitter-cluster-plots.pdf")
    
    def _setup_cli_options(self, parser):
        
        ## Note these are passed as a string but will later be converted to dictionaries
        parser.add_argument(
            "--DBSCAN-args", 
            help="arguments to pass to the DBSCAN algorithm used to build 'blob' clusters.", 
            required = False, default = "{\"eps\": 14.5}", type = str
        )
        parser.add_argument(
            "--make-plots", 
            help="Whether to make debug plots.", 
            action='store_true'
        )
        parser.add_argument(
            "--neighbourhood-dist", 
            help=" The distance around a fiber that is considered its local neighbourhood. Only fibers in the same neighbourhood are considered when sharing light in order to speed up the fit, defaults to 45.0", 
            type=float,
            required=False,
            default=45.0
        )
        parser.add_argument(
            "--peak-candidate-threshold", 
            help="Fibers with a charge above this are considered candidate peak hits, defaults to 100.0", 
            type=float,
            required=False,
            default=100.0
        )
        parser.add_argument(
            "--amplitude-threshold", 
            help="Fibers whose post-fit laplace distribution amplitude is above this are considered peak hits, defaults to 50.0", 
            type=float,
            required=False,
            default=50.0
        )
        parser.add_argument(
            "--fix-width", 
            help="Set this to fix the width of the laplace distributions in the fit. If None then it will be fit as a parameter, defaults to None", 
            type=float,
            required=False,
            default=None
        )
        parser.add_argument(
            "--min-n-neighbours", 
            help="Fibers must have this many other fibers in their neighbourhood to be included in the fit, defaults to 0", 
            type=int,
            required=False,
            default=0
        )

    def _finalise(self):

        self._pdf.close()
        self._cluster_pdf.close()

    def _process(self, event):
        
        x_fiber_hits: typing.List['Hit2D'] = event["x_fiber_hits"]
        y_fiber_hits: typing.List['Hit2D'] = event["y_fiber_hits"]
        z_fiber_hits: typing.List['Hit2D'] = event["z_fiber_hits"]

        laplace_x_peak_hits = list()
        laplace_y_peak_hits = list()
        laplace_z_peak_hits = list()

        for fiber_hits, laplace_peak_hits, u, v in zip(
            [x_fiber_hits, y_fiber_hits, z_fiber_hits],
            [laplace_x_peak_hits, laplace_y_peak_hits, laplace_z_peak_hits],
            ["z", "x", "x"],
            ["y", "z", "y"]
        ):
            
            if fiber_hits is None:
                continue

            ## try and fit the unused hits
            unused_clusters = self._cluster_hits(list(fiber_hits), u, v)
            print(f"N {u}{v} clusters: {len(unused_clusters)}")
            for cluster in unused_clusters:

                if(len(cluster) < 20):
                    continue

                _laplace_peaks = self._do_fit(cluster, u, v)
            
                for p in _laplace_peaks:
                    laplace_peak_hits.append(p)

        if self._make_plots:

            fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(5, 5))
            fig.suptitle("Homo-FGD Hits - Peaks")

            make_corner_plot_fiber_hits(
                fig,
                axs, 
                laplace_x_peak_hits ,
                laplace_y_peak_hits,
                laplace_z_peak_hits,
                colour_override="m"
            )

        event.add_data("laplace_x_peaks", laplace_x_peak_hits)
        event.add_data("laplace_y_peaks", laplace_y_peak_hits)
        event.add_data("laplace_z_peaks", laplace_z_peak_hits)

        return
    
    def _cluster_hits(
        self,
        fiber_hits:typing.List['Hit2D'],
        u:str, v:str,
    ) -> typing.List[typing.List['Hit2D']]:
        """Cluster hits using this PeakFinder2Ds _clusterer (default is DBSCAN but you can set it to whatever you want)

        :param fiber_hits: The hits to cluster
        :type fiber_hits: typing.List['Hit2D']
        :param u: the "u" direction, should be 'x', 'y' or 'z' depending on the projection
        :type u: str
        :param v:  the "v" direction, should be 'x', 'y' or 'z' depending on the projection
        :type v: str
        :return: hits broken down into clusters
        :rtype: typing.List[typing.List['Hit2D']]
        """
        
        positions = np.array([[getattr(hit, u), getattr(hit, v)] for hit in fiber_hits])
        cluster_ids = self._clusterer.fit_predict(positions)

        if self._make_plots:
            fig, ax = plt.subplots()
            if u == "y" and v == "z":
                ax.scatter(positions[:, 1], positions[:, 0], c = cluster_ids)
                ax.set_title(f"{v}{u} unused hit clusters")
                ax.set_xlabel(f"{v} [mm]")
                ax.set_ylabel(f"{u} [mm]")

            else:
                ax.scatter(positions[:, 1], positions[:, 0], c = cluster_ids)
                ax.set_title(f"{u}{v} unused hit clusters")
                ax.set_xlabel(f"{u} [mm]")
                ax.set_ylabel(f"{v} [mm]")

            self._cluster_pdf.savefig(fig)
            plt.close(fig)

        clusters = list()

        for id in np.unique(cluster_ids):
            ## -1 indicates "noise" cluster
            if id == -1:
                continue

            cluster = [fiber_hits[i] for i in np.where(cluster_ids == id)[0]]
            clusters.append(cluster)

        return clusters
        

    def _do_fit(
        self,
        fiber_hits: typing.List['Hit2D'], 
        u:str, v:str
    ) -> typing.Tuple[typing.List['Hit2D']]:
        """Find peaks by attempting to fit laplace distributions to prominent fibers

        :param fiber_hits: The hits to fit to
        :type fiber_hits: typing.List['Hit2D']
        :param u: The u direction, should be "x", "y" or "z"
        :type u: str
        :param v: The v direction, should be "x", "y" or "z"
        :type v: str
        :return: Peak hits found by the fit
        :rtype: typing.Tuple[typing.List['Hit2D']]
        """
        
        print(f"  - LAPLACE FIT: n {u}{v} fibers: {len(fiber_hits)}")
        
        ## now do the fit fit
        try:
            laplace_amplitudes = self._fit_laplace(
                fiber_hits,
                u,
                v
            )

        ## fit might fail for whatever reason, but we don't want that to bring the whole thing crashing down
        except scipy.optimize.OptimizeWarning:
            return []
        except RuntimeError:
            return []
        
        return fiber_hits

    def _fit_laplace(
            self, 
            hits:typing.List['Hit2D'],
            u:str,
            v:str,
        ) -> typing.List[float]:
        """Performs the fit of the laplace distributions

        :param hits: The hits that should be considered in the fit
        :type hits: typing.List['Hit2D'] 
        :param peak_candidates: The hits that should be considered peak candidates
        :type peak_candidates: typing.List['Hit2D'] 
        :param u: The u direction, should be "x", "y" or "z"
        :type u: str
        :param v: The v direction, should be "x", "y" or "z"
        :type v: str
        :return: The fitted laplace amplitudes of all peak candidate fibers
        :rtype: typing.List[float]
        """

        u_values = [getattr(hit, u) for hit in hits]
        v_values = [getattr(hit, v) for hit in hits]

        u_pitch = GeometryManager().get_pitch(u)
        v_pitch = GeometryManager().get_pitch(v)
        
        u_bins = np.arange(start=min(u_values) - 3.0 * u_pitch / 2.0, stop=max(u_values) + 5.0 * u_pitch / 2.0, step = u_pitch) 
        v_bins = np.arange(start=min(v_values) - 3.0 * v_pitch / 2.0, stop=max(v_values) + 5.0 * v_pitch / 2.0, step = v_pitch) 

        hist, _, _ = np.histogram2d(
            u_values,
            v_values,
            weights=[hit.weight for hit in hits],
            bins = (u_bins, v_bins)
        )

        u_fiber_positions = (u_bins[1:] + u_bins[:-1]) / 2.0
        v_fiber_positions = (v_bins[1:] + v_bins[:-1]) / 2.0

        fiber_positions = np.array(list(itertools.product(u_fiber_positions, v_fiber_positions)))
        indices = np.array(list(itertools.product(range(u_bins.shape[0] - 1), range(v_bins.shape[0] - 1))))
        charges = hist[indices[:, 0], indices[:, 1]]

        ## first do a check on the number of neighbours of each hit, discard ones with too few
        ## this speeds up algorithm as we don't want to consider things that are clearly not peaks
        hit_positions = np.array([u_values, v_values]).transpose()
        _, fiber_neighbour_indices = self._neighbour_algo.fit(hit_positions).radius_neighbors(hit_positions)

        considered_hits = []
        for i, hit in enumerate(hits):
            
            if hit.weight > self._peak_candidate_threshold and fiber_neighbour_indices[i].shape[0] > self._min_n_neighbours:

                considered_hits.append(hit)
    
        considered_hits.sort(key=lambda x: x.weight, reverse=True)

        pixel_pos_list = []
        p0 = []

        for fiber_hit in considered_hits:

            if len(pixel_pos_list) == 0 or len(hits) / (len(pixel_pos_list) + 8) > 2.0:
                
                fiber_u = getattr(fiber_hit, u)
                fiber_v = getattr(fiber_hit, v)

                pixel_pos_list.append([fiber_u + 2.5, fiber_v + 2.5])
                pixel_pos_list.append([fiber_u + 2.5, fiber_v - 2.5])
                pixel_pos_list.append([fiber_u - 2.5, fiber_v + 2.5])
                pixel_pos_list.append([fiber_u - 2.5, fiber_v - 2.5])

                p0.append(fiber_hit.weight)
                p0.append(fiber_hit.weight)
                p0.append(fiber_hit.weight)
                p0.append(fiber_hit.weight)

        p0.append(1.0) # <- the width param

        pixel_positions = np.array(pixel_pos_list)

        laplace_fn = self._get_laplace_fn(
            fiber_positions,
            pixel_positions
        )

        sigma = np.sqrt(charges)
        sigma[sigma == 0.0] = 1.0
        optimal_params, cov_mat = curve_fit(laplace_fn, xdata=None, ydata=charges, p0=p0, bounds=(0.0, np.inf), sigma = sigma)

        print(f"Optimal params: {optimal_params.shape} \n{optimal_params.tolist()}")

        weights = laplace_fn(None, *optimal_params.tolist())

        fig, axs = plt.subplots(3,1)
        
        if self._make_plots:
            mappable = axs[0].scatter(
                fiber_positions[:, 0], fiber_positions[:, 1], 
                c = charges,
                cmap=plt.get_cmap("coolwarm")
            )

            fig.colorbar(mappable, ax=axs[0])

            mappable = axs[1].scatter(
                fiber_positions[:, 0], fiber_positions[:, 1], 
                c = weights,
                cmap=plt.get_cmap("coolwarm")
            )
            
            fig.colorbar(mappable, ax=axs[1])
            
            mappable = axs[2].scatter(
                pixel_positions[:, 0],
                pixel_positions[:, 1],
                c = optimal_params.tolist()[:-1], 
                cmap=plt.get_cmap("coolwarm")
            )

            fig.colorbar(mappable, ax=axs[2])

            self._pdf.savefig(fig)

            plt.clf()

        return optimal_params.tolist()[:-1]
    
    def _get_laplace_fn(
            self, 
            fiber_positions:np.ndarray, 
            pixel_positions:np.ndarray,
        ) -> typing.Callable:
        """Construct the ensemble of laplace functions used for fitting
        """

        n_fibers = fiber_positions.shape[0]
        n_pixels = pixel_positions.shape[0]

        ## get which fibers should be affected by energy deposits from which pixels
        pixel_distances, pixel_indices = self._neighbour_algo.fit(pixel_positions).radius_neighbors(fiber_positions)

        print(f"N fibers:   {n_fibers}")
        print(f"N pixels:   {n_pixels}")

        def _laplace_fn(x_data=None, *params, fix_width:float=self._fix_width):

            amplitudes = np.array(params[:-1])

            print(amplitudes)

            if fix_width is None:
                width = params[-1]
            else:
                width = fix_width

            ret_charges = np.zeros(n_fibers)

            for fiber_index in range(n_fibers):
                    
                pixel_amplitudes = amplitudes[pixel_indices[fiber_index]]

                l = pixel_amplitudes * laplace.pdf(pixel_distances[fiber_index] / width ) / width

                ret_charges[fiber_index] = np.sum(l)

            return ret_charges

        return _laplace_fn


class Deconv2D(ModuleBase):
    """Performs a deconvolution from fiber hits to light deposited in detector "pixels" in 2D
    """

    def __init__(
        self,
    ):
        """Initialiser
        """

        self.requirements = ["x_fiber_hits", "y_fiber_hits", "z_fiber_hits"]

        self.outputs = [
            "x_fiber_hits", "y_fiber_hits", "z_fiber_hits"
        ]

    def _initialise(self):

        self.DBSCAN_args = json.loads(self.args.DBSCAN_args)
        self._clusterer = DBSCAN(**self.DBSCAN_args)
        
        self._amplitude_threshold = self.args.amplitude_threshold
        self._make_plots = self.args.make_plots
        self._kernel_size = self.args.kernel_size
        self._pixel_divisions = self.args.pixel_divisions
        self._laplace_width = self.args.laplace_width
        self._n_steps = self.args.n_steps

        self._pdf = matplotlib.backends.backend_pdf.PdfPages("Deconv2D-plots.pdf")
        self._cluster_pdf = matplotlib.backends.backend_pdf.PdfPages("Deconv2D-cluster-plots.pdf")

        ## make kernels once here so we don't have to do it for every event
        self._kernels: typing.Dict[str, Tensor] = {
            "zy": self._make_kernel("z", "y"),
            "xz": self._make_kernel("x", "z"),
            "xy": self._make_kernel("x", "y")
        }
    
    def _setup_cli_options(self, parser):
        
        ## Note these are passed as a string but will later be converted to dictionaries
        parser.add_argument(
            "--DBSCAN-args", 
            help="arguments to pass to the DBSCAN algorithm used to build 'blob' clusters.", 
            required = False, default = "{\"eps\": 14.5}", type = str
        )
        parser.add_argument(
            "--make-plots", 
            help="Whether to make debug plots.", 
            action='store_true'
        )
        parser.add_argument(
            "--amplitude-threshold", 
            help="Fibers whose post-fit laplace distribution amplitude is above this are considered peak hits, defaults to 50.0", 
            type=float,
            required=False,
            default=50.0
        )
        parser.add_argument(
            "--n-steps", 
            help="The number of iterations to run the optimiser for. defaults to 20000", 
            type=int,
            required=False,
            default=20000
        )
        parser.add_argument(
            "--kernel-size", 
            help="size of the convolution kernel in units of fiber pitch (kernel-size = 4 means kernel will span 4 fibers - 2 on either side of the central fiber)", 
            type=int,
            required=False,
            default=6
        )
        parser.add_argument(
            "--pixel-divisions", 
            help="The number of 'pixels' to make between each fiber", 
            type=int,
            required=False,
            default=2
        )
        parser.add_argument(
            "--laplace-width", 
            help="The width of the laplace distribution", 
            type=float,
            required=False,
            default=7.0
        )

    def _finalise(self):

        self._pdf.close()
        self._cluster_pdf.close()

    def _process(self, event):
        
        x_fiber_hits: typing.List['Hit2D'] = event["x_fiber_hits"]
        y_fiber_hits: typing.List['Hit2D'] = event["y_fiber_hits"]
        z_fiber_hits: typing.List['Hit2D'] = event["z_fiber_hits"]

        x_deposits = list()
        y_deposits = list()
        z_deposits = list()

        for fiber_hits, deposits, u, v in zip(
            [x_fiber_hits, y_fiber_hits, z_fiber_hits],
            [x_deposits, y_deposits, z_deposits],
            ["z", "x", "x"],
            ["y", "z", "y"]
        ):
            
            if fiber_hits is None:
                continue
            
            ## try and fit the unused hits
            unused_clusters = self._cluster_hits(list(fiber_hits), u, v)
            print(f"N {u}{v} clusters: {len(unused_clusters)}")
            for cluster in unused_clusters:

                if(len(cluster) < 20):
                    continue

                _hits = self._do_fit(cluster, u, v)

                for hit in _hits:
                    deposits.append(hit)

        if self._make_plots:
            pass

        event.add_data("x_fiber_hits", x_deposits)
        event.add_data("y_fiber_hits", y_deposits)
        event.add_data("z_fiber_hits", z_deposits)

        return
    
    def _make_kernel(
            self,
            u:str, v:str
        ) -> Tensor: 

        assert (
            (GeometryManager().get_pitch(u) == GeometryManager().get_pitch(v))
        ), "Sorry, Deconv2D only supports uniform fiber grids at the moment (need pitch u == pitch v) :("

        assert self._kernel_size %2 == 0, "Kernel size must be even!!!!"

        kernel_size_pixels = int(self._kernel_size * self._pixel_divisions)
        pitch = GeometryManager().get_pitch("x")
        pixel_width = pitch / self._pixel_divisions

        np_kernel = np.zeros((kernel_size_pixels, kernel_size_pixels)) 

        for i  in range(-int(kernel_size_pixels / 2) + 1, int(kernel_size_pixels / 2)):
            for j in range(-int(kernel_size_pixels / 2) + 1, int(kernel_size_pixels / 2)):

                kernel_i = int(kernel_size_pixels / 2 + i)
                kernel_j = int(kernel_size_pixels / 2 + j)

                distance = np.linalg.norm([(i + 0.5) * pixel_width, (j + 0.5) * pixel_width]) - 0.7 * pixel_width

                weight = laplace.pdf(distance / self._laplace_width ) / self._laplace_width

                np_kernel[kernel_i, kernel_j] = weight
                
        fig = plt.figure()
        plt.imshow(np_kernel)
        plt.colorbar()

        self._pdf.savefig(fig)
        
        return tensor(np_kernel)
    
    def _get_kernel(
            self,
            u:str, v:str
        ) -> Tensor: 

        return self._kernels[f'{u}{v}']


    def _cluster_hits(
        self,
        fiber_hits:typing.List['Hit2D'],
        u:str, v:str,
    ) -> typing.List[typing.List['Hit2D']]:
        """Cluster hits using this PeakFinder2Ds _clusterer (default is DBSCAN but you can set it to whatever you want)

        :param fiber_hits: The hits to cluster
        :type fiber_hits: typing.List['Hit2D']
        :param u: the "u" direction, should be 'x', 'y' or 'z' depending on the projection
        :type u: str
        :param v:  the "v" direction, should be 'x', 'y' or 'z' depending on the projection
        :type v: str
        :return: hits broken down into clusters
        :rtype: typing.List[typing.List['Hit2D']]
        """
        
        positions = np.array([[getattr(hit, u), getattr(hit, v)] for hit in fiber_hits])
        cluster_ids = self._clusterer.fit_predict(positions)

        if self._make_plots:
            fig, ax = plt.subplots()
            if u == "y" and v == "z":
                ax.scatter(positions[:, 1], positions[:, 0], c = cluster_ids)
                ax.set_title(f"{v}{u} unused hit clusters")
                ax.set_xlabel(f"{v} [mm]")
                ax.set_ylabel(f"{u} [mm]")

            else:
                ax.scatter(positions[:, 1], positions[:, 0], c = cluster_ids)
                ax.set_title(f"{u}{v} unused hit clusters")
                ax.set_xlabel(f"{u} [mm]")
                ax.set_ylabel(f"{v} [mm]")

            self._cluster_pdf.savefig(fig)
            plt.close(fig)

        clusters = list()

        for id in np.unique(cluster_ids):
            ## -1 indicates "noise" cluster
            if id == -1:
                continue

            cluster = [fiber_hits[i] for i in np.where(cluster_ids == id)[0]]
            clusters.append(cluster)

        return clusters

    def _do_fit(
        self,
        fiber_hits: typing.List['Hit2D'], 
        u:str, v:str
    ) -> typing.Tuple[typing.List['Hit2D']]:
        """Find peaks by attempting to fit laplace distributions to prominent fibers

        :param fiber_hits: The hits to fit to
        :type fiber_hits: typing.List['Hit2D']
        :param u: The u direction, should be "x", "y" or "z"
        :type u: str
        :param v: The v direction, should be "x", "y" or "z"
        :type v: str
        :return: Peak hits found by the fit
        :rtype: typing.Tuple[typing.List['Hit2D']]
        """
        
        print(f"  - LAPLACE FIT: n {u}{v} fibers: {len(fiber_hits)}")
        
        ## now do the fit fit
        pixel_tensor, u_pixel_positions, v_pixel_positions = self._fit_laplace(
            fiber_hits,
            u,
            v
        )

        hits = []
        for i in range(pixel_tensor.shape[1]):
            for j in range(pixel_tensor.shape[2]):

                if pixel_tensor[0, i, j] > self._amplitude_threshold:
                    pos_temp = {"x": None, "y": None, "z": None}
                    pos_temp[u] = u_pixel_positions[i]
                    pos_temp[v] = v_pixel_positions[j]
                    hit_tmp = Hit2D(pos_temp, weight = pixel_tensor[0, i, j].detach().numpy())
                    hits.append(hit_tmp)

        return hits

    def _fit_laplace(
            self, 
            hits:typing.List['Hit2D'],
            u:str,
            v:str,
        ) -> typing.Tuple[Tensor, typing.List[float], typing.List[float]]:
        """Performs the fit of the laplace distributions

        :param hits: The hits that should be considered in the fit
        :type hits: typing.List['Hit2D'] 
        :param peak_candidates: The hits that should be considered peak candidates
        :type peak_candidates: typing.List['Hit2D'] 
        :param u: The u direction, should be "x", "y" or "z"
        :type u: str
        :param v: The v direction, should be "x", "y" or "z"
        :type v: str
        :return: The fitted laplace amplitudes of all peak candidate fibers
        :rtype: typing.List[float]
        """

        u_values = [getattr(hit, u) for hit in hits]
        v_values = [getattr(hit, v) for hit in hits]

        u_pitch = GeometryManager().get_pitch(u)
        v_pitch = GeometryManager().get_pitch(v)
        
        fiber_u_bins = np.arange(start=min(u_values) - 3.0 * u_pitch / 2.0, stop=max(u_values) + 5.0 * u_pitch / 2.0, step = u_pitch) 
        fiber_v_bins = np.arange(start=min(v_values) - 3.0 * v_pitch / 2.0, stop=max(v_values) + 5.0 * v_pitch / 2.0, step = v_pitch) 

        fiber_hist, u_bins, v_bins = np.histogram2d(
            u_values,
            v_values,
            weights=[hit.weight for hit in hits],
            bins = (fiber_u_bins, fiber_v_bins)
        )

        fiber_tensor = tensor(fiber_hist)

        pixel_tensor = torch.repeat_interleave(fiber_tensor * 10.0, self._pixel_divisions, dim=0)
        pixel_tensor = torch.repeat_interleave(pixel_tensor, self._pixel_divisions, dim=1)
        pixel_tensor = torch.unsqueeze(pixel_tensor, 0)
        
        kernel = torch.unsqueeze(torch.unsqueeze(self._get_kernel(u, v), 0), 0)
    
        padding = (
            int(kernel.shape[2] /2 - 1), 
            int(kernel.shape[2] /2),
            int(kernel.shape[3] /2 - 1), 
            int(kernel.shape[3] /2)
        )

        ## get the positions of the fibers
        u_fiber_positions = (u_bins[1:] + u_bins[:-1]) / 2.0
        v_fiber_positions = (v_bins[1:] + v_bins[:-1]) / 2.0

        ## get geometry stuff
        u_pitch = GeometryManager().get_pitch(u)
        v_pitch = GeometryManager().get_pitch(v)
        u_pixel_width = u_pitch / self._pixel_divisions
        v_pixel_width = v_pitch / self._pixel_divisions

        u_pixel_positions = []
        v_pixel_positions = []

        for u_fiber in u_fiber_positions:
            for i_pixel in range(self._pixel_divisions):
                u_pixel_positions.append(u_fiber + (i_pixel - 0.5) * u_pixel_width)
        for v_fiber in v_fiber_positions:
            for i_pixel in range(self._pixel_divisions):
                v_pixel_positions.append(v_fiber + (i_pixel - 0.5) * v_pixel_width)

        pixel_tensor.requires_grad = True

        print(f'fiber_tensor shape: {fiber_tensor.shape}')
        print(f'pixel_tensor shape: {pixel_tensor.shape}')
        print(f'kernel shape: {kernel.shape}')

        loss_fn = L1Loss() #PoissonNLLLoss(log_input=False)
        optimiser = Adam(params = [pixel_tensor], lr = 1.0)
        
        for lr in [1.0]:

            optimiser.lr = lr
            
            print(f'##### LR = {lr} #####')

            for step in range (self._n_steps):

                pixel_tensor.grad = None
                conv = conv2d(pad(pixel_tensor, padding), kernel, stride = self._pixel_divisions)
                loss = loss_fn(conv, fiber_tensor)
                
                loss.backward()

                optimiser.step()
                optimiser.zero_grad()

                with torch.no_grad():
                    pixel_tensor[:] = pixel_tensor.clamp(min = 0.0)
                
                if step % int(m.floor(self._n_steps / 10)) == 0:
                    print(f'  - step: {step} :: loss: {loss}')

                step += 1

        if self._make_plots:
            
            fig, axs = plt.subplots(3,1, figsize=(20, 60))

            mappable = axs[0].imshow(
                fiber_hist, 
                cmap=plt.get_cmap("coolwarm")
            )

            fig.colorbar(mappable, ax=axs[0])

            conv = conv2d(pad(pixel_tensor, padding), kernel, stride = self._pixel_divisions)

            mappable = axs[1].imshow(
                conv.detach().numpy()[0, ...], 
                cmap=plt.get_cmap("coolwarm")
            )

            fig.colorbar(mappable, ax=axs[1])

            mappable = axs[2].imshow(
                pixel_tensor.detach().numpy()[0, ...], 
                cmap=plt.get_cmap("coolwarm")
            )

            fig.colorbar(mappable, ax=axs[2])

            self._pdf.savefig(fig)

            plt.clf()

        return pixel_tensor, u_pixel_positions, v_pixel_positions
    




class Deconv3D(ModuleBase):
    """Performs a deconvolution from fiber hits to light deposited in detector "voxels" in 3D
    """

    def __init__(
        self,
    ):
        """Initialiser
        """

        self.requirements = ["x_fiber_hits", "y_fiber_hits", "z_fiber_hits"]

        self.outputs = [
            "3d_hits"
        ]

    def _initialise(self):
        
        self._amplitude_threshold = self.args.amplitude_threshold
        self._make_plots = self.args.make_plots
        self._kernel_size = self.args.kernel_size
        self._pixel_divisions = self.args.pixel_divisions
        self._laplace_width = self.args.laplace_width
        self._n_steps = self.args.n_steps

        self._pdf = matplotlib.backends.backend_pdf.PdfPages("Deconv3D-plots.pdf")
        self._cluster_pdf = matplotlib.backends.backend_pdf.PdfPages("Deconv3D-cluster-plots.pdf")

        ## make kernels once here so we don't have to do it for every event
        self._kernels: typing.Dict[str, Tensor] = {
            "yz": self._make_kernel("y", "z"),
            "xz": self._make_kernel("x", "z"),
            "xy": self._make_kernel("x", "y")
        }
    
    def _setup_cli_options(self, parser):
        
        ## Note these are passed as a string but will later be converted to dictionaries
        parser.add_argument(
            "--make-plots", 
            help="Whether to make debug plots.", 
            action='store_true'
        )
        parser.add_argument(
            "--amplitude-threshold", 
            help="Fibers whose post-fit laplace distribution amplitude is above this are considered peak hits, defaults to 50.0", 
            type=float,
            required=False,
            default=50.0
        )
        parser.add_argument(
            "--n-steps", 
            help="The number of iterations to run the optimiser for. defaults to 20000", 
            type=int,
            required=False,
            default=20000
        )
        parser.add_argument(
            "--kernel-size", 
            help="size of the convolution kernel in units of fiber pitch (kernel-size = 4 means kernel will span 4 fibers - 2 on either side of the central fiber)", 
            type=int,
            required=False,
            default=6
        )
        parser.add_argument(
            "--pixel-divisions", 
            help="The number of 'pixels' to make between each fiber", 
            type=int,
            required=False,
            default=2
        )
        parser.add_argument(
            "--laplace-width", 
            help="The width of the laplace distribution", 
            type=float,
            required=False,
            default=7.0
        )

    def _finalise(self):

        self._pdf.close()
        self._cluster_pdf.close()

    def _process(self, event):
        
        x_fiber_hits: typing.List['Hit2D'] = event["x_fiber_hits"]
        y_fiber_hits: typing.List['Hit2D'] = event["y_fiber_hits"]
        z_fiber_hits: typing.List['Hit2D'] = event["z_fiber_hits"]

        hits_3d = list()
            
        ## try and fit the 2d hits
        _hits = self._do_fit(x_fiber_hits, y_fiber_hits, z_fiber_hits)

        for hit in _hits:
            hits_3d.append(hit)

        if self._make_plots:
            pass

        event.add_data("3d_hits", hits_3d)

        return
    
    def _make_kernel(
            self,
            u:str, v:str
        ) -> Tensor: 

        assert (
            (GeometryManager().get_pitch(u) == GeometryManager().get_pitch(v))
        ), "Sorry, Deconv2D only supports uniform fiber grids at the moment (need pitch u == pitch v) :("

        assert self._kernel_size %2 == 0, "Kernel size must be even!!!!"

        kernel_size_pixels = int(self._kernel_size * self._pixel_divisions)
        pitch = GeometryManager().get_pitch("x")
        pixel_width = pitch / self._pixel_divisions

        np_kernel = np.zeros((kernel_size_pixels, kernel_size_pixels)) 

        for i  in range(-int(kernel_size_pixels / 2) + 1, int(kernel_size_pixels / 2)):
            for j in range(-int(kernel_size_pixels / 2) + 1, int(kernel_size_pixels / 2)):

                kernel_i = int(kernel_size_pixels / 2 + i)
                kernel_j = int(kernel_size_pixels / 2 + j)

                distance = np.linalg.norm([(i + 0.5) * pixel_width, (j + 0.5) * pixel_width]) - 0.7 * pixel_width

                weight = laplace.pdf(distance / self._laplace_width ) / self._laplace_width

                np_kernel[kernel_i, kernel_j] = weight
                
        fig = plt.figure()
        plt.imshow(np_kernel)
        plt.colorbar()

        self._pdf.savefig(fig)
        
        return tensor(np_kernel)
    
    def _get_kernel(
            self,
            u:str, v:str
        ) -> Tensor: 

        return self._kernels[f'{u}{v}']

    def _do_fit(
        self,
        x_fiber_hits: typing.List['Hit2D'], 
        y_fiber_hits: typing.List['Hit2D'], 
        z_fiber_hits: typing.List['Hit2D']
    ) -> typing.Tuple[typing.List['Hit2D']]:
        """Find peaks by attempting to fit laplace distributions to prominent fibers

        :param x_fiber_hits: The x hits to fit to
        :type fiber_hits: typing.List['Hit2D']
        :param y_fiber_hits: The y hits to fit to
        :type fiber_hits: typing.List['Hit2D']
        :param z_fiber_hits: The z hits to fit to
        :type fiber_hits: typing.List['Hit2D']
        :return: Peak hits found by the fit
        :rtype: typing.Tuple[typing.List['Hit2D']]
        """
        
        ## now do the fit fit
        pixel_tensor, u_pixel_positions, v_pixel_positions, w_pixel_positions = self._fit_laplace(
            x_fiber_hits,
            y_fiber_hits,
            z_fiber_hits
        )

        hits = []
        for i in range(pixel_tensor.shape[1]):
            for j in range(pixel_tensor.shape[2]):
                for k in range(pixel_tensor.shape[3]):

                    if pixel_tensor[0, i, j, k] > self._amplitude_threshold:
                        pos_temp = {"x": None, "y": None, "z": None}
                        pos_temp["x"] = u_pixel_positions[i]
                        pos_temp["y"] = v_pixel_positions[j]
                        pos_temp["z"] = w_pixel_positions[k]
                        hit_tmp = Hit3D(pos_temp, weight = pixel_tensor[0, i, j, k].detach().numpy())
                        hits.append(hit_tmp)

        return hits

    def _fit_laplace(
            self, 
            x_hits:typing.List['Hit2D'],
            y_hits:typing.List['Hit2D'],
            z_hits:typing.List['Hit2D']
        ) -> typing.Tuple[Tensor, typing.List[float], typing.List[float]]:
        """Performs the fit of the laplace distributions

        :param x_hits: The x hits that should be considered in the fit
        :type hits: typing.List['Hit2D'] 
        :param y_hits: The y hits that should be considered in the fit
        :type hits: typing.List['Hit2D'] 
        :param z_hits: The z hits that should be considered in the fit
        :type hits: typing.List['Hit2D'] 
        :return: The fitted laplace amplitudes of all peak candidate fibers
        :rtype: typing.List[float]
        """
        
        x_values = [getattr(hit, "x") for hit in [*y_hits, *z_hits]]
        y_values = [getattr(hit, "y") for hit in [*x_hits, *z_hits]]
        z_values = [getattr(hit, "z") for hit in [*x_hits, *y_hits]]

        x_pitch = GeometryManager().get_pitch("x")
        y_pitch = GeometryManager().get_pitch("y")
        z_pitch = GeometryManager().get_pitch("z")

        fiber_x_bins = np.arange(start=min(x_values) - 3.0 * x_pitch / 2.0, stop=max(x_values) + 5.0 * x_pitch / 2.0, step = x_pitch) 
        fiber_y_bins = np.arange(start=min(y_values) - 3.0 * y_pitch / 2.0, stop=max(y_values) + 5.0 * y_pitch / 2.0, step = y_pitch) 
        fiber_z_bins = np.arange(start=min(z_values) - 3.0 * z_pitch / 2.0, stop=max(z_values) + 5.0 * z_pitch / 2.0, step = z_pitch) 

        x_fiber_hist, y_bins, z_bins = np.histogram2d(
            [hit.y for hit in x_hits],
            [hit.z for hit in x_hits],
            weights=[hit.weight for hit in x_hits],
            bins = (fiber_y_bins, fiber_z_bins)
        )

        y_fiber_hist, x_bins, z_bins = np.histogram2d(
            [hit.x for hit in y_hits],
            [hit.z for hit in y_hits],
            weights=[hit.weight for hit in y_hits],
            bins = (fiber_x_bins, fiber_z_bins)
        )

        z_fiber_hist, x_bins, y_bins = np.histogram2d(
            [hit.x for hit in z_hits],
            [hit.y for hit in z_hits],
            weights=[hit.weight for hit in z_hits],
            bins = (fiber_x_bins, fiber_y_bins)
        )

        x_fiber_tensor = tensor(x_fiber_hist)
        y_fiber_tensor = tensor(y_fiber_hist)
        z_fiber_tensor = tensor(z_fiber_hist)

        pixel_tensor = tensor(
            np.zeros(
                (
                    (fiber_x_bins.shape[0] - 1) * self._pixel_divisions, 
                    (fiber_y_bins.shape[0] - 1) * self._pixel_divisions, 
                    (fiber_z_bins.shape[0] - 1) * self._pixel_divisions
                )
            )
        )
        pixel_tensor = torch.unsqueeze(pixel_tensor, 0)
        
        x_kernel = torch.unsqueeze(torch.unsqueeze(self._get_kernel("y", "z"), 0), 0)
        y_kernel = torch.unsqueeze(torch.unsqueeze(self._get_kernel("x", "z"), 0), 0)
        z_kernel = torch.unsqueeze(torch.unsqueeze(self._get_kernel("x", "y"), 0), 0)
    
        padding = (
            int(self._kernel_size * self._pixel_divisions / 2 - 1), 
            int(self._kernel_size * self._pixel_divisions / 2),
            int(self._kernel_size * self._pixel_divisions / 2 - 1), 
            int(self._kernel_size * self._pixel_divisions / 2)
        )

        pixel_x_positions = np.arange(start=min(x_values) - 3.0 * x_pitch / 2.0, stop=max(x_values) + 5.0 * x_pitch / 2.0, step = x_pitch / self._pixel_divisions) 
        pixel_y_positions = np.arange(start=min(y_values) - 3.0 * y_pitch / 2.0, stop=max(y_values) + 5.0 * y_pitch / 2.0, step = y_pitch / self._pixel_divisions) 
        pixel_z_positions = np.arange(start=min(z_values) - 3.0 * z_pitch / 2.0, stop=max(z_values) + 5.0 * z_pitch / 2.0, step = z_pitch / self._pixel_divisions) 

        pixel_tensor.requires_grad = True

        print(f'x fiber_tensor shape: {x_fiber_tensor.shape}')
        print(f'y fiber_tensor shape: {y_fiber_tensor.shape}')
        print(f'z fiber_tensor shape: {z_fiber_tensor.shape}')
        print(f'pixel_tensor shape: {pixel_tensor.shape}')
        print(f'x kernel shape: {x_kernel.shape}')
        print(f'y kernel shape: {y_kernel.shape}')
        print(f'z kernel shape: {z_kernel.shape}')
        print(f'x positions shape: {pixel_x_positions.shape}')
        print(f'y positions shape: {pixel_y_positions.shape}')
        print(f'z positions shape: {pixel_z_positions.shape}')

        loss_fn = L1Loss() #PoissonNLLLoss(log_input=False)
        optimiser = Adam(params = [pixel_tensor])
        
        for lr in [0.01, 0.001, 0.0001]:

            optimiser.lr = lr

            print(f'##### LR = {lr} #####')

            for step in range (self._n_steps):

                pixel_tensor.grad = None
                
                x_conv = conv2d(pad(torch.sum(pixel_tensor, dim=1), padding), x_kernel, stride = self._pixel_divisions)
                y_conv = conv2d(pad(torch.sum(pixel_tensor, dim=2), padding), y_kernel, stride = self._pixel_divisions)
                z_conv = conv2d(pad(torch.sum(pixel_tensor, dim=3), padding), z_kernel, stride = self._pixel_divisions)

                x_loss = loss_fn(x_conv[0], x_fiber_tensor)
                y_loss = loss_fn(y_conv[0], y_fiber_tensor)
                z_loss = loss_fn(z_conv[0], z_fiber_tensor)

                loss = tensor([0], dtype = torch.double)

                loss += x_loss 
                loss += y_loss
                loss += z_loss
                
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                
                with torch.no_grad():
                    pixel_tensor[:] = pixel_tensor.clamp(min = 0.0)

                if step % int(m.floor(self._n_steps / 10)) == 0:
                    print(f'  - step: {step} :: loss: {loss} (x loss = {x_loss} y_loss = {y_loss} z_loss = {z_loss})')

                step += 1

        if self._make_plots:
            
            fig, axs = plt.subplots(3,1, figsize=(20, 60))

            mappable = axs[0].imshow(
                x_fiber_hist, 
                cmap=plt.get_cmap("coolwarm")
            )

            fig.colorbar(mappable, ax=axs[0])

            conv = conv2d(pad(torch.sum(pixel_tensor, 1), padding), x_kernel, stride = self._pixel_divisions)

            mappable = axs[1].imshow(
                conv.detach().numpy()[0, ...], 
                cmap=plt.get_cmap("coolwarm")
            )

            fig.colorbar(mappable, ax=axs[1])

            mappable = axs[2].imshow(
                torch.sum(pixel_tensor.detach(), dim=1).numpy()[0, ...], 
                cmap=plt.get_cmap("coolwarm")
            )

            fig.colorbar(mappable, ax=axs[2])

            self._pdf.savefig(fig)

            plt.clf()

        return pixel_tensor, pixel_x_positions, pixel_y_positions, pixel_z_positions
    