from liquidreco.plotting import make_corner_plot, make_corner_plot_fiber_hits

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import scipy
from scipy.stats import laplace
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.linalg import eig

import typing
import json

from liquidreco.hit import Hit, Hit2D, Hit3D
from liquidreco.modules.module_base import ModuleBase
from liquidreco.event import Event

class LaplaceFitter:
    """Performs a fit of an ensemble of laplace distributions to observed collected charges in fibers in order to try to determine which 
    """

    def __init__(
        self,
        neighbourhood_dist:float = 45.0,
        peak_candidate_threshold:float = 100.0,
        amplitude_threshold:float = 50.0,
        fix_width: float = None,
        min_n_neighbours: int = 0,
        make_plots = False
    ):
        """Initialiser

        :param neighbourhood_dist: The distance around a fiber that is considered its local neighbourhood. Only fibers in the same neighbourhood are considered when sharing light in order to speed up the fit, defaults to 45.0
        :type neighbourhood_dist: float, optional
        :param peak_candidate_threshold: Fibers with a charge above this are considered candidate peak hits, defaults to 100.0
        :type peak_candidate_threshold: float, optional
        :param amplitude_threshold: Fibers whose post-fit laplace distribution amplitude is above this are considered peak hits, defaults to 50.0
        :type amplitude_threshold: float, optional
        :param fix_width: Set this to fix the width of the laplace distributions in the fit. If None then it will be fit as a parameter, defaults to None
        :type fix_width: float, optional
        :param min_n_neighbours: Fibers must have this many other fibers in their neighbourhood to be included in the fit, defaults to 0
        :type min_n_neighbours: int, optional
        :param make_plots: Whether this fitter should make debug plots while running
        :type fix_width: bool, optional
        """

        self._neighbourhood_dist:float = neighbourhood_dist
        self._peak_candidate_threshold:float = peak_candidate_threshold
        self._amplitude_threshold:float = amplitude_threshold
        self._fix_width:float = fix_width
        self._min_n_neighbours = min_n_neighbours
        self._make_plots:bool = make_plots

        ## use this to define neighbourhood around fibers
        self._neighbour_algo = NearestNeighbors(radius = self._neighbourhood_dist)

        self._pdf = matplotlib.backends.backend_pdf.PdfPages("LaplaceFitter-plots.pdf")

    def finalise(self):

        self._pdf.close()

    
    def __call__(
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
        
        ## get positions of the hits for convenience
        positions = np.array([[getattr(hit, u), getattr(hit, v)] for hit in fiber_hits])
    
        ## first do a check on the number of neighbours of each hit, discard ones with too few
        ## this speeds up algorithm as we don't want to consider things that are clearly not peaks
        _, fiber_neighbour_indices = self._neighbour_algo.fit(positions).radius_neighbors(positions)
        considered_hits = [fiber_hits[i] for i in range(len(fiber_hits)) if fiber_neighbour_indices[i].shape[0] > self._min_n_neighbours]

        print(f"  - LAPLACE FIT: n {u}{v} fibers passing N neighbours cut: {len(considered_hits)}")

        if len(considered_hits) == 0:
            return []
        
        ## now apply condition on the charge in the fiber
        peak_candidates = [hit for hit in considered_hits if hit.weight > self._peak_candidate_threshold]

        print(f"  - LAPLACE FIT: n {u}{v} peak candidate fibers: {len(peak_candidates)}")

        if len(peak_candidates) == 0:
            return []

        ## now do the fit fit
        try:
            laplace_amplitudes = self._fit_laplace(
                considered_hits,
                peak_candidates,
                u,
                v
            )

        ## fit might fail for whatever reason, but we don't want that to bring the whole thing crashing down
        except scipy.optimize.OptimizeWarning:
            return []
        except RuntimeError:
            return []
        
        return [peak_candidates[i] for i in range(len(peak_candidates)) if laplace_amplitudes[i] > self._amplitude_threshold]

    def _fit_laplace(
            self, 
            hits:typing.List['Hit2D'],
            peak_candidates:typing.List['Hit2D'],
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

        fiber_positions = np.array([[getattr(hit, u), getattr(hit, v)] for hit in hits])
        peak_candidate_positions = np.array([[getattr(hit, u), getattr(hit, v)] for hit in peak_candidates])
        
        ## now get neighbours of the peak candidates
        neighbour_distances, neighbour_indices = self._neighbour_algo.fit(peak_candidate_positions).radius_neighbors(fiber_positions)
        charges = np.array([hit.weight for hit in hits])

        laplace_fn, p0 = self._get_laplace_fn(
            fiber_positions, 
            charges, 
            neighbour_distances, 
            neighbour_indices
        )

        optimal_params, cov_mat = curve_fit(laplace_fn, xdata=None, ydata=charges, p0=p0, bounds=(0.0, np.inf))

        print(f"Optimal params: {optimal_params.shape} \n{optimal_params.tolist()}")

        weights = laplace_fn(None, *optimal_params.tolist())

        fig, axs = plt.subplots(3,1)
        
        if self._make_plots:
            h, x_bins, y_bins, _ = axs[0].hist2d(
                fiber_positions[:, 0], fiber_positions[:, 1], 
                weights = charges, bins=(20, 20), 
                cmap=plt.get_cmap("coolwarm"),
                vmax=self._peak_candidate_threshold
            )

            mappable = axs[1].hist2d(
                fiber_positions[:, 0], fiber_positions[:, 1], 
                weights = weights, bins=(20, 20), 
                cmap=plt.get_cmap("coolwarm"), 
                vmax=self._peak_candidate_threshold
            )
            
            fig.colorbar(mappable[3], ax=axs)
            
            axs[2].hist2d(
                [fiber_positions[i, 0] for i in range(len(charges)) if charges[i] > self._peak_candidate_threshold], 
                [fiber_positions[i, 1] for i in range(len(charges)) if charges[i] > self._peak_candidate_threshold], 
                weights = optimal_params.tolist()[:-1], 
                bins=(x_bins, y_bins), 
                cmap=plt.get_cmap("coolwarm"),
                #cmin = 0.0001,
                vmax = self._peak_candidate_threshold
            )

            self._pdf.savefig(fig)

            plt.clf()

            plt.scatter([charge for charge in charges if charge > self._peak_candidate_threshold], optimal_params[:-1])
            self._pdf.savefig(fig)
            plt.close(fig)

        return optimal_params.tolist()[:-1]
    
    def _get_laplace_fn(
            self, 
            fiber_positions:np.ndarray, 
            fiber_charges:np.ndarray, 
            neighbour_distances:np.array, 
            neighbour_indices:typing.List[np.array],
        ) -> typing.Callable:
        """Construct the ensemble of laplace functions used for fitting

        :param fiber_positions: Positions of all of the fibers included in the fit
        :type fiber_positions: np.ndarray
        :param fiber_charges: Measured charges for all fibers included in the fit
        :type fiber_charges: np.ndarray
        :param neighbour_distances: The distances from all fibers to the peak candidates
        :type neighbour_distances: np.array
        :param neighbour_indices: The indices indicating which peak candidate fibers contribute to the light in each fiber
        :type neighbour_indices: np.array
        :return: Function that predicts charges in fibers, can then be used in fit.
        :rtype: typing.Callable
        """

        N_FIBERS = len(fiber_positions)
        
        peak_candidates = [i for i in range(N_FIBERS) if fiber_charges[i] > self._peak_candidate_threshold]

        N_PEAK_CANDIDATES = len(peak_candidates)

        p0 = [2.0 * fiber_charges[i] for i in range(N_PEAK_CANDIDATES)]
        p0.append(1.0) # <- the width param

        def _laplace_fn(x_data=None, *params, fix_width:float=self._fix_width):

            amplitudes = np.array(params[:N_PEAK_CANDIDATES])

            if fix_width is None:
                width      = params[N_PEAK_CANDIDATES]
            else:
                width = fix_width

            ret_charges = np.zeros(N_FIBERS)

            for fiber_index in range(N_FIBERS):
                    
                    neighbour_amplitudes = amplitudes[neighbour_indices[fiber_index]]

                    l = neighbour_amplitudes * laplace.pdf( neighbour_distances[fiber_index] / width ) / width

                    ret_charges[fiber_index] = np.sum(l)

            return ret_charges

        return _laplace_fn, p0


class PeakFinder2D(ModuleBase):
    """Finds peaks in raw fiber hits and performs position corrections
    """

    def _help(self) -> str:
        return """
This module tries to find peaks in raw 2D hits using a very simple algorithm

Each hit is checked in turn, it is considered to be a peak if in any of the 8 
directions (up, down, left right and each diagonal). It is the highest point in its local
neighbourhood.
"""
    
    def __init__(self):
        """Initialiser
        """

        super().__init__()
        
        self.requirements = ["x_fiber_hits", "y_fiber_hits", "z_fiber_hits"]
        self.outputs = [
            "x_fiber_hits", "y_fiber_hits", "z_fiber_hits",
            "x_peak_hits", "y_peak_hits", "z_peak_hits",
            "unused_x_hits", "unused_y_hits", "unused_z_hits",
            "laplace_x_peaks", "laplace_y_peaks", "laplace_z_peaks"
        ]

    def _initialise(self):

        self._pdf = matplotlib.backends.backend_pdf.PdfPages("PeakFinder2D-plots.pdf")
        self._cluster_pdf = matplotlib.backends.backend_pdf.PdfPages("PeakFinder2D-unused-hit-cluster-plots.pdf")

        self.DBSCAN_args = json.loads(self.args.DBSCAN_args)
        self.laplace_fit_args = json.loads(self.args.laplace_fit_args)

        self._clusterer = DBSCAN(**self.DBSCAN_args)
        self._laplace_fitter = LaplaceFitter(**self.laplace_fit_args, make_plots=self.args._make_plots)

        self._peak_prominance_threshold = self.args.peak_prominance_threshold
        self._peak_candidate_weight_threshold = self.args.peak_candidate_weight_threshold
        self._fit_blobs = self.args.fit_blobs
        self._make_plots = self.args._make_plots

    def _setup_cli_options(self, parser):
        
        parser.add_argument(
            "--peak-prominance-threshold", 
            help="For a hit to be considered a 'simple' peak, its neighbours must have smaller charge than peak_prominance_threshold * hit charge. 1.0 is most general, smaller values mean only sharper peaks get accepted.", 
            required = False, default = 1.0, type = float,
        )
        parser.add_argument(
            "--peak_candidate_weight_threshold", 
            help="For a hit to be considered a peak candidate, it must have at least this weight.", 
            required = False, default = 0.0, type = float,
        )
        parser.add_argument(
            "--fit-blobs", 
            help="Whether we should try to fit peaks in the hits left over from the initial simple peak finding pass.", 
            action='store_true'
        )
        parser.add_argument(
            "--make-plots", 
            help="Whether to make debug plots.", 
            action='store_true'
        )
        ## Note these are passed as a string but will later be converted to dictionaries
        parser.add_argument(
            "--DBSCAN-args", 
            help="arguments to pass to the DBSCAN algorithm used to build 'blob' clusters.", 
            required = False, default = "{\"eps\": 10.5}", type = str
        )
        parser.add_argument(
            "--laplace_fit_args", 
            help="parameters to pass to the :class:`LaplaceFitter` that is used to fit peaks in blobs", 
            required = False, default = "{}", type = str
        )
        
    def _finalise(self):
        """Tidy up and close open pdfs
        """
        self._pdf.close()
        self._cluster_pdf.close()
        self._laplace_fitter.finalise()

    def _process(self, event: Event) -> None:
        """Perform the peak finding

        :return: peak hits in each projection
        :rtype: typing.Tuple[typing.List['Hit2D']]
        """
        
        x_fiber_hits: typing.List['Hit2D'] = event["x_fiber_hits"]
        y_fiber_hits: typing.List['Hit2D'] = event["y_fiber_hits"]
        z_fiber_hits: typing.List['Hit2D'] = event["z_fiber_hits"]

        x_peak_hits = list()
        y_peak_hits = list()
        z_peak_hits = list()

        laplace_x_peak_hits = list()
        laplace_y_peak_hits = list()
        laplace_z_peak_hits = list()

        x_used = set()
        y_used = set()
        z_used = set()

        x_unused = set()
        y_unused = set()
        z_unused = set()

        ## this is vile :(
        for fiber_hits, peak_hits, laplace_peak_hits, used, unused in zip(
            [x_fiber_hits, y_fiber_hits, z_fiber_hits],
            [x_peak_hits, y_peak_hits, z_peak_hits],
            [laplace_x_peak_hits, laplace_y_peak_hits, laplace_z_peak_hits],
            [x_used, y_used, z_used],
            [x_unused, y_unused, z_unused],
        ):
            
            if fiber_hits is None:
                continue
            
            if fiber_hits is x_fiber_hits:
                u, v = "y", "z"
            elif fiber_hits is y_fiber_hits:
                u, v = "x", "z"
            elif fiber_hits is z_fiber_hits:
                u, v = "x", "y"
                
            _peaks, _used, _unused = self._find_2d_peaks(fiber_hits, u, v)

            # add peaks to the outer list
            for p in _peaks:
                peak_hits.append(p)
            for used_hit in _used:
                used.add(used_hit)
            for unused_hit in _unused:
                unused.add(unused_hit)

            if self._fit_blobs:
                ## try and fit the unused hits
                unused_clusters = self._cluster_hits(list(_unused), u, v)
                print(f"N {u}{v} clusters: {len(unused_clusters)}")
                for cluster in unused_clusters:
                    _laplace_peaks = self._laplace_fitter(cluster, u, v)
                
                    for p in _laplace_peaks:
                        laplace_peak_hits.append(p)

        ### make plot of the hits, colour coded depending on if they have been used
        if self._make_plots:
            fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(5, 5))
            fig.suptitle("Homo-FGD Hits - Peaks")

            make_corner_plot_fiber_hits(
                fig,
                axs, 
                list(x_used),
                list(y_used),
                list(z_used),
                colour_override="g"
            )

            make_corner_plot_fiber_hits(
                fig,
                axs, 
                x_peak_hits,
                y_peak_hits,
                z_peak_hits,
                colour_override="b"
            )

            make_corner_plot_fiber_hits(
                fig,
                axs, 
                list(x_unused),
                list(y_unused),
                list(z_unused),
                colour_override="r"
            )

            make_corner_plot_fiber_hits(
                fig,
                axs, 
                laplace_x_peak_hits ,
                laplace_y_peak_hits,
                laplace_z_peak_hits,
                colour_override="m"
            )

            self._pdf.savefig(fig)
            plt.close(fig)


        event.add_data("x_peak_hits", x_peak_hits)
        event.add_data("y_peak_hits", y_peak_hits)
        event.add_data("z_peak_hits", z_peak_hits)
        
        event.add_data("unused_x_hits", x_unused)
        event.add_data("unused_y_hits", y_unused)
        event.add_data("unused_z_hits", z_unused)
        
        event.add_data("laplace_x_peaks", laplace_x_peak_hits)
        event.add_data("laplace_y_peaks", laplace_y_peak_hits)
        event.add_data("laplace_z_peaks", laplace_z_peak_hits)

        event.add_data("x_fiber_hits", x_peak_hits)
        event.add_data("y_fiber_hits", y_peak_hits)
        event.add_data("z_fiber_hits", z_peak_hits)

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
        
    def _find_2d_peaks(
            self,
            fiber_hits:typing.List['Hit2D'], 
            u:str, v:str,
            neighbourhood_dist:float = 15.1, extended_neighbourhood_dist:float = 30.1,
            u_pitch:float=10.0, v_pitch:float=10.0
        ) -> typing.Tuple[typing.List['Hit2D'], typing.Set['Hit2D'], typing.Set['Hit2D']]:
        
        # keep track of what hits have been used and what havent
        # initialise the unused set to all the input hits, will move them across as we go
        used_hits = set()
        unused_hits = set(fiber_hits)

        peak_hits = list()

        neighbour_algo = NearestNeighbors(radius = neighbourhood_dist)
        extended_neighbour_algo = NearestNeighbors(radius = extended_neighbourhood_dist)

        data = np.array([[getattr(hit, u), getattr(hit, v)] for hit in fiber_hits])
        
        _, indices = neighbour_algo.fit(data).radius_neighbors(data)
        _, extended_indices = extended_neighbour_algo.fit(data).radius_neighbors(data)

        for hit_id in reversed(sorted(range(len(fiber_hits)), key=lambda x: fiber_hits.__getitem__(x).weight)):
            
            hit = fiber_hits[hit_id]
            charge = hit.weight
            direction = {u: 0.0, v: 0.0}

            if charge < self._peak_candidate_weight_threshold:
                continue
            
            ## modify charge based on the prominence threshold supplied
            modified_charge = charge * self._peak_prominance_threshold

            neighbourhood = [fiber_hits[id] for id in indices[hit_id]]
            extended_neighbourhood = [fiber_hits[id] for id in extended_indices[hit_id]]

            u_line_hits = [neighbour for neighbour in neighbourhood if getattr(neighbour, v) == getattr(hit, v)]
            v_line_hits = [neighbour for neighbour in neighbourhood if getattr(neighbour, u) == getattr(hit, u)]
            
            u_line_charges = np.array([neighbour.weight for neighbour in u_line_hits])
            v_line_charges = np.array([neighbour.weight for neighbour in v_line_hits])

            extended_u_line_hits = [neighbour for neighbour in extended_neighbourhood if getattr(neighbour, v) == getattr(hit, v)]
            extended_v_line_hits = [neighbour for neighbour in extended_neighbourhood if getattr(neighbour, u) == getattr(hit, u)]

            u_info_hits = [hit]
            v_info_hits = [hit]

            is_peak = False
            if np.sum(u_line_charges < modified_charge) >= 2:
                is_peak = True
                local_peak_hits = self._find_peak_hits(hit, extended_u_line_hits, u)
                for h in local_peak_hits:
                    u_info_hits.append(h)

                direction[v] = 1

            if np.sum(v_line_charges < modified_charge) >= 2:
                is_peak = True
                local_peak_hits = self._find_peak_hits(hit, extended_v_line_hits, v)
                for h in local_peak_hits:
                    v_info_hits.append(h)

                if direction[v] == 1:
                    direction[u] = 0
                    direction[v] = 0
                else:
                    direction[u] = 1

            ## If it's not already a peak, check if it's a diagonal peak
            if not is_peak:
                diag_uv_line_hits = self._get_diagonal_neighbours(hit, neighbourhood, u, v, u_pitch, v_pitch)
                diag_vu_line_hits = self._get_diagonal_neighbours(hit, neighbourhood, u, v, u_pitch, v_pitch, diagonal_sign=-1)

                diag_uv_line_charges = np.array([neighbour.weight for neighbour in diag_uv_line_hits])
                diag_vu_line_charges = np.array([neighbour.weight for neighbour in diag_vu_line_hits])

                extended_diag_uv_line_hits = self._get_diagonal_neighbours(hit, extended_neighbourhood, u, v, u_pitch, v_pitch)
                extended_diag_vu_line_hits = self._get_diagonal_neighbours(hit, extended_neighbourhood, u, v, u_pitch, v_pitch, diagonal_sign=-1)

                if np.sum(diag_uv_line_charges < modified_charge) >= 2:
                    is_peak = True
                    
                    direction[u] = -1
                    direction[v] = 1

                    local_peak_hits = self._find_peak_hits(hit, extended_diag_uv_line_hits, u)
                    for h in local_peak_hits:
                        u_info_hits.append(h)
                        v_info_hits.append(h)

                if np.sum(diag_vu_line_charges < modified_charge) >= 2:
                    is_peak = True
                    local_peak_hits = self._find_peak_hits(hit, extended_diag_vu_line_hits, u)
                    for h in local_peak_hits:
                        u_info_hits.append(h)
                        v_info_hits.append(h)

                    direction[u] = 1
                    direction[v] = 1
                
            if is_peak:
                new_hit = Hit2D.copy(hit)

                new_hit.set_is_peak({u: direction[v], v: direction[u]})

                new_hit.set_direction(direction)

                setattr(new_hit, u, Hit2D.get_mean_pos(u_info_hits, u))
                setattr(new_hit, v, Hit2D.get_mean_pos(v_info_hits, v))

                peak_hits.append(new_hit)

                # keep track of what has been used and what not
                used_hits.add(hit)
                if hit in unused_hits:
                    unused_hits.remove(hit)
                
                for u_hit in u_info_hits:
                    used_hits.add(u_hit)
                
                    if u_hit in unused_hits:
                        unused_hits.remove(u_hit)
                
                for v_hit in v_info_hits:
                    used_hits.add(v_hit)
                
                    if v_hit in unused_hits:
                        unused_hits.remove(v_hit)

        return peak_hits, used_hits, unused_hits

    
    def _find_peak_hits(
        self,
        main_hit:'Hit2D',
        line_hits:list['Hit2D'],
        direction:str
    ) -> typing.List['Hit2D']:
        """Finds hits in a list whose charges are monotonically decreasing relative to some central hit

        e.g. running on 
        
              ↓       ____
             ___     /    \
            /   \___/      \
        ___/                \ 

        where the arrow indicates the "main hit"
        would give
        
             ___
            /   \__
        ___/        

        :param main_hit: The central hit that defines the summit of the peak
        :type main_hit: Hit2D
        :param line_hits: The hits to be searched (should include the main hit)
        :type line_hits: list['Hit2D']
        :param direction: The direction along the "line" of hits, should be "x", "y" or "z"
        :type direction: str
        :return: list of hits that belong to the same peak as the main hit
        :rtype: typing.List['Hit2D']
        """
        
        # print()
        # print("###########################")
        # print(f"direction = {direction}")
        # print(f"main hit: {main_hit}")
        # input()
        ## will be the list of all hits that are part of the peak
        ret_list = []

        ## sort by the direction coordinate so that the main hit will be in the middle
        line_hits.sort(key = lambda h: getattr(h, direction))

        # print("sorted hits:")
        # for sorted_hit in line_hits:
        #     main_arrow = ""
        #     if sorted_hit is main_hit:
        #         main_arrow = " <- main"
        #     print("  " + str(sorted_hit) + main_arrow)

        ## get the index of the main hit in the list
        main_hit_position = 0
        for test_hit in line_hits:
            if test_hit is main_hit:
                break
            main_hit_position += 1
        
        current_charge = float(main_hit.weight)
        for i in range(main_hit_position - 1, -1, -1):

            if line_hits[i].weight > current_charge:
                break

            current_charge = float(line_hits[i].weight)
            # input(f"adding hit : {line_hits[i]}")
            ret_list.append(line_hits[i])

        current_charge = float(main_hit.weight)
        for i in range(main_hit_position + 1, len(line_hits)):

            if line_hits[i].weight > current_charge:
                break

            current_charge = float(line_hits[i].weight)
            # input(f"adding hit : {line_hits[i]}")
            ret_list.append(line_hits[i])

        return ret_list


    def _get_diagonal_neighbours(
            self, 
            hit:'Hit2D', 
            neighbourhood:typing.List['Hit2D'], 
            u:str, v:str, 
            u_pitch:float, v_pitch:float, 
            diagonal_sign = +1
        ) -> typing.List['Hit2D']:
        """Gets neighbours of a hit along a diagonal line

        :param hit: The main hit
        :type hit: Hit2D
        :param neighbourhood: The hits to search for diagonal neighbours in
        :type neighbourhood: typing.List['Hit2D']
        :param u: the u direction, should be either "x", "y" or "z"
        :type u: str
        :param v: the v direction, should be either "x", "y" or "z"
        :type v: str
        :param u_pitch: fiber pitch in the u direction
        :type u_pitch: float
        :param v_pitch: fiber pitch in the v direction
        :type v_pitch: float
        :param diagonal_sign: The gradient of the diagonal, defaults to +1
        :type diagonal_sign: int, optional
        :return: Hits from the neighbourhood that lie along the specified diagonal
        :rtype: typing.List['Hit2D']
        """

        ret_list = list()
        for neighbour in neighbourhood:
            u_dist = (getattr(neighbour, u) - getattr(hit, u)) / u_pitch
            v_dist = (getattr(neighbour, v) - getattr(hit, v)) / v_pitch
            
            if abs(u_dist - diagonal_sign * v_dist) < 0.0001:

                ret_list.append(neighbour)

        return ret_list


class HesseRidgeDetection2D(ModuleBase):
    """Performs "ridge detection" using the Hessian of a 2D image of the detector
    """

    def __init__(
            self
        ):

        super().__init__()

        self.requirements = ["x_fiber_hits", "y_fiber_hits", "z_fiber_hits"]
        self.outputs = [
            "x_fiber_hits", "y_fiber_hits", "z_fiber_hits",
            "x_peak_hits", "y_peak_hits", "z_peak_hits",
            "unused_x_hits", "unused_y_hits", "unused_z_hits"
        ]

    def _initialise(self) -> None:
        
        self._pitch = self.args.pitch

        self._min_charge = self.args.min_charge
        self._max_pos_curvature = self.args.max_positive_curvature
        self._min_negative_curvature = self.args.min_negative_curvature

        self._debug_pdf = None
        self._pdf = None

        self._make_plots = self.args.make_plots
        self._make_debug_plots = self.args.make_debug_plots
        
        if self.args.make_debug_plots:
            self._debug_pdf = matplotlib.backends.backend_pdf.PdfPages(self.args.debug_plot_file_name)
        if self.args.make_plots:
            self._pdf = matplotlib.backends.backend_pdf.PdfPages(self.args.plot_file_name)

    def _setup_cli_options(self, parser):

        ## TODO: Make geometry manager class to store this kind of thing
        parser.add_argument(
            "--pitch", 
            help="The fiber pitch", 
            required = False, default = 10.0, type = float,
        )
        parser.add_argument(
            "--min-charge", 
            help="The minimum charge that a hit must have to be considered a peak hit", 
            required = False, default = 50.0, type = float,
        )
        parser.add_argument(
            "--max-positive-curvature", 
            help="The maximum local positive curvature that is allowed in the neighbourhood of a hit for it to be considered a peak. If this is 0.0 then only strict local maximum points may be peaks, the larger it is, the more extreme 'sadle points' are allowed", 
            required = False, default = 100.0, type = float,
        )
        parser.add_argument(
            "--min-negative-curvature", 
            help="The minimum negative or 'downwards' curvature that is required in the neighbourhood of a hit for it to be considered a peak. The closer this is to 0.0, the more shallow peaks are allowed, the higher it is, the sharper the peaks must be", 
            required = False, default = 50.0, type = float,
        )
        parser.add_argument(
            "--make-plots", 
            help="Whether to make basic plots", 
            action='store_true'
        )
        parser.add_argument(
            "--plot-file-name", 
            help="Name of file to save plots to if --make-plots option is true", 
            required = False, default = "Hesse-example-plots.pdf", type = str,
        )
        parser.add_argument(
            "--make-debug-plots", 
            help="Whether to make debug plots", 
            action='store_true'
        )
        parser.add_argument(
            "--debug-plot-file-name", 
            help="Name of file to save debug plots to if --make-debug-plots option is true", 
            required = False, default = "Hesse-debug-plots.pdf", type = str,
        )
        
    def _gradient(self, hist: np.array, normalise: bool = False) -> typing.Tuple[np.array]:
        """Calculate the gradient of an input image using central finite difference

        :param hist: Histogram you want the gradient of
        :type hist: np.array
        :param normalise: Do per-bin normalisation to the cantral value
        :type normalise: bool
        :return: arrays du and dv containing derivatives wrt u and v
        :rtype: Tuple[np.array]
        """

        assert len(hist.shape) == 2, f"Wrong number of dimensions, can only do 2D but got {len(hist.shape)}!"

        u_grad = np.gradient(hist, axis = 0)
        v_grad = np.gradient(hist, axis = 1)

        if normalise:
            u_grad = np.divide(u_grad, hist, where = hist != 0.0)
            v_grad = np.divide(v_grad, hist, where = hist != 0.0)

        return u_grad, v_grad
    
    def _hessian(self, hist: np.array, normalise: bool = False) -> typing.Tuple[np.array]:
        """Calculate the hessian matrix of an input image using central finite difference

        :param hist: Histogram you want the hessian of
        :type hist: np.array
        :param normalise: Do per-bin normalisation to the cantral value
        :type normalise: bool
        :return: arrays huu, hvv, huv, hvu containing each of the necessary double derivatives
        :rtype: Tuple[np.array]
        """

        du, dv = self._gradient(hist)

        huu, huv = self._gradient(du)
        hvu, hvv = self._gradient(dv)

        if normalise:
            huu = np.divide(huu, hist, where = hist != 0.0)
            huv = np.divide(huv, hist, where = hist != 0.0)
            hvv = np.divide(hvv, hist, where = hist != 0.0)
            hvu = np.divide(hvu, hist, where = hist != 0.0)

        return huu, hvv, huv, hvu
    
    def _hess_eigen(self, hist: np.array) -> typing.Tuple[np.array]:
        """Get the eigenvalues and vectors of the hessian matrix of an input image

        :param hist: The input image
        :type hist: np.array
        :return: The Hessian eigenvalues and eigenvectors
        :rtype: typing.Tuple[np.array]
        """

        huu, hvv, huv, hvu = self._hessian(hist)
        
        hess_eigenvals = np.ndarray((2, *huu.shape))
        hess_eigenvecs = np.ndarray((2, 2, *huu.shape))
        
        for dim0 in range(0, huu.shape[-2]):
            for dim1 in range(0, huu.shape[-1]):

                if(hist[dim0, dim1] == 0.0):
                    hess_eigenvals[:, dim0, dim1] = 0.0
                    hess_eigenvecs[:, :, dim0, dim1] = 0.0
                
                else:
                    evals, evecs = eig(
                        np.array(
                            [
                                [huu[dim0, dim1], hvu[dim0, dim1]],
                                [huv[dim0, dim1], hvv[dim0, dim1]]
                            ]
                        )
                    )

                    if (np.any(np.imag(evals) != 0.0)):
                        print(f"WARNING: complex eigenvalues found in Hessian!!!")

                    hess_eigenvals[:, dim0, dim1] = np.real(evals[:])
                    hess_eigenvecs[:, :, dim0, dim1] = np.real(evecs[:, :])

        return hess_eigenvals, hess_eigenvecs

    def _compute_ridgeness(self, hist: np.array, hess_eigenvals: np.array) -> np.array:
        """Compute the "ridgeness" score for each pixel in an input image

        :param hist: The 2D input image
        :type hist: np.array
        :param hess_eigenvals: The eigenvalues of the hessian for the image (computed using the `_hess_eigen()` method)
        :type hess_eigenvals: np.array
        :return: The 2D array of ridgeness scores
        :rtype: np.array
        """

        ridgeness = np.zeros(shape=(*hess_eigenvals.shape[1:], 1))
        for dim0 in range(0, hess_eigenvals.shape[-2]):
            for dim1 in range(0, hess_eigenvals.shape[-1]):

                if (
                    hist[dim0, dim1] > self._min_charge and
                    np.all(hess_eigenvals[:, dim0, dim1] < self._max_pos_curvature) and
                    -np.min(hess_eigenvals[:, dim0, dim1]) > self._min_negative_curvature
                ):
                    
                    ridgeness[dim0, dim1, 0] = -np.min(hess_eigenvals[:, dim0, dim1])

        return ridgeness

        
    def _process(self, event:Event):
        """ Perform Hough transform on an event and save the result to a given file

        :param event: Object describing the hits in an event
        :type event: Event
        """

        x_fiber_hits = event["x_fiber_hits"]
        y_fiber_hits = event["y_fiber_hits"]
        z_fiber_hits = event["z_fiber_hits"]

        x_peak_hits = list()
        y_peak_hits = list()
        z_peak_hits = list()

        x_unused = set()
        y_unused = set()
        z_unused = set()

        fig, axs = None, None
        if self._make_plots:
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
            fig.suptitle("Hessian Filter")

        for fiber_hits, u_name, v_name, peak_hits, unused_hits, ax_ids in zip(
            [
                x_fiber_hits, 
                y_fiber_hits, 
                z_fiber_hits
            ],
            ["z", "x", "x"],
            ["y", "z", "y"],
            [
                x_peak_hits,
                y_peak_hits,
                z_peak_hits
            ],
            [
                x_unused,
                y_unused,
                z_unused
            ],
            [[1,1], [0,0], [1,0]]
        ):
            
            u_values = [getattr(hit, u_name) for hit in fiber_hits]
            v_values = [getattr(hit, v_name) for hit in fiber_hits]

            u_bins = np.arange(start=min(u_values) - 3.0 * self._pitch / 2.0, stop=max(u_values) + 5.0 * self._pitch / 2.0, step = self._pitch) 
            v_bins = np.arange(start=min(v_values) - 3.0 * self._pitch / 2.0, stop=max(v_values) + 5.0 * self._pitch / 2.0, step = self._pitch) 

            hist, _, _ = np.histogram2d(
                u_values, v_values,
                bins = (u_bins, v_bins), weights=[hit.weight for hit in fiber_hits]
            )
            
            ## get eigenvalues and eigenvectors of Hessian
            hess_eigenvals, hess_eigenvecs = self._hess_eigen(hist)

            ## compute the ridgeness score
            ridgeness = self._compute_ridgeness(hist, hess_eigenvals)

            if self._debug_pdf is not None:
                self._do_make_debug_plots(
                    np.transpose(hist, axes=(1,0)),
                    u_name, v_name
                )
                
            if self._pdf is not None:
                self._make_plot(
                    ridgeness,
                    hess_eigenvals,
                    hess_eigenvecs,
                    ax = axs[ax_ids[0], ax_ids[1]]
                )

            ## now make the peak hits
            ## loop over the fiber hits, check if the "pixel" it falls into is a ridge, if so save it as a peak hit
            for hit in fiber_hits:

                u = getattr(hit, u_name)
                v = getattr(hit, v_name)

                u_bin = np.digitize(u, u_bins)
                v_bin = np.digitize(v, v_bins)

                ## have already applied all our conditions when calculating ridgeness and don't fill it if it fails
                ## so here we just need to check if it's not 0
                if ridgeness[u_bin -1, v_bin -1] > 0.0:

                    peak_hits.append(hit)

                    ## get the eigenvector corresponding to the smallest eigenvalue
                    ## this will be the one that points along the "ridge"
                    max_eval_id = np.argmax(hess_eigenvals[:, u_bin - 1, v_bin - 1])
                    evec = hess_eigenvecs[:, max_eval_id, u_bin - 1, v_bin - 1]

                    hit.set_direction({u_name: evec[0], v_name: evec[1]})
                    hit.set_is_peak({u_name: True, v_name: True})

                else:
                    unused_hits.add(hit)

        if self._make_plots:
        
            make_corner_plot_fiber_hits(
                fig,
                axs, 
                [],
                [],
                [],
                label = ("x [pixel]", "y [pixel]", "z [pixel]")
            )

            self._pdf.savefig(fig)
            plt.close(fig)

        event.add_data("x_fiber_hits", x_peak_hits)
        event.add_data("y_fiber_hits", y_peak_hits)
        event.add_data("z_fiber_hits", z_peak_hits)

        event.add_data("x_peak_hits", x_peak_hits)
        event.add_data("y_peak_hits", y_peak_hits)
        event.add_data("z_peak_hits", z_peak_hits)
        
        event.add_data("unused_x_hits", x_unused)
        event.add_data("unused_y_hits", y_unused)
        event.add_data("unused_z_hits", z_unused)
        
    def _finalise(self):

        if self._debug_pdf is not None:
            self._debug_pdf.close()
        
        if self._pdf is not None:
            self._pdf.close()

    def _make_plot(
            self, 
            ridgeness: np.array, 
            hess_eigenvals: np.array, 
            hess_eigenvecs: np.array,
            ax: plt.axis
        ) -> None:
        """Make plot of the ridgeness score of each pixel (fiber) with direction of the detected ridges overlaid

        :param ridgeness: The 2D array defining the ridgeness score for each pixel
        :type ridgeness: np.array
        :param hess_eigenvals: The 3D array of the eigenvalues of the hessian at each pixel
        :type hess_eigenvals: np.array
        :param hess_eigenvecs: The 3D array of the eigenvectors of the hessian at each pixel
        :type hess_eigenvecs: np.array
        :param u_name: The label of the u direction ("x", "y" or "z")
        :type u_name: str
        :param v_name: The label of the v direction ("x", "y" or "z")
        :type v_name: str
        :param ax: The pyplot axis object to plot to
        :type ax: plt.axis
        """
        
        mappable = ax.imshow(np.transpose(ridgeness, axes=(1,0,2)), cmap=plt.get_cmap("gray"), origin='lower')
        plt.colorbar(mappable)

        for dim0 in range(0, hess_eigenvecs.shape[-2]):
            for dim1 in range(0, hess_eigenvecs.shape[-1]):

                if ridgeness[dim0, dim1] > 0.0:
                    max_eval_id = np.argmax(hess_eigenvals[:, dim0, dim1])

                    ax.plot(
                        (
                            dim0 - 0.5 * hess_eigenvecs[0, max_eval_id, dim0, dim1],
                            dim0 + 0.5 * hess_eigenvecs[0, max_eval_id, dim0, dim1]
                        ),
                        (
                            dim1 - 0.5 * hess_eigenvecs[1, max_eval_id, dim0, dim1],
                            dim1 + 0.5 * hess_eigenvecs[1, max_eval_id, dim0, dim1]
                        ), 
                        c = "r",
                        linewidth = 0.25
                    )

    def _do_make_debug_plots(
        self,
        hist: np.array,
        u_name: str, v_name: str
    ):
        """
        Make detailed plots of values used in the Hesse ridge detection algorithm
        
        :param hist: The input image
        :type hist: np.array
        :param u_name: The label of the u direction ("x", "y" or "z")
        :type u_name: str
        :param v_name: The label of the v direction ("x", "y" or "z")
        :type v_name: str
        """

        huu, hvv, huv, hvu = self._hessian(hist)
        fig, ax = plt.subplots(1, 8, figsize=(50, 10))
        ax[0].imshow(hist, cmap=plt.get_cmap("coolwarm"), origin='lower')
        ax[0].set_title("Original Event")

        ax[1].imshow(huu, cmap=plt.get_cmap("gray"), origin='lower')
        ax[1].set_title(f"H_{u_name}{u_name}")
        ax[2].imshow(huv, cmap=plt.get_cmap("gray"), origin='lower')
        ax[2].set_title(f"H_{u_name}{v_name}")
        ax[3].imshow(hvu, cmap=plt.get_cmap("gray"), origin='lower')
        ax[3].set_title(f"H_{v_name}{u_name}")
        ax[4].imshow(hvv, cmap=plt.get_cmap("gray"), origin='lower')
        ax[4].set_title(f"H_{v_name}{v_name}")

        du, dv = self._gradient(hist)
        ax[5].imshow(du, cmap=plt.get_cmap("gray"), origin='lower')
        ax[5].set_title(f"D_{u_name}")
        ax[6].imshow(dv, cmap=plt.get_cmap("gray"), origin='lower')
        ax[6].set_title(f"D_{v_name}")

        hess_eigenvals, _ = self._hess_eigen(hist)
        ridgeness = self._compute_ridgeness(hist, hess_eigenvals)
        ax[7].imshow(ridgeness, cmap=plt.get_cmap("gray"), origin='lower')
        ax[7].set_title("Hessian Filter")

        self._debug_pdf.savefig(fig)
        plt.close(fig)
        

class HesseRidgeDetection3D(ModuleBase):
    """Performs "ridge detection" using the Hessian of a 3D image of the detector
    """

    def __init__(
            self
        ):

        super().__init__()

        self.requirements = ["3d_hits"]
        self.outputs = [
            "3d_hits",
            "3d_peak_hits",
            "unused_3d_hits"
        ]

    def _initialise(self) -> None:
        
        self._pitch = self.args.pitch

        self._min_charge = self.args.min_charge
        self._max_pos_curvature = self.args.max_positive_curvature
        self._min_negative_curvature = self.args.min_negative_curvature

        self._make_plots = self.args.make_plots
        self._make_debug_plots = self.args.make_debug_plots

        self._debug_pdf = None
        self._pdf = None

        if self._make_debug_plots:
            self._debug_pdf = matplotlib.backends.backend_pdf.PdfPages(self.args.debug_plot_file_name)
        if self._make_plots:
            self._pdf = matplotlib.backends.backend_pdf.PdfPages(self.args.plot_file_name)

    def _setup_cli_options(self, parser):

        ## TODO: Make geometry manager class to store this kind of thing
        parser.add_argument(
            "--pitch", 
            help="The fiber pitch", 
            required = False, default = 10.0, type = float,
        )
        parser.add_argument(
            "--min-charge", 
            help="The minimum charge that a hit must have to be considered a peak hit", 
            required = False, default = 50.0, type = float,
        )
        parser.add_argument(
            "--max-positive-curvature", 
            help="The maximum local positive curvature that is allowed in the neighbourhood of a hit for it to be considered a peak. If this is 0.0 then only strict local maximum points may be peaks, the larger it is, the more extreme 'sadle points' are allowed", 
            required = False, default = 100.0, type = float,
        )
        parser.add_argument(
            "--min-negative-curvature", 
            help="The minimum negative or 'downwards' curvature that is required in the neighbourhood of a hit for it to be considered a peak. The closer this is to 0.0, the more shallow peaks are allowed, the higher it is, the sharper the peaks must be", 
            required = False, default = 50.0, type = float,
        )
        parser.add_argument(
            "--make-plots", 
            help="Whether to make basic plots", 
            action='store_true'
        )
        parser.add_argument(
            "--plot-file-name", 
            help="Name of file to save plots to if --make-plots option is true", 
            required = False, default = "Hesse-example-plots.pdf", type = str,
        )
        parser.add_argument(
            "--make-debug-plots", 
            help="Whether to make debug plots", 
            action='store_true'
        )
        parser.add_argument(
            "--debug-plot-file-name", 
            help="Name of file to save debug plots to if --make-debug-plots option is true", 
            required = False, default = "Hesse-debug-plots.pdf", type = str,
        )
        
    def _gradient(self, hist: np.array, normalise: bool = False) -> typing.Tuple[np.array]:
        """Calculate the gradient of an input image using central finite difference

        :param hist: Histogram you want the gradient of
        :type hist: np.array
        :param normalise: Do per-bin normalisation to the cantral value
        :type normalise: bool
        :return: arrays du and dv containing derivatives wrt u and v
        :rtype: Tuple[np.array]
        """

        assert len(hist.shape) == 3, f"Wrong number of dimensions, can only do 3D but got {len(hist.shape)}!"

        u_grad = np.gradient(hist, axis = 0)
        v_grad = np.gradient(hist, axis = 1)
        w_grad = np.gradient(hist, axis = 2)
        
        if normalise:
            u_grad = np.divide(u_grad, hist, where = hist != 0.0)
            v_grad = np.divide(v_grad, hist, where = hist != 0.0)
            w_grad = np.divide(w_grad, hist, where = hist != 0.0)

        return u_grad, v_grad, w_grad
    
    def _hessian(self, hist: np.array, normalise: bool = False) -> typing.Tuple[np.array]:
        """Calculate the hessian matrix of an input image using central finite difference

        :param hist: Histogram you want the hessian of
        :type hist: np.array
        :param normalise: Do per-bin normalisation to the cantral value
        :type normalise: bool
        :return: arrays huu, hvv, huv, hvu containing each of the necessary double derivatives
        :rtype: Tuple[np.array]
        """

        du, dv, dw = self._gradient(hist)

        huu, huv, huw = self._gradient(du)
        hvu, hvv, hvw = self._gradient(dv)
        hwu, hwv, hww = self._gradient(dw)

        if normalise:
            huu = np.divide(huu, hist, where = hist != 0.0)
            huv = np.divide(huv, hist, where = hist != 0.0)
            huw = np.divide(huw, hist, where = hist != 0.0)
            hvv = np.divide(hvv, hist, where = hist != 0.0)
            hvu = np.divide(hvu, hist, where = hist != 0.0)
            hvw = np.divide(hvw, hist, where = hist != 0.0)
            hwv = np.divide(hwv, hist, where = hist != 0.0)
            hwu = np.divide(hwu, hist, where = hist != 0.0)
            hww = np.divide(hww, hist, where = hist != 0.0)

        return huu, hvv, huw, huv, hvu, hvw, hwu, hwv, hww
    
    def _hess_eigen(self, hist: np.array) -> typing.Tuple[np.array]:
        """Get the eigenvalues and vectors of the hessian matrix of an input image

        :param hist: The input image
        :type hist: np.array
        :return: The Hessian eigenvalues and eigenvectors
        :rtype: typing.Tuple[np.array]
        """

        huu, hvv, huw, huv, hvu, hvw, hwu, hwv, hww = self._hessian(hist)
        
        hess_eigenvals = np.ndarray((3, *huu.shape))
        hess_eigenvecs = np.ndarray((3, 3, *huu.shape))
        
        for dim0 in range(0, huu.shape[-3]):
            for dim1 in range(0, huu.shape[-2]):
                for dim2 in range(0, huu.shape[-1]):

                    if(hist[dim0, dim1, dim2] == 0.0):
                        hess_eigenvals[:, dim0, dim1, dim2] = 0.0
                        hess_eigenvecs[:, :, dim0, dim1, dim2] = 0.0
                    
                    else:
                        evals, evecs = eig(
                            np.array(
                                [
                                    [huu[dim0, dim1, dim2], hvu[dim0, dim1, dim2], hwu[dim0, dim1, dim2]],
                                    [huv[dim0, dim1, dim2], hvv[dim0, dim1, dim2], hwv[dim0, dim1, dim2]],
                                    [huw[dim0, dim1, dim2], hvw[dim0, dim1, dim2], hww[dim0, dim1, dim2]]
                                ]
                            )
                        )

                        if (np.any(np.imag(evals) != 0.0)):
                            print(f"WARNING: complex eigenvalues found in Hessian!!!")

                        hess_eigenvals[:, dim0, dim1, dim2] = np.real(evals[:])
                        hess_eigenvecs[:, :, dim0, dim1, dim2] = np.real(evecs[:, :])

        return hess_eigenvals, hess_eigenvecs

    def _compute_ridgeness(self, hist: np.array, hess_eigenvals: np.array) -> np.array:
        """Compute the "ridgeness" score for each pixel in an input image

        :param hist: The 2D input image
        :type hist: np.array
        :param hess_eigenvals: The eigenvalues of the hessian for the image (computed using the `_hess_eigen()` method)
        :type hess_eigenvals: np.array
        :return: The 2D array of ridgeness scores
        :rtype: np.array
        """

        ridgeness = np.zeros(shape=(*hess_eigenvals.shape[1:], 1))
        for dim0 in range(0, hess_eigenvals.shape[-3]):
            for dim1 in range(0, hess_eigenvals.shape[-2]):
                for dim2 in range(0, hess_eigenvals.shape[-1]):

                    if (
                        hist[dim0, dim1, dim2] > self._min_charge and
                        np.all(hess_eigenvals[:, dim0, dim1, dim2] < self._max_pos_curvature) and
                        (np.sum(-hess_eigenvals[:, dim0, dim1, dim2] > self._min_negative_curvature) >= 2)
                    ):
                        
                        ridgeness[dim0, dim1, dim2, 0] = -np.min(hess_eigenvals[:, dim0, dim1, dim2])

        return ridgeness
        
    def _process(self, event:Event):
        """ Perform Hough transform on an event and save the result to a given file

        :param event: Object describing the hits in an event
        :type event: Event
        """

        hits = event["3d_hits"]
        peak_hits = list()
        unused = set()

        
        u_values = [hit.x for hit in hits]
        v_values = [hit.y for hit in hits]
        w_values = [hit.z for hit in hits]

        u_bins = np.arange(start=min(u_values) - 3.0 * self._pitch / 4.0, stop=max(u_values) + 5.0 * self._pitch / 4.0, step = self._pitch / 2.0) 
        v_bins = np.arange(start=min(v_values) - 3.0 * self._pitch / 4.0, stop=max(v_values) + 5.0 * self._pitch / 4.0, step = self._pitch / 2.0) 
        w_bins = np.arange(start=min(w_values) - 3.0 * self._pitch / 4.0, stop=max(w_values) + 5.0 * self._pitch / 4.0, step = self._pitch / 2.0) 

        hist, _ = np.histogramdd(
            (u_values, v_values, w_values),
            bins = (u_bins, v_bins, w_bins), weights=[hit.weight for hit in hits]
        )
        
        ## get eigenvalues and eigenvectors of Hessian
        hess_eigenvals, hess_eigenvecs = self._hess_eigen(hist)

        ## compute the ridgeness score
        ridgeness = self._compute_ridgeness(hist, hess_eigenvals)

        ## now make the peak hits
        ## loop over the fiber hits, check if the "pixel" it falls into is a ridge, if so save it as a peak hit
        for hit in hits:

            u = hit.x
            v = hit.y
            w = hit.z

            u_bin = np.digitize(u, u_bins)
            v_bin = np.digitize(v, v_bins)
            w_bin = np.digitize(w, w_bins)

            ## have already applied all our conditions when calculating ridgeness and don't fill it if it fails
            ## so here we just need to check if it's not 0
            if ridgeness[u_bin -1, v_bin -1, w_bin - 1] > 0.0:
                peak_hits.append(hit)

                ## get the eigenvector corresponding to the smallest eigenvalue
                ## this will be the one that points along the "ridge"
                max_eval_id = np.argmax(hess_eigenvals[:, u_bin - 1, v_bin - 1, w_bin - 1])
                evec = hess_eigenvecs[:, max_eval_id, u_bin - 1, v_bin - 1, w_bin - 1]

                hit.set_direction({"x": evec[0], "y": evec[1], "z": evec[2]})

            else:
                unused.add(hit)

        event.add_data("3d_hits", peak_hits)
        event.add_data("3d_peak_hits", peak_hits)
        event.add_data("unused_3d_hits", unused)
        
    def _finalise(self):

        if self._debug_pdf is not None:
            self._debug_pdf.close()
        
        if self._pdf is not None:
            self._pdf.close()
