from liquidreco.plotting import make_corner_plot, make_corner_plot_fiber_hits

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import laplace
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

import typing

from liquidreco.hit import Hit, Hit2D, Hit3D

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


class PeakFinder2D:
    """Finds peaks in raw fiber hits and performs position corrections
    """
    
    def __init__(
            self, 
            peak_prominance_threshold:float,
            make_plots:bool = False,
            DBSCAN_args = {"eps": 10.5},
            laplace_fit_args = {},
        ):

        self._pdf = matplotlib.backends.backend_pdf.PdfPages("PeakFinder2D-plots.pdf")
        self._cluster_pdf = matplotlib.backends.backend_pdf.PdfPages("PeakFinder2D-unused-hit-cluster-plots.pdf")

        self._peak_prominance_threshold = peak_prominance_threshold

        self._make_plots = make_plots

        self._clusterer = DBSCAN(**DBSCAN_args)

        self._laplace_fitter = LaplaceFitter(**laplace_fit_args, make_plots=make_plots)

    def finalise(self):
        """Tidy up and close open pdfs
        """
        self._pdf.close()
        self._cluster_pdf.close()
        self._laplace_fitter.finalise()

    def __call__(
        self, 
        x_fiber_hits:typing.List['Hit2D'] = None,
        y_fiber_hits:typing.List['Hit2D'] = None,
        z_fiber_hits:typing.List['Hit2D'] = None
    ) -> typing.Tuple[typing.List['Hit2D']]:
        """Perform the peak finding

        :param x_fiber_hits: Hits from x fibers (fibers orthogonal to the yz plane), defaults to None
        :type x_fiber_hits: typing.List['Hit2D'], optional
        :param y_fiber_hits: Hits from y fibers (fibers orthogonal to the xz plane), defaults to None
        :type y_fiber_hits: typing.List['Hit2D'], optional
        :param z_fiber_hits: Hits from z fibers (fibers orthogonal to the xy plane), defaults to None
        :type z_fiber_hits: typing.List['Hit2D'], optional
        :return: peak hits in each projection
        :rtype: typing.Tuple[typing.List['Hit2D']]
        """
        
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
                
            _peaks, _used, _unused = self.find_2d_peaks(fiber_hits, u, v, self._peak_prominance_threshold)

            # add peaks to the outer list
            for p in _peaks:
                peak_hits.append(p)
            for used_hit in _used:
                used.add(used_hit)
            for unused_hit in _unused:
                unused.add(unused_hit)

            print(f"used {u}{v} hits: {len(_used)}, unused {u}{v} hits: {len(_unused)}")

            unused_clusters = self.cluster_hits(list(_unused), u, v)
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

        return x_peak_hits, y_peak_hits, z_peak_hits

    def cluster_hits(
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
        
    def find_2d_peaks(
            self,
            fiber_hits:typing.List['Hit2D'], 
            u:str, v:str,
            peak_prominance_threshold:float = 0.0,
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

        for hit_id in range(len(fiber_hits)):
            
            hit = fiber_hits[hit_id]
            charge = hit.weight
            
            ## modify charge based on the prominence threshold supplied
            modified_charge = charge * (1.0 - peak_prominance_threshold)

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

            if np.sum(v_line_charges < modified_charge) >= 2:
                is_peak = True
                local_peak_hits = self._find_peak_hits(hit, extended_v_line_hits, v)
                for h in local_peak_hits:
                    v_info_hits.append(h)

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
                
            if is_peak:
                new_hit = Hit2D.copy(hit)

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
        
        # print()
        # print("finding diagonal")

        ret_list = list()
        for neighbour in neighbourhood:
            u_dist = (getattr(neighbour, u) - getattr(hit, u)) / u_pitch
            v_dist = (getattr(neighbour, v) - getattr(hit, v)) / v_pitch
            
            #print(f"u diff = {u_dist} :: v diff = {v_dist}")
            if abs(u_dist - diagonal_sign * v_dist) < 0.0001:
                #print("adding!")

                ret_list.append(neighbour)

        return ret_list
