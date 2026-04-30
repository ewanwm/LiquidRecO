from liquidreco.plotting import make_corner_plot, make_corner_plot_fiber_hits

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.linalg import eig

import typing

from liquidreco.hit import Hit, Hit2D, Hit3D
from liquidreco.modules.module_base import ModuleBase
from liquidreco.event import Event
from liquidreco.geometry import GeometryManager

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
        ]

    def _initialise(self):

        self._pdf = matplotlib.backends.backend_pdf.PdfPages("PeakFinder2D-plots.pdf")

        self._peak_prominance_threshold = self.args.peak_prominance_threshold
        self._peak_candidate_weight_threshold = self.args.peak_candidate_weight_threshold
        self._make_plots = self.args.make_plots

    def _setup_cli_options(self, parser):
        
        parser.add_argument(
            "--peak-prominance-threshold", 
            help="For a hit to be considered a 'simple' peak, its neighbours must have smaller charge than peak_prominance_threshold * hit charge. 1.0 is most general, smaller values mean only sharper peaks get accepted.", 
            required = False, default = 1.0, type = float,
        )
        parser.add_argument(
            "--peak-candidate-weight-threshold", 
            help="For a hit to be considered a peak candidate, it must have at least this weight.", 
            required = False, default = 0.0, type = float,
        )
        parser.add_argument(
            "--make-plots", 
            help="Whether to make debug plots.", 
            action='store_true'
        )
        
    def _finalise(self):
        """Tidy up and close open pdfs
        """

        self._pdf.close()

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

        x_used = set()
        y_used = set()
        z_used = set()

        x_unused = set()
        y_unused = set()
        z_unused = set()

        ## this is vile :(
        for fiber_hits, peak_hits, used, unused in zip(
            [x_fiber_hits, y_fiber_hits, z_fiber_hits],
            [x_peak_hits, y_peak_hits, z_peak_hits],
            [x_used, y_used, z_used],
            [x_unused, y_unused, z_unused],
        ):
            
            if fiber_hits is None:
                continue
            
            if fiber_hits is x_fiber_hits:
                u, v = "z", "y"
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

            self._pdf.savefig(fig)
            plt.close(fig)


        event.add_data("x_peak_hits", x_peak_hits)
        event.add_data("y_peak_hits", y_peak_hits)
        event.add_data("z_peak_hits", z_peak_hits)
        
        event.add_data("unused_x_hits", x_unused)
        event.add_data("unused_y_hits", y_unused)
        event.add_data("unused_z_hits", z_unused)

        event.add_data("x_fiber_hits", x_peak_hits)
        event.add_data("y_fiber_hits", y_peak_hits)
        event.add_data("z_fiber_hits", z_peak_hits)

        return
            
    def _find_2d_peaks(
            self,
            fiber_hits:typing.List['Hit2D'], 
            u:str, v:str,
            neighbourhood_dist:float = 15.1, extended_neighbourhood_dist:float = 30.1
        ) -> typing.Tuple[typing.List['Hit2D'], typing.Set['Hit2D'], typing.Set['Hit2D']]:
        
        ## get pitches in each direction
        u_pitch = GeometryManager().get_pitch(u)
        v_pitch = GeometryManager().get_pitch(v)

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
                diag_uv_line_hits = self._get_diagonal_neighbours(hit, neighbourhood, u, v)
                diag_vu_line_hits = self._get_diagonal_neighbours(hit, neighbourhood, u, v, diagonal_sign=-1)

                diag_uv_line_charges = np.array([neighbour.weight for neighbour in diag_uv_line_hits])
                diag_vu_line_charges = np.array([neighbour.weight for neighbour in diag_vu_line_hits])

                extended_diag_uv_line_hits = self._get_diagonal_neighbours(hit, extended_neighbourhood, u, v)
                extended_diag_vu_line_hits = self._get_diagonal_neighbours(hit, extended_neighbourhood, u, v, diagonal_sign=-1)

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
        :param diagonal_sign: The gradient of the diagonal, defaults to +1
        :type diagonal_sign: int, optional
        :return: Hits from the neighbourhood that lie along the specified diagonal
        :rtype: typing.List['Hit2D']
        """

        ## get pitches in each direction
        u_pitch = GeometryManager().get_pitch(u)
        v_pitch = GeometryManager().get_pitch(v)

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

            u_pitch = GeometryManager().get_pitch(u_name)
            v_pitch = GeometryManager().get_pitch(v_name)

            u_bins = np.arange(start=min(u_values) - 3.0 * u_pitch / 2.0, stop=max(u_values) + 5.0 * u_pitch / 2.0, step = u_pitch) 
            v_bins = np.arange(start=min(v_values) - 3.0 * v_pitch / 2.0, stop=max(v_values) + 5.0 * v_pitch / 2.0, step = v_pitch) 

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

        fig, ax = plt.subplots(1, 8, figsize=(40, 5))
        m = ax[0].imshow(hist, cmap=plt.get_cmap("coolwarm"), origin='lower')
        plt.colorbar(m)
        ax[0].set_title("Original Event")

        m = ax[1].imshow(huu, cmap=plt.get_cmap("gray"), origin='lower')
        ax[1].set_title(f"H_{u_name}{u_name}")
        plt.colorbar(m)
        m = ax[2].imshow(huv, cmap=plt.get_cmap("gray"), origin='lower')
        ax[2].set_title(f"H_{u_name}{v_name}")
        plt.colorbar(m)
        m = ax[3].imshow(hvu, cmap=plt.get_cmap("gray"), origin='lower')
        ax[3].set_title(f"H_{v_name}{u_name}")
        plt.colorbar(m)
        m = ax[4].imshow(hvv, cmap=plt.get_cmap("gray"), origin='lower')
        ax[4].set_title(f"H_{v_name}{v_name}")
        plt.colorbar(m)

        du, dv = self._gradient(hist)
        m = ax[5].imshow(du, cmap=plt.get_cmap("gray"), origin='lower')
        ax[5].set_title(f"D_{u_name}")
        plt.colorbar(m)
        m = ax[6].imshow(dv, cmap=plt.get_cmap("gray"), origin='lower')
        ax[6].set_title(f"D_{v_name}")
        plt.colorbar(m)

        hess_eigenvals, _ = self._hess_eigen(hist)
        ridgeness = self._compute_ridgeness(hist, hess_eigenvals)
        m = ax[7].imshow(ridgeness, cmap=plt.get_cmap("gray"), origin='lower')
        ax[7].set_title("Hessian Filter")
        plt.colorbar(m)

        self._debug_pdf.savefig(fig)
        plt.close(fig)
        
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        z_low, z_high = np.min(hess_eigenvals[0,...]), np.max(hess_eigenvals[0,...])
        max_z = max(-z_low, z_high)
        m = ax[0].imshow(hess_eigenvals[0,...], cmap=plt.get_cmap("coolwarm"), origin='lower', vmax = -max_z, vmin = max_z)
        ax[0].set_title("evals[0]")
        plt.colorbar(m)

        z_low, z_high = np.min(hess_eigenvals[1,...]), np.max(hess_eigenvals[1,...])
        max_z = max(-z_low, z_high)
        m = ax[1].imshow(hess_eigenvals[1,...], cmap=plt.get_cmap("coolwarm"), origin='lower', vmax = -max_z, vmin = max_z)
        plt.colorbar(m)
        ax[1].set_title("evals[1]")
        
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

        u_pitch = GeometryManager().get_pitch(u)
        v_pitch = GeometryManager().get_pitch(v)
        w_pitch = GeometryManager().get_pitch(w)

        u_bins = np.arange(start=min(u_values) - 3.0 * u_pitch / 4.0, stop=max(u_values) + 5.0 * u_pitch / 4.0, step = u_pitch / 2.0) 
        v_bins = np.arange(start=min(v_values) - 3.0 * v_pitch / 4.0, stop=max(v_values) + 5.0 * v_pitch / 4.0, step = v_pitch / 2.0) 
        w_bins = np.arange(start=min(w_values) - 3.0 * w_pitch / 4.0, stop=max(w_values) + 5.0 * w_pitch / 4.0, step = w_pitch / 2.0) 

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
