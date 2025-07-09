import typing
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage
from scipy.stats import laplace
from scipy.optimize import curve_fit
import math as m
import matplotlib
from matplotlib import pyplot as plt

from sklearn.cluster import DBSCAN

from liquidreco.plotting import make_corner_plot_fiber_hits


class Hit:
    def __init__(
        self,
        x = None,
        y = None,
        z = None,
        time = None,
        weight = None
    ):
        
        self.x = x
        self.y = y
        self.z = z
        self.time = time
        self.weight = weight

    def __str__(self):
        return f"pos = ({self.x}, {self.y}, {self.z}) :: t = {self.time} :: w = {self.weight}"
        
class Hit2D(Hit):
    """Describes a 2D WLS fiber hit"""

    @staticmethod
    def copy(old_hit:'Hit2D') -> 'Hit2D':
        """Create a copy of an existing 2D hit

        :param old_hit: the hit to copy
        :type old_hit: Hit2D
        :return: copied hit
        :rtype: Hit2D
        """

        new_hit = Hit2D()

        if old_hit.x is not None:
            new_hit.x = float(old_hit.x)
        if old_hit.y is not None:
            new_hit.y = float(old_hit.y)
        if old_hit.z is not None:
            new_hit.z = float(old_hit.z)
        if old_hit.fiber_x is not None:
            new_hit.fiber_x = float(old_hit.fiber_x)
        if old_hit.fiber_y is not None:
            new_hit.fiber_y = float(old_hit.fiber_y)
        if old_hit.fiber_z is not None:
            new_hit.fiber_z = float(old_hit.fiber_z)
        if old_hit.time is not None:
            new_hit.time = float(old_hit.time)
        if old_hit.weight is not None:
            new_hit.weight = float(old_hit.weight)
        if old_hit.secondary_hits is not []:
            new_hit.secondary_hits = list(old_hit.secondary_hits)

        return new_hit

    @staticmethod
    def get_mean_pos(hits:typing.List['Hit2D'], direction:str) -> float:
        """ Get the mean position of a list of 2D hits in a particular dimension

        :param hits: The hits to take the average of
        :type hits: typing.List[Hit2D]
        :param direction: the dimension to take the average in. e.g. 'x' to get the mean x position 'y' for mean y...
        :type direction: str
        :return: the weighted mean position
        :rtype: float
        """

        accum = 0.0
        mean = 0.0

        for hit in hits:
            if getattr(hit, direction) is not None:
                mean += getattr(hit, direction) * hit.weight
                accum += hit.weight

        return mean / accum


    def __init__(
        self,
        x = None,
        y = None,
        z = None,
        fiber_x = None,
        fiber_y = None,
        fiber_z = None,
        time = None,
        weight = None,
        secondary_hits = None
    ):
        
        self.fiber_x = fiber_x
        self.fiber_y = fiber_y
        self.fiber_z = fiber_z
        
        super().__init__(x, y, z, time, weight)

        if secondary_hits is None:
            self.secondary_hits = list()
        else:
            self.secondary_hits = secondary_hits

    def __str__(self):
        return super().__str__() + f" :: secondaries: {len(self.secondary_hits)}"


    def add_secondary_hit(self, new_hit, add_tertiary=False):

        if add_tertiary:
            for tertiary_hit in new_hit.secondary_hits:
                if not tertiary_hit in self.secondary_hits and tertiary_hit is not self:
                    self.secondary_hits.append(tertiary_hit)

        self.secondary_hits.append(new_hit)
        
    def get_weight(self):

        if self.secondary_hits != []:
            w = self.weight

            for other_hit in self.secondary_hits:
                if other_hit.weight is not None:
                    w += other_hit.weight

            return w
        
        else:
            return self.weight
        
    def is_peak(self, direction:str) -> bool:
        """Checks if this hit is considered a peak in a particular direction i.e. more position info than just the fiber position

        :param direction: The direction to test
        :type direction: str
        :return: True if this hit has peak info in the specified direction. 
        :rtype: bool
        """

        ## literally just check if the position is the same as the fiber position
        ## if not it must have been modified and therefore must be a peak... could be done more elegantly
        return getattr(self, direction) != getattr(self, "fiber_" + direction) 

class Hit3D(Hit):
    """Describes a 3D hit constructed from two or three WLS fibers"""

    @staticmethod
    def from_fiber_hits(x_fiber_hit:Hit2D=None, y_fiber_hit:Hit2D=None, z_fiber_hit:Hit2D=None, n_required_peaks:int = None) -> 'Hit3D':
        """Create a Hit3D from some 2D fiber hits

        :param x_fiber_hit: The x fiber
        :type x_fiber_hit: Hit2D
        :param y_fiber_hit: The y fiber
        :type y_fiber_hit: Hit2D
        :param z_fiber_hit: The (optional) z fiber, defaults to None
        :type z_fiber_hit: Hit2D, optional
        :param n_required_peaks: require this number of peak hits to be present in each direction to successfuly create a hit.
            e.g. if n_required_peaks = 1 and 0 of the provided fibers are classified as z peaks, the hit will not be valid. defaults to None
        :type n_required_peaks: int
        :return: The constructed 3D hit, None if the n_required_peaks condition is not satisfied in any direction
        :rtype: Hit3D
        """

        hit = Hit3D()

        hit.x_fiber_hit = x_fiber_hit
        hit.y_fiber_hit = y_fiber_hit
        hit.z_fiber_hit = z_fiber_hit

        hit.voxel_z = (x_fiber_hit.z + y_fiber_hit.z) / 2.0
        hit.voxel_x = y_fiber_hit.x
        if z_fiber_hit is not None:
            hit.voxel_x += z_fiber_hit.x
            hit.voxel_x /= 2.0

        hit.voxel_y = x_fiber_hit.y
        if z_fiber_hit is not None:
            hit.voxel_y += z_fiber_hit.y
            hit.voxel_y /= 2.0

        hit.time = min (x_fiber_hit.time, y_fiber_hit.time)
        if z_fiber_hit is not None:
            hit.time = min(hit.time, z_fiber_hit.time)

        hit.weight = min(x_fiber_hit.weight, y_fiber_hit.weight)
        if z_fiber_hit is not None:
            hit.weight = min(hit.weight, z_fiber_hit.weight)

        ## check the number of peak hits in each dimension
        if n_required_peaks:
            n_x_peaks:int = int(y_fiber_hit.is_peak("x"))
            if z_fiber_hit:
                n_x_peaks += int(z_fiber_hit.is_peak("x"))

            n_y_peaks:int = int(x_fiber_hit.is_peak("y"))
            if z_fiber_hit:
                n_y_peaks += int(z_fiber_hit.is_peak("y"))
                
            n_z_peaks:int = int(x_fiber_hit.is_peak("z") + y_fiber_hit.is_peak("z"))

            ## if any are below the threshold, return a None
            if (
                (n_x_peaks < n_required_peaks) or
                (n_y_peaks < n_required_peaks) or
                (n_z_peaks < n_required_peaks)
            ):
                return None

        ## if hit is valid, calculate the means 
        hit.x = hit.get_mean_x()
        hit.y = hit.get_mean_y()
        hit.z = hit.get_mean_z()
        
        return hit

    def __init__(
        self,
        x = None,
        y = None,
        z = None,
        voxel_x = None,
        voxel_y = None,
        voxel_z = None, 
        time = None,
        weight = None,
        x_fiber_hit = None,
        y_fiber_hit = None,
        z_fiber_hit = None
    ):
        super().__init__(
            x,
            y,
            z,
            time,
            weight
        )
        
        self.voxel_x = voxel_x
        self.voxel_y = voxel_y
        self.voxel_z = voxel_z

        self.x_fiber_hit = x_fiber_hit
        self.y_fiber_hit = y_fiber_hit
        self.z_fiber_hit = z_fiber_hit
    
    def get_mean_x(self) -> float:

        if self.z_fiber_hit is not None:

            ## if neither 2d hit contains extra x information, just get voxel position
            if not self.y_fiber_hit.is_peak("x") and not self.z_fiber_hit.is_peak("x"):
                return self.voxel_x
            
            ## otherwise do a weighted mean of the two
            x = 0.0
            accum = 0.0

            if self.y_fiber_hit.is_peak("x"):
                x += self.y_fiber_hit.x
                accum += 1
            if self.z_fiber_hit.is_peak("x"):
                x += self.z_fiber_hit.x
                accum += 1
            
            return x / accum
        
        ## if no z fiber then we only have position info from one fiber anyway
        else:
            return self.y_fiber_hit.x

    def get_mean_y(self) -> float:

        if self.z_fiber_hit is not None:

            ## if neither 2d hit contains extra x information, just get voxel position
            if not self.x_fiber_hit.is_peak("y") and not self.z_fiber_hit.is_peak("y"):
                return self.voxel_y
            
            ## otherwise do a weighted mean of the two
            y = 0.0
            accum = 0.0

            if self.x_fiber_hit.is_peak("y"):
                y += self.x_fiber_hit.y
                accum += 1
            if self.z_fiber_hit.is_peak("y"):
                y += self.z_fiber_hit.y
                accum += 1
            
            return y / accum
        
        ## if no z fiber then we only have position info from one fiber anyway
        else:
            return self.x_fiber_hit.y
        
    def get_mean_z(self) -> float:

        ## if neither 2d hit contains extra x information, just get voxel position
        if not self.x_fiber_hit.is_peak("z") and not self.y_fiber_hit.is_peak("z"):
            return self.voxel_z
        
        ## otherwise do a weighted mean of the two
        z = 0.0
        accum = 0.0

        if self.x_fiber_hit.is_peak("z"):
            z += self.x_fiber_hit.z
            accum += 1
        if self.y_fiber_hit.is_peak("z"):
            z += self.y_fiber_hit.z
            accum += 1
        
        return z / accum

class Event():
    
    def __init__(
        self,
        hits_3d:list[Hit] = None,
    ):
        
        self.hits_3d:list[Hit] = hits_3d

class PeakFinder2D:
    """Finds peaks in raw fiber hits and performs position corrections
    """
    
    def __init__(
            self, 
            peak_prominance_threshold:float,
            make_plots:bool = False
        ):

        self._pdf = matplotlib.backends.backend_pdf.PdfPages("PeakFinder2D-plots.pdf")
        self._laplace_pdf = matplotlib.backends.backend_pdf.PdfPages("PeakFinder2D-laplace-fit-plots.pdf")
        self._cluster_pdf = matplotlib.backends.backend_pdf.PdfPages("PeakFinder2D-unused-hit-cluster-plots.pdf")

        self._peak_prominance_threshold = peak_prominance_threshold

        self._make_plots = make_plots

        self._clusterer = DBSCAN(10.5)

    def finalise(self):
        """Tidy up and close open pdfs
        """
        self._pdf.close()
        self._laplace_pdf.close()
        self._cluster_pdf.close()

    def __call__(
        self, 
        x_fiber_hits:typing.List[Hit2D] = None,
        y_fiber_hits:typing.List[Hit2D] = None,
        z_fiber_hits:typing.List[Hit2D] = None
    ) -> typing.Tuple[typing.List[Hit2D]]:
        """Perform the peak finding

        :param x_fiber_hits: Hits from x fibers (fibers orthogonal to the yz plane), defaults to None
        :type x_fiber_hits: typing.List[Hit2D], optional
        :param y_fiber_hits: Hits from y fibers (fibers orthogonal to the xz plane), defaults to None
        :type y_fiber_hits: typing.List[Hit2D], optional
        :param z_fiber_hits: Hits from z fibers (fibers orthogonal to the xy plane), defaults to None
        :type z_fiber_hits: typing.List[Hit2D], optional
        :return: peak hits in each projection
        :rtype: typing.Tuple[typing.List[Hit2D]]
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
                _laplace_peaks = self.find_2d_peaks_laplace(cluster, u, v)
            
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
        fiber_hits:typing.List[Hit2D],
        u:str, v:str,
    ) -> typing.List[typing.List[Hit2D]]:
        """Cluster hits using this PeakFinder2Ds _clusterer (default is DBSCAN but you can set it to whatever you want)

        :param fiber_hits: The hits to cluster
        :type fiber_hits: typing.List[Hit2D]
        :param u: the "u" direction, should be 'x', 'y' or 'z' depending on the projection
        :type u: str
        :param v:  the "v" direction, should be 'x', 'y' or 'z' depending on the projection
        :type v: str
        :return: hits broken down into clusters
        :rtype: typing.List[typing.List[Hit2D]]
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
            fiber_hits:typing.List[Hit2D], 
            u:str, v:str,
            peak_prominance_threshold:float = 0.0,
            neighbourhood_dist:float = 15.1, extended_neighbourhood_dist:float = 30.1,
            u_pitch:float=10.0, v_pitch:float=10.0
        ) -> typing.Tuple[typing.List[Hit2D], typing.Set[Hit2D], typing.Set[Hit2D]]:
        
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
        main_hit:Hit2D,
        line_hits:list[Hit2D],
        direction:str
    ) -> typing.List[Hit2D]:
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
        :type line_hits: list[Hit2D]
        :param direction: The direction along the "line" of hits, should be "x", "y" or "z"
        :type direction: str
        :return: list of hits that belong to the same peak as the main hit
        :rtype: typing.List[Hit2D]
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
            hit:Hit, 
            neighbourhood:typing.List[Hit], 
            u:str, v:str, 
            u_pitch:float, v_pitch:float, 
            diagonal_sign = +1
        ) -> typing.List[Hit2D]:
        """Gets neighbours of a hit along a diagonal line

        :param hit: The main hit
        :type hit: Hit
        :param neighbourhood: The hits to search for diagonal neighbours in
        :type neighbourhood: typing.List[Hit]
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
        :rtype: typing.List[Hit2D]
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


    def find_2d_peaks_laplace(
        self,
        fiber_hits: typing.List[Hit2D], 
        u:str, v:str,
        neighbourhood_dist:float = 45.0,
        peak_candidate_threshold:float = 100.0,
        amplitude_threshold:float = 50.0,
        min_n_neighbours:int = 0
    ) -> typing.Tuple[typing.List[Hit]]:
        """Find peaks by attempting to fit laplace distributions near prominent fibers

        :param fiber_hits: The hits to fit to
        :type fiber_hits: typing.List[Hit2D]
        :param u: The u direction, should be "x", "y" or "z"
        :type u: str
        :param v: The v direction, should be "x", "y" or "z"
        :type v: str
        :param neighbourhood_dist: The distance around a fiber that is considered its local neighbourhood. Only fibers in the same neighbourhood are considered when sharing light in order to speed up the fit, defaults to 45.0
        :type neighbourhood_dist: float, optional
        :param peak_candidate_threshold: Fibers with a charge above this are considered candidate peak hits, defaults to 100.0
        :type peak_candidate_threshold: float, optional
        :param amplitude_threshold: Fibers whose post-fit laplace distribution amplitude is above this are considered peak hits, defaults to 50.0
        :type amplitude_threshold: float, optional
        :param min_n_neighbours: Fibers must have this many other fibers in their neighbourhood to be included in the fit, defaults to 0
        :type min_n_neighbours: int, optional
        :return: Peak hits found by the fit
        :rtype: typing.Tuple[typing.List[Hit]]
        """
        
        print(f"  - LAPLACE FIT: n {u}{v} fibers: {len(fiber_hits)}")
        
        ## get positions of the hits for convenience
        positions = np.array([[getattr(hit, u), getattr(hit, v)] for hit in fiber_hits])

        ## use this to define neighbourhood around fibers
        neighbour_algo = NearestNeighbors(radius = neighbourhood_dist)
    
        ## first do a check on the number of neighbours of each hit, discard ones with too few
        ## this speeds up algorithm as we don't want to consider things that are clearly not peaks
        _, fiber_neighbour_indices = neighbour_algo.fit(positions).radius_neighbors(positions)
        considered_hits = [fiber_hits[i] for i in range(len(fiber_hits)) if fiber_neighbour_indices[i].shape[0] > min_n_neighbours]
        considered_positions = np.array([[getattr(hit, u), getattr(hit, v)] for hit in considered_hits])

        print(f"  - LAPLACE FIT: n {u}{v} fibers passing N neighbours cut: {len(considered_hits)}")

        if len(considered_hits) == 0:
            return []
        
        ## now apply condition on the charge in the fiber
        peak_candidates = [hit for hit in considered_hits if hit.weight > peak_candidate_threshold]
        peak_candidate_positions = np.array([[getattr(hit, u), getattr(hit, v)] for hit in peak_candidates])

        print(f"  - LAPLACE FIT: n {u}{v} peak candidate fibers: {len(peak_candidates)}")

        if len(peak_candidates) == 0:
            return []
        
        ## now get neighbours of the peak candidates
        neighbour_distances, neighbour_indices = neighbour_algo.fit(peak_candidate_positions).radius_neighbors(considered_positions)

        ## now do the fit fit
        try:
            laplace_amplitudes = self._fit_laplace(
                considered_positions, 
                charges=np.array([hit.weight for hit in considered_hits]), 
                neighbour_distances=neighbour_distances, 
                neighbour_indices=neighbour_indices, 
                peak_candidate_threshold=peak_candidate_threshold)

        ## fit might fail for whatever reason, but we don't want that to bring the whole thing crashing down
        except:
            return []
        
        return [peak_candidates[i] for i in range(len(peak_candidates)) if laplace_amplitudes[i] > amplitude_threshold]

    def _get_laplace_fn(
            self, 
            fiber_positions:np.ndarray, 
            fiber_charges:np.ndarray, 
            neighbour_distances:np.array, 
            neighbour_indices:typing.List[np.array], 
            peak_candidate_threshold: float=100.0, 
            fix_width:float=None
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
        :param peak_candidate_threshold: The charge threshold above which fibers are considered peak candidates, defaults to 100.0
        :type peak_candidate_threshold: float, optional
        :param fix_width: Set this to fix the width of the laplace distributions in the fit. If None then it will be fit as a parameter, defaults to None
        :type fix_width: float, optional
        :return: Function that predicts charges in fibers, can then be used in fit.
        :rtype: typing.Callable
        """

        N_FIBERS = len(fiber_positions)
        
        peak_candidates = [i for i in range(N_FIBERS) if fiber_charges[i] > peak_candidate_threshold]

        N_PEAK_CANDIDATES = len(peak_candidates)

        p0 = [1.0 for _ in range(N_PEAK_CANDIDATES)] ##2.0 * fiber_charges[i] for i in range(N_PEAK_CANDIDATES)]
        p0.append(1.0) # <- the width param

        def _laplace_fn(x_data=None, *params, fix_width:float=fix_width):

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

    def _fit_laplace(
            self, 
            fiber_positions:np.ndarray, 
            charges:np.array, 
            neighbour_distances:np.array, 
            neighbour_indices:typing.List[np.array], 
            peak_candidate_threshold:float=100.0
        ) -> typing.List[float]:
        """Performs the fit of the laplace distributions

        :param fiber_positions: Positions of all fibers considered in the fit
        :type fiber_positions: np.ndarray
        :param charges: measured charges of all fibers considered in the fit
        :type charges: np.array
        :param neighbour_distances: Distances from each fiber to peak candidates
        :type neighbour_distances: np.array
        :param neighbour_indices: indices defining which peak candidate should contribute to each fibers charge
        :type neighbour_indices: typing.List[np.array]
        :param peak_candidate_threshold: Threshold above which to consider a fiber a peak candidate, defaults to 100.0
        :type peak_candidate_threshold: float, optional
        :return: The fitted laplace amplitudes of all peak candidate fibers
        :rtype: typing.List[float]
        """

        laplace_fn, p0 = self._get_laplace_fn(fiber_positions, charges, neighbour_distances, neighbour_indices, peak_candidate_threshold, fix_width=12.0)

        optimal_params, cov_mat = curve_fit(laplace_fn, xdata=None, ydata=charges, p0=p0, bounds=(0.0, np.inf))

        print(f"Optimal params: {optimal_params.shape} \n{optimal_params.tolist()}")

        weights = laplace_fn(None, *optimal_params.tolist())

        fig, axs = plt.subplots(3,1)
        
        h, x_bins, y_bins, _ = axs[0].hist2d(
            fiber_positions[:, 0], fiber_positions[:, 1], 
            weights = charges, bins=(20, 20), 
            cmap=plt.get_cmap("coolwarm"),
            vmax=peak_candidate_threshold
        )

        mappable = axs[1].hist2d(
            fiber_positions[:, 0], fiber_positions[:, 1], 
            weights = weights, bins=(20, 20), 
            cmap=plt.get_cmap("coolwarm"), 
            vmax=peak_candidate_threshold
        )
        
        fig.colorbar(mappable[3], ax=axs)
        
        axs[2].hist2d(
            [fiber_positions[i, 0] for i in range(len(charges)) if charges[i] > peak_candidate_threshold], 
            [fiber_positions[i, 1] for i in range(len(charges)) if charges[i] > peak_candidate_threshold], 
            weights = optimal_params.tolist()[:-1], 
            bins=(x_bins, y_bins), 
            cmap=plt.get_cmap("coolwarm"),
            #cmin = 0.0001,
            #vmax = 100.0
        )

        self._laplace_pdf.savefig(fig)

        plt.clf()

        plt.scatter([charge for charge in charges if charge > peak_candidate_threshold], optimal_params[:-1])
        self._laplace_pdf.savefig(fig)
        plt.close(fig)

        return optimal_params.tolist()[:-1]


def build_2d_hits(positions:np.ndarray, weights:np.ndarray, times:np.ndarray,
        x_fiber_x_pos:float = 0.0, y_fiber_y_pos:float = 60.0, z_fiber_z_pos:float = 910.0) -> typing.Tuple[typing.List[Hit]]:
    
    assert (
        len(positions.shape) == 2 and 
        (positions.shape[1] == 2 or positions.shape[1] == 3)
    ), f"invalid position array. Has shape {positions.shape} but expected shape [nHits, 2 or 3]"
    
    x_hit_ids = np.where(positions[:, 0] == x_fiber_x_pos)[0]
    y_hit_ids = np.where(positions[:, 1] == y_fiber_y_pos)[0]
    z_hit_ids = np.where(positions[:, 2] == z_fiber_z_pos)[0]

    x_positions = positions[x_hit_ids, :][:, [1,2]]
    y_positions = positions[y_hit_ids, :][:, [0,2]]
    z_positions = positions[z_hit_ids, :][:, [0,1]]

    x_weights = weights[x_hit_ids]
    y_weights = weights[y_hit_ids]
    z_weights = weights[z_hit_ids]

    x_times = times[x_hit_ids]
    y_times = times[y_hit_ids]
    z_times = times[z_hit_ids]

    x_fiber_hits = []
    y_fiber_hits = []
    z_fiber_hits = []

    for x_id in range(len(x_positions)):
        x_fiber_hits.append(
            Hit2D(
                x = None, 
                y = x_positions[x_id, 0], 
                z = x_positions[x_id, 1], 
                fiber_x = None, 
                fiber_y = x_positions[x_id, 0], 
                fiber_z = x_positions[x_id, 1], 
                time = x_times[x_id], 
                weight = x_weights[x_id]
            )
        )


    for y_id in range(len(y_positions)):
        y_fiber_hits.append(
            Hit2D(
                x = y_positions[y_id, 0], 
                y = None, 
                z = y_positions[y_id, 1], 
                fiber_x = y_positions[y_id, 0], 
                fiber_y = None, 
                fiber_z = y_positions[y_id, 1], 
                time = y_times[y_id], 
                weight = y_weights[y_id]
            )
        )


    for z_id in range(len(z_positions)):
        z_fiber_hits.append(
            Hit2D(
                x = z_positions[z_id, 0], 
                y = z_positions[z_id, 1], 
                z = None, 
                fiber_x = z_positions[z_id, 0], 
                fiber_y = z_positions[z_id, 1], 
                fiber_z = None, 
                time = z_times[z_id], 
                weight = z_weights[z_id]
            )
        )

    return x_fiber_hits, y_fiber_hits, z_fiber_hits
    
def build_3d_hits(
        x_fiber_hits:typing.List[Hit2D], y_fiber_hits:typing.List[Hit2D], z_fiber_hits:typing.List[Hit2D], 
        require_3_fibers:bool = True,
        pitch:typing.Tuple[float] = (10.0, 10.0, 10.0),
        min_2d_hit_weight:float = 0.0,
        n_required_peaks:int = None,
        max_weighted_distance: float = None
        ) -> typing.List[Hit]:
    """Makes 3D hits from arrays of info about 2d fiber hits

    :param x_fiber_hits: List of 2D YZ fibers
    :type x_fiber_hits: typing.List[Hit2D]
    :param y_fiber_hits: List of 2D XZ fibers
    :type y_fiber_hits: typing.List[Hit2D]
    :param z_fiber_hits: List of 2D XY fibers
    :type z_fiber_hits: typing.List[Hit2D]
    :param require_3_fibers: Whether to require 3 fibers to form a hit or allow only on x and one y fiber, defaults to True
    :type require_3_fibers: bool, optional
    :param pitch: The distance between fibers in each direction, defaults to (10.0, 10.0, 10.0)
    :type pitch: typing.Tuple[float], optional
    :param min_2d_hit_weight: The minimum weight a 2D fiber hit must have to be considered when building 3D hits, defaults to 0.0
    :type min_2d_hit_weight: float, optional
    :param n_required_peaks: The number of "peak hits" required in each direction to form a valid 3D hit (see :func:`Hit3D.from_fiber_hits` for more details), defaults to None
    :type n_required_peaks: int, optional
    :param max_weighed_distance: The maximum "real" i.e. peak weighted discance between 2D hits in order for them to be considered for combining into a 3D hit. Specified as a fraction of fiber pitch, defaults to None
    :type max_weighted_distance: float, optional
    :return: list of 3d hits
    :rtype: typing.List[Hit]
    """

    
    x_positions = np.array([[hit.fiber_y, hit.fiber_z] for hit in x_fiber_hits])
    y_positions = np.array([[hit.fiber_x, hit.fiber_z] for hit in y_fiber_hits])
    z_positions = np.array([[hit.fiber_x, hit.fiber_y] for hit in z_fiber_hits])

    ## first build hits from two fibers
    two_fiber_hits = []

    ## x and y hits sorted by z position
    x_sorted_indices = x_positions[:,1].argsort()
    y_sorted_indices = y_positions[:,1].argsort()
    x_sorted = x_positions[x_sorted_indices]
    y_sorted = y_positions[y_sorted_indices]

    for x_pos_id, x_pos in enumerate(x_sorted):
        low = np.searchsorted(y_sorted[:, 1], x_pos[1] - pitch[2])
        high = np.searchsorted(y_sorted[:, 1], x_pos[1] + pitch[2])

        ## construct x and y positions of the two fiber hit

        for i in range(low, high):

            ## the x and y fiber hits under consideration
            x_hit = x_fiber_hits[x_sorted_indices[x_pos_id]]
            y_hit = y_fiber_hits[y_sorted_indices[i]]

            if (
                abs( y_hit.fiber_z - x_hit.fiber_z) > pitch[2] * 0.75 
            ): 
                continue

            if max_weighted_distance is not None:
                if (
                    abs( y_hit.z - x_hit.z) > pitch[2] * max_weighted_distance
                ): 
                    continue

            if (
                x_hit.weight < min_2d_hit_weight or
                y_hit.weight < min_2d_hit_weight
            ):
                continue

            ## if we're only requiring 2D hits, apply min n peak condition here, otherwise hold off until we build the 3 fiber hit
            if not require_3_fibers:
                hit = Hit3D.from_fiber_hits(x_hit, y_hit, n_required_peaks=n_required_peaks)
            else:
                hit = Hit3D.from_fiber_hits(x_hit, y_hit)

            # print(f"x hit:  {x_hit}")
            # print(f"y hit:  {y_hit}")
            # print(f"3d hit: {hit}")

            if hit:
                two_fiber_hits.append(hit)

    if not require_3_fibers:
        return two_fiber_hits

    ## z fiber hits sorted by x position
    z_hits_x_sorted_indices = z_positions[:, 0].argsort() 
    z_hits_x_sorted = z_positions[z_hits_x_sorted_indices]

    three_fiber_hits = [] 
    ## now check for corresponding z fiber hits
    for two_fiber_hit in two_fiber_hits:
        low = np.searchsorted(z_hits_x_sorted[:, 0], two_fiber_hit.voxel_x - pitch[0])
        high = np.searchsorted(z_hits_x_sorted[:, 0], two_fiber_hit.voxel_x + pitch[0])

        for i in range(low, high):

            ## the z fiber hit under consideration
            z_hit = z_fiber_hits[z_hits_x_sorted_indices[i]]

            # print(f"two fiber hit: {two_fiber_hit}")

            if (
                abs(z_hit.fiber_x - two_fiber_hit.voxel_x ) > pitch[0] * 0.75 or
                abs(two_fiber_hit.voxel_y - z_hit.fiber_y ) > pitch[1] * 0.75
            ):
                continue


            if max_weighted_distance is not None:
                if (
                    abs( z_hit.x - two_fiber_hit.x) > pitch[0] * max_weighted_distance or
                    abs( z_hit.y - two_fiber_hit.y) > pitch[1] * max_weighted_distance
                ): 
                    continue
            
            if (
                z_hit.weight < min_2d_hit_weight
            ):
                continue

            # build the three fiber hit
            hit = Hit3D.from_fiber_hits(
                two_fiber_hit.x_fiber_hit, 
                two_fiber_hit.y_fiber_hit, 
                z_hit,
                n_required_peaks = n_required_peaks
            )

            if hit:
                three_fiber_hits.append(hit)

    return three_fiber_hits


def local_normalisation(event_hits, window_size, x_bins, y_bins, z_bins):

    """do a local normalisation of the hits so that lokal peaks have "charge" of 1
    and hits in the neighbourhood have their charges normalised by this value """
    
    event_3d_histogram, _ = np.histogramdd(
        [
            [ev.x for ev in event_hits],
            [ev.y for ev in event_hits],
            [ev.z for ev in event_hits],
        ],
        bins = [
            x_bins.shape[0] -1, y_bins.shape[0] -1, z_bins.shape[0] -1,
        ],
        range = (
            (x_bins[0], x_bins[-1]),
            (y_bins[0], y_bins[-1]),
            (z_bins[0], z_bins[-1])
        ),
        weights = [hit.weight for hit in event_hits]
    )

    def safe_norm(a):

        if np.max(a) == 0:
            return 0.0
        else:

            central_val = a[m.floor(len(a) / 2) + 1]

            if central_val == 0.0:
                return 0.0
            
            return central_val# / np.max(a)

    local_max = ndimage.generic_filter(
        event_3d_histogram,
        np.max,
        size = (window_size, window_size, window_size)
    )

    event_3d_histogram = event_3d_histogram / local_max

    for hit in event_hits:
        x = np.searchsorted(x_bins, hit.x)
        y = np.searchsorted(y_bins, hit.y)
        z = np.searchsorted(z_bins, hit.z)

        hit.weight = 1.0 #event_3d_histogram[x, y, z]