import typing
import numpy as np
from scipy import ndimage
import math as m

class Hit:
    """Very generic detector hit, probably shouldn't use directly, should instead use one of the derived classes
    """

    def __init__(
        self,
        x = None,
        y = None,
        z = None,
        time = None,
        weight = None
    ):
        
        self.x = x
        """x position of the hit"""
        self.y = y
        """y position of the hit"""
        self.z = z
        """z position of the hit"""
        self.time = time
        """time of the hit"""
        self.weight = weight
        """the "weight" of this hit. Typically the charge/light collected"""

    def __str__(self) -> str:
        """Summarise this hit as a string

        :return: summary
        :rtype: str
        """
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