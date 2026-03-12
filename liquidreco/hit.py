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

        hit.time = min(x_fiber_hit.time, y_fiber_hit.time)
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