import typing
from typing import Any

import numpy as np

## alias for all things we accept as a 3-vector
Vector3 = typing.Union[typing.Tuple[Any], typing.List[Any], typing.Dict[str, Any]]

def tuple_from_maybe_dict(vec: Vector3) -> typing.Tuple[Any]:
    """Geta tuple representing a 3 vector from input that could be iterable or could be dict of form {"x": *, "y": *, "z": *}

    :param vec: The variable to extract 3 vector from
    :type vec: typing.Tuple[Any] | typing.List[Any] | typing.Dict[str, Any]
    :return: 3-tuple representing 3 vector
    :rtype: typing.Tuple[Any]
    """

    ret = None

    if type(vec) in [tuple, list]:

        assert len(vec) == 3, "3-vector must have length 3!!!"

        ret = tuple(vec)

    elif type(vec) == dict:

        keys = vec.keys()

        assert len(keys) <= 3, "Too many entries in position dict!!!"

        x, y, z = None, None, None
        
        if "x" in keys:
            x = vec["x"]

        if "y" in keys:
            y = vec["y"]

        if "z" in keys:
            z = vec["z"]

        ret = (x, y, z)

    else:
        raise ValueError("What the hell is this????? I need something more vector-y")

    return ret
    
class Hit:
    """Very generic detector hit, probably shouldn't use directly, should instead use one of the derived classes
    """
    def __init__(
        self,
        position = (None, None, None),
        time = None,
        weight = None,
        direction = (None, None, None),
        is_peak = (False, False, False)
    ):
        
        self.pos = position
        """position of the hit"""
        self.x = self.pos[0]
        """x position of the hit"""
        self.y = self.pos[1]
        """y position of the hit"""
        self.z = self.pos[2]
        """z position of the hit"""
        self.time = time
        """time of the hit"""
        self.weight = weight
        """the "weight" of this hit. Typically the charge/light collected"""
        self.dir_x = None
        """x position of the hit"""
        self.dir_y = None
        """y position of the hit"""
        self.dir_z = None
        """z position of the hit"""
        self.is_x_peak = None
        """is a peak in x direction"""
        self.is_y_peak = None
        """is a peak in the y direction"""
        self.is_z_peak = None
        ## set with function so properly normalises (also sets dir_* variables)
        self.set_is_peak(is_peak)
        """Whether or not this hit is a peak hit"""
        self.set_direction(direction) 
        """the direction of the hit. e.g. the direction along the ridge for a peak hit"""

    def set_is_peak(self, new_is_peak: Vector3) -> None:
        """Set info about peakness of the hit
        
        :param new_is_peak: New position (must have length 3)
        :type new_is_peak: typing.Tuple[bool]
        """

        self.is_peak = tuple_from_maybe_dict(new_is_peak)

        self.is_x_peak = self.is_peak[0]
        self.is_y_peak = self.is_peak[1]
        self.is_z_peak = self.is_peak[2]

    def set_position(self, new_pos: Vector3) -> None:
        """Set a new position for this hit

        :param new_pos: New position (must have length 3)
        :type new_pos: typing.Tuple[float]
        """

        self.pos = tuple_from_maybe_dict(new_pos)

        self.x = self.pos[0]
        self.y = self.pos[1]
        self.z = self.pos[2]

    def set_direction(self, new_dir: Vector3) -> None:
        """Set a new direction for this hit

        This will normalise the direction before setting the variables

        :param new_dir: New direction (must have length 3)
        :type new_dir: typing.Tuple[float]
        """

        unnormed = tuple_from_maybe_dict(new_dir)
        
        ## normalise
        accum = 0.0
        for i in range(3):
            if unnormed[i] is not None:
                accum += unnormed[i] * unnormed[i]

        mag = np.sqrt(accum)

        x, y, z = None, None, None
        
        if mag != 0.0:
            if unnormed[0] is not None:
                x = unnormed[0] / mag
            if unnormed[1] is not None:
                y = unnormed[1] / mag
            if unnormed[2] is not None:
                z = unnormed[2] / mag

        ## set members
        self.dir = (x, y, z)
        self.dir_x = self.dir[0]
        self.dir_y = self.dir[1]
        self.dir_z = self.dir[2]

    def is_peak(self, direction:str = None) -> bool:
        """Checks if this hit is considered
         
        Can specify a direction to check if it is a peak in a particular direction i.e. more position info than just the fiber position

        :param direction: Check if the hit is a peak in a specific direction, defaults to None
        :type direction: str
        :return: True if this hit is a peak hit 
        :rtype: bool
        """

        if direction is None:
            return self.is_x_peak or self.is_y_peak or self.is_z_peak
        
        else:
            ## Just check if the fiber has a direction defined along the given axis
            return getattr(self, f"is_{direction}_peak")
        
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

        new_hit.set_position(old_hit.pos)
        new_hit.set_fiber_position(old_hit.fiber_pos)
        new_hit.set_direction(old_hit.dir)

        if old_hit.time is not None:
            new_hit.time = float(old_hit.time)
        if old_hit.weight is not None:
            new_hit.weight = float(old_hit.weight)
        if old_hit.secondary_hits is not []:
            new_hit.secondary_hits = list(old_hit.secondary_hits)

        new_hit.is_peak = old_hit.is_peak

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
        pos = (None, None, None),
        fiber_pos = (None, None, None),
        time = None,
        weight = None,
        secondary_hits = None
    ):
        
        self.fiber_x = None
        self.fiber_y = None
        self.fiber_z = None

        self.set_fiber_position(fiber_pos)
        
        super().__init__(pos, time, weight)

        if secondary_hits is None:
            self.secondary_hits = list()
        else:
            self.secondary_hits = secondary_hits

    def __str__(self):
        return super().__str__() + f" :: secondaries: {len(self.secondary_hits)}"

    def set_fiber_position(self, new_fiber_pos: Vector3) -> None:
        """Set a new fiber position for this hit

        :param new_fiber_pos: New fiber direction (must have length 3)
        :type new_fiber_post: typing.Tuple[float]
        """

        self.fiber_pos = tuple_from_maybe_dict(new_fiber_pos)
        
        ## Now set the individual position element variables
        self.fiber_x = self.fiber_pos[0]
        self.fiber_y = self.fiber_pos[1]
        self.fiber_z = self.fiber_pos[2]

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

        direction = [0.0, 0.0, 0.0]

        hit.x_fiber_hit = x_fiber_hit
        hit.y_fiber_hit = y_fiber_hit
        hit.z_fiber_hit = z_fiber_hit

        hit.voxel_z = (x_fiber_hit.z + y_fiber_hit.z) / 2.0
        direction[2] = (x_fiber_hit.dir_z + y_fiber_hit.dir_z) / 2.0

        hit.voxel_x = y_fiber_hit.x
        direction[0] = y_fiber_hit.dir_x
        if z_fiber_hit is not None:
            hit.voxel_x += z_fiber_hit.x
            hit.voxel_x /= 2.0

            direction[0] += z_fiber_hit.dir_x
            direction[0] /= 2.0

        hit.voxel_y = x_fiber_hit.y
        direction[1] = x_fiber_hit.dir_y
        if z_fiber_hit is not None:
            hit.voxel_y += z_fiber_hit.y
            hit.voxel_y /= 2.0

            direction[1] += z_fiber_hit.dir_y
            direction[1] /= 2.0

        hit.set_direction(direction)

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
        position = (None, None, None),
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
            position,
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