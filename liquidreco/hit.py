import typing
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import ndimage
import math as m


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
        
    def is_cluster(self, direction):
        """True if this hit has cluster info in the specified direction. i.e. more position info than just the fiber position"""

        return getattr(self, direction) != getattr(self, "fiber_" + direction) 

class Hit3D(Hit):
    """Describes a 3D hit constructed from two or three WLS fibers"""

    @staticmethod
    def from_fiber_hits(x_fiber_hit:Hit2D, y_fiber_hit:Hit2D, z_fiber_hit:Hit2D=None) -> 'Hit3D':
        """Create a Hit3D from some 2D fiber hits

        :param x_fiber_hit: The x fiber
        :type x_fiber_hit: Hit2D
        :param y_fiber_hit: The y fiber
        :type y_fiber_hit: Hit2D
        :param z_fiber_hit: The (optional) z fiber, defaults to None
        :type z_fiber_hit: Hit2D, optional
        :return: The constructed 3D hit
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
            if not self.y_fiber_hit.is_cluster("x") and not self.z_fiber_hit.is_cluster("x"):
                return self.voxel_x
            
            ## otherwise do a weighted mean of the two
            x = 0.0
            accum = 0.0

            if self.y_fiber_hit.is_cluster("x"):
                x += self.y_fiber_hit.x
                accum += 1
            if self.z_fiber_hit.is_cluster("x"):
                x += self.z_fiber_hit.x
                accum += 1
            
            return x / accum
        
        ## if no z fiber then we only have position info from one fiber anyway
        else:
            return self.y_fiber_hit.x

    def get_mean_y(self) -> float:

        if self.z_fiber_hit is not None:

            ## if neither 2d hit contains extra x information, just get voxel position
            if not self.x_fiber_hit.is_cluster("y") and not self.z_fiber_hit.is_cluster("y"):
                return self.voxel_y
            
            ## otherwise do a weighted mean of the two
            y = 0.0
            accum = 0.0

            if self.x_fiber_hit.is_cluster("y"):
                y += self.x_fiber_hit.y
                accum += 1
            if self.z_fiber_hit.is_cluster("y"):
                y += self.z_fiber_hit.y
                accum += 1
            
            return y / accum
        
        ## if no z fiber then we only have position info from one fiber anyway
        else:
            return self.x_fiber_hit.y
        
    def get_mean_z(self) -> float:

        ## if neither 2d hit contains extra x information, just get voxel position
        if not self.x_fiber_hit.is_cluster("z") and not self.y_fiber_hit.is_cluster("z"):
            return self.voxel_z
        
        ## otherwise do a weighted mean of the two
        z = 0.0
        accum = 0.0

        if self.x_fiber_hit.is_cluster("z"):
            z += self.x_fiber_hit.z
            accum += 1
        if self.y_fiber_hit.is_cluster("z"):
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
    
def _find_peak_hits(
    main_hit:Hit2D,
    line_hits:list[Hit2D],
    direction:str
):
    
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


def _get_diagonal_neighbours(hit:Hit, neighbourhood:typing.List[Hit], u:str, v:str, u_pitch:float, v_pitch:float, diagonal_sign = +1):
    
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

def find_2d_peaks(
        x_fiber_hits, y_fiber_hits, z_fiber_hits, 
        neighbourhood_dist = 15.1, extended_neighbourhood_dist = 30.1,
        u_pitch=10.0, v_pitch=10.0) -> typing.Tuple[typing.List[Hit]]:

    x_peak_hits = []
    y_peak_hits = []
    z_peak_hits = []

    neighbour_algo = NearestNeighbors(radius = neighbourhood_dist)
    extended_neighbour_algo = NearestNeighbors(radius = extended_neighbourhood_dist)

    for peak_hits, fiber_hits in zip(
        [x_peak_hits, y_peak_hits, z_peak_hits],
        [x_fiber_hits, y_fiber_hits, z_fiber_hits]
    ):
        
        if fiber_hits is x_fiber_hits:
            u = "y"
            v = "z"
        elif fiber_hits is y_fiber_hits:
            u = "x"
            v = "z"
        elif fiber_hits is z_fiber_hits:
            u = "x"
            v = "y"

        data = np.array([[getattr(hit, u), getattr(hit, v)] for hit in fiber_hits])
        
        _, indices = neighbour_algo.fit(data).radius_neighbors(data)
        _, extended_indices = extended_neighbour_algo.fit(data).radius_neighbors(data)

        for hit_id in range(len(fiber_hits)):
            
            hit = fiber_hits[hit_id]
            charge = hit.weight

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
            if np.sum(u_line_charges < charge) >= 2:
                is_peak = True
                local_peak_hits = _find_peak_hits(hit, extended_u_line_hits, u)
                for h in local_peak_hits:
                    u_info_hits.append(h)

            if np.sum(v_line_charges < charge) >= 2:
                is_peak = True
                local_peak_hits = _find_peak_hits(hit, extended_v_line_hits, v)
                for h in local_peak_hits:
                    v_info_hits.append(h)


            ## check if it's a diagonal peak

            if not is_peak:
                diag_uv_line_hits = _get_diagonal_neighbours(hit, neighbourhood, u, v, u_pitch, v_pitch)
                diag_vu_line_hits = _get_diagonal_neighbours(hit, neighbourhood, u, v, u_pitch, v_pitch, diagonal_sign=-1)

                diag_uv_line_charges = np.array([neighbour.weight for neighbour in diag_uv_line_hits])
                diag_vu_line_charges = np.array([neighbour.weight for neighbour in diag_vu_line_hits])

                extended_diag_uv_line_hits = _get_diagonal_neighbours(hit, extended_neighbourhood, u, v, u_pitch, v_pitch)
                extended_diag_vu_line_hits = _get_diagonal_neighbours(hit, extended_neighbourhood, u, v, u_pitch, v_pitch, diagonal_sign=-1)

                if np.sum(diag_uv_line_charges < charge) >= 2:
                    is_peak = True
                    local_peak_hits = _find_peak_hits(hit, extended_diag_uv_line_hits, u)
                    for h in local_peak_hits:
                        u_info_hits.append(h)
                        v_info_hits.append(h)

                if np.sum(diag_vu_line_charges < charge) >= 2:
                    is_peak = True
                    local_peak_hits = _find_peak_hits(hit, extended_diag_vu_line_hits, u)
                    for h in local_peak_hits:
                        u_info_hits.append(h)
                        v_info_hits.append(h)
                
            if is_peak:
                new_hit = Hit2D.copy(hit)

                setattr(new_hit, u, Hit2D.get_mean_pos(u_info_hits, u))
                setattr(new_hit, v, Hit2D.get_mean_pos(v_info_hits, v))

                peak_hits.append(new_hit)


    return x_peak_hits, y_peak_hits, z_peak_hits


def build_3d_hits(
        x_fiber_hits, y_fiber_hits, z_fiber_hits, require_3_fibers:bool = True,
        pitch:typing.Tuple[float] = (10.0, 10.0, 10.0),
        min_2d_hit_weight:float = 0.0,
        ) -> typing.List[Hit]:
    
    """Makes 3D hits from arrays of info about 2d fiber hits

    :param positions: positions of the 2d hits, should be of shape [nHits, 3] where 2nd index is over xPos, yPos, zPos
    :type positions: np.ndarraye
    :param weights: Weights to apply to each event, typically charge, should be of shape [nHits, 1]
    :type weights: np.ndarray
    :param times: times for each hit, should be of shape [nHits, 1]
    :type times: np.ndarray
    :return: list of 3d hits
    :rtype: list[Hit]
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

            if (
                x_hit.weight < min_2d_hit_weight or
                y_hit.weight < min_2d_hit_weight
            ):
                continue

            hit = Hit3D.from_fiber_hits(x_hit, y_hit)

            # print(f"x hit:  {x_hit}")
            # print(f"y hit:  {y_hit}")
            # print(f"3d hit: {hit}")

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
            
            if (
                z_hit.weight < min_2d_hit_weight
            ):
                continue

            # build the three fiber hit
            hit = Hit3D.from_fiber_hits(
                two_fiber_hit.x_fiber_hit, 
                two_fiber_hit.y_fiber_hit, 
                z_hit
            )

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