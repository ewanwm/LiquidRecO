import typing

import numpy as np

from liquidreco.modules.module_base import ModuleBase
from liquidreco.event import Event
from liquidreco.hit import Hit, Hit2D, Hit3D

class HitBuilder2D(ModuleBase):
    """Absolute base level module to build 2D hits from raw hit information

    The starting point for any good reconstruction chain.
    Takes in raw hit info and turns it into more structiured hits. 
    """

    def __init__(self):
        """Initialise the module
        """

        super().__init__()

        self.requirements = ["raw_positions", "raw_weights", "raw_times"]
        self.outputs = ["x_fiber_hits", "y_fiber_hits", "z_fiber_hits"]

    def _setup_cli_options(self, parser):

        parser.add_argument(
            "--x-fiber-x-pos", 
            help="The x position that x fibers have", 
            required = False, default = 0.0, type = float,
        )
        parser.add_argument(
            "--y-fiber-y-pos", 
            help="The y position that y fibers have", 
            required = False, default = 30.0, type = float,
        )
        parser.add_argument(
            "--z-fiber-z-pos", 
            help="The z position that z fibers have", 
            required = False, default = 910.0, type = float,
        )

    def _initialise(self):
        
        ## TODO: use some geometry manager class to handle this sort of thing
        ## The positions that define fibers in each direction
        self.x_fiber_x_pos: float = self.args.x_fiber_x_pos
        self.y_fiber_y_pos: float = self.args.y_fiber_y_pos
        self.z_fiber_z_pos: float = self.args.z_fiber_z_pos

    def _process(self, event: Event) -> None:

        positions = event["raw_positions"]
        weights = event["raw_weights"]
        times = event["raw_times"]

        assert (
            len(positions.shape) == 2 and 
            (positions.shape[1] == 2 or positions.shape[1] == 3)
        ), f"invalid position array. Has shape {positions.shape} but expected shape [nHits, 2 or 3]"

        ## find which hits live in which plane
        x_hit_ids = np.where(positions[:, 0] == self.x_fiber_x_pos)[0]
        y_hit_ids = np.where(positions[:, 1] == self.y_fiber_y_pos)[0]
        z_hit_ids = np.where(positions[:, 2] == self.z_fiber_z_pos)[0]

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

        event.add_data("x_fiber_hits", x_fiber_hits)
        event.add_data("y_fiber_hits", y_fiber_hits)
        event.add_data("z_fiber_hits", z_fiber_hits)
        


class HitBuilder3D(ModuleBase):
    """Makes 3D hits from 2d fiber hits
    """

    def __init__(self) -> None:
        """Initialise the module
        """

        super().__init__()

        self.requirements = ["x_fiber_hits", "y_fiber_hits", "z_fiber_hits"]
        self.outputs = ["3d_hits"]

    def _setup_cli_options(self, parser):
        
        parser.add_argument(
            "--require-3-fibers", 
            help="Whether to require 3 fibers to form a hit or allow only on x and one y fiber", 
            required = False, default = True, type = bool
        )
        parser.add_argument(
            "--pitch", 
            help="The distance between fibers in each direction", 
            required = False, default = (10, 10, 10), nargs=3
        )
        parser.add_argument(
            "--min-2d-hit-weight", 
            help="The minimum weight a 2D fiber hit must have to be considered when building 3D hits", 
            required = False, default = 0.0, type = float
        )
        parser.add_argument(
            "--n-required-peaks", 
            help="The number of 'peak hits' required in each direction to form a valid 3D hit (see :func:`Hit3D.from_fiber_hits` for more details)", 
            required = False, default = None, type = int
        )
        parser.add_argument(
            "--max-weighted-distance", 
            help="The maximum 'real' i.e. peak weighted discance between 2D hits in order for them to be considered for combining into a 3D hit. Specified as a fraction of fiber pitch", 
            required = False, default = None, type = float
        )

    def _initialise(self):
        
        self.require_3_fibers: bool = self.args.require_3_fibers
        self.pitch: typing.Tuple[float] = self.args.pitch
        self.min_2d_hit_weight: float = self.args.min_2d_hit_weight
        self.n_required_peaks: int = self.args.n_required_peaks
        self.max_weighted_distance: float = self.args.max_weighted_distance

    def _process(self, event: Event) -> None:
        
        ## get the fiber hits from the event
        x_fiber_hits: typing.List[Hit2D] = event["x_fiber_hits"]
        y_fiber_hits: typing.List[Hit2D] = event["y_fiber_hits"]
        z_fiber_hits: typing.List[Hit2D] = event["z_fiber_hits"]

        ## get the projected positions for each hit
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
            low = np.searchsorted(y_sorted[:, 1], x_pos[1] - self.pitch[2])
            high = np.searchsorted(y_sorted[:, 1], x_pos[1] + self.pitch[2])

            ## construct x and y positions of the two fiber hit

            for i in range(low, high):

                ## the x and y fiber hits under consideration
                x_hit = x_fiber_hits[x_sorted_indices[x_pos_id]]
                y_hit = y_fiber_hits[y_sorted_indices[i]]

                if (
                    abs( y_hit.fiber_z - x_hit.fiber_z) > self.pitch[2] * 0.75 
                ): 
                    continue

                if self.max_weighted_distance is not None:
                    if (
                        abs( y_hit.z - x_hit.z) > self.pitch[2] * self.max_weighted_distance
                    ): 
                        continue

                if (
                    x_hit.weight < self.min_2d_hit_weight or
                    y_hit.weight < self.min_2d_hit_weight
                ):
                    continue

                ## if we're only requiring 2D hits, apply min n peak condition here, otherwise hold off until we build the 3 fiber hit
                if not self.require_3_fibers:
                    hit = Hit3D.from_fiber_hits(x_hit, y_hit, n_required_peaks=self.n_required_peaks)
                else:
                    hit = Hit3D.from_fiber_hits(x_hit, y_hit)

                if hit:
                    two_fiber_hits.append(hit)

        ## if we're not requiring z hits we can stop here
        if not self.require_3_fibers:

            event.add_data("3d_hits", three_fiber_hits)
            
            return

        ## z fiber hits sorted by x position
        z_hits_x_sorted_indices = z_positions[:, 0].argsort() 
        z_hits_x_sorted = z_positions[z_hits_x_sorted_indices]

        three_fiber_hits = [] 
        ## now check for corresponding z fiber hits
        for two_fiber_hit in two_fiber_hits:
            low = np.searchsorted(z_hits_x_sorted[:, 0], two_fiber_hit.voxel_x - self.pitch[0])
            high = np.searchsorted(z_hits_x_sorted[:, 0], two_fiber_hit.voxel_x + self.pitch[0])

            for i in range(low, high):

                ## the z fiber hit under consideration
                z_hit = z_fiber_hits[z_hits_x_sorted_indices[i]]

                if (
                    abs(z_hit.fiber_x - two_fiber_hit.voxel_x ) > self.pitch[0] * 0.75 or
                    abs(two_fiber_hit.voxel_y - z_hit.fiber_y ) > self.pitch[1] * 0.75
                ):
                    continue


                if self.max_weighted_distance is not None:
                    if (
                        abs( z_hit.x - two_fiber_hit.x) > self.pitch[0] * self.max_weighted_distance or
                        abs( z_hit.y - two_fiber_hit.y) > self.pitch[1] * self.max_weighted_distance
                    ): 
                        continue
                
                if (
                    z_hit.weight < self.min_2d_hit_weight
                ):
                    continue

                # build the three fiber hit
                hit = Hit3D.from_fiber_hits(
                    two_fiber_hit.x_fiber_hit, 
                    two_fiber_hit.y_fiber_hit, 
                    z_hit,
                    n_required_peaks = self.n_required_peaks
                )

                if hit:
                    three_fiber_hits.append(hit)

        event.add_data("3d_hits", three_fiber_hits)
        
        return
