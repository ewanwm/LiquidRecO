import numpy as np
import uproot
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
from tqdm import tqdm

from liquidreco.reconstruction import LINSCAN, LocalMeanDBSCAN, HoughTransform, HesseRidgeDetection
from liquidreco.plotting import make_corner_plot, make_corner_plot_fiber_hits, make_rotating_gif
from liquidreco.hit import Event, build_2d_hits, build_3d_hits, local_normalisation, find_2d_peaks


class EventProcessor:
    def __init__(self, arg_parser, file_name:str, make_fiber_hit_plots=False, make_3d_hit_plots=False):

        self._file_name = file_name
        self._make_fiber_hit_plots = make_fiber_hit_plots
        self._make_3d_hit_plots = make_3d_hit_plots

        self._positions = None
        self._weights   = None
        self._times     = None
        self._event_ids = None

        self._fibers_pdf  = None
        self._3d_hits_pdf = None

        self._fit_algorithm = None

        self.args = arg_parser


        # define bins to use for histograms
        self.x_bins = np.arange(
            start=self.args.plot_centre_x - self.args.plot_size_x / 2.0, 
            stop=self.args.plot_centre_x + self.args.plot_size_x / 2.0 + 5.0, 
            step=5.0
        )
        self.y_bins = np.arange(
            start=self.args.plot_centre_y - self.args.plot_size_y / 2.0, 
            stop=self.args.plot_centre_y + self.args.plot_size_y / 2.0 + 5.0, 
            step=5.0
        )
        self.z_bins = np.arange(
            start=self.args.plot_centre_z - self.args.plot_size_z / 2.0, 
            stop=self.args.plot_centre_z + self.args.plot_size_z / 2.0 + 5.0, 
            step=5.0
        )

    def read_hit_info(self, hit_tree_name:str):

        with uproot.open(self._file_name) as input_file:

            ## get info from the hit tree
            hit_tree = input_file[hit_tree_name]

            assert hit_tree is not None, f"No hit tree found with the name {hit_tree_name}"

            self._positions = np.array(hit_tree["pos"].array(library="np").tolist())
            self._weights   = np.array(hit_tree["charge"].array(library="np").tolist())
            self._times     = np.array(hit_tree["time"].array(library="np").tolist())
            self._event_ids = np.array(hit_tree["event"].array(library="np").tolist())

    def apply_weight_scaling(self, scaling: float) -> None:
        """Apply scaling to the charges of the events

        Might be useful if you want e.g. to compare truth hits with hits that 
        have DAQ or fiber effects applied.

        :param scaling: The scaling to apply
        :type scaling: float
        """
        self._weights *= scaling

    def initialise(self):
        self._corner_pdf = matplotlib.backends.backend_pdf.PdfPages("3dHitCornerPlots.pdf")

        if self._make_fiber_hit_plots:
            self._fibers_pdf = matplotlib.backends.backend_pdf.PdfPages("fiberHitPositions.pdf")

        if self._make_3d_hit_plots:
            self._3d_hits_pdf = matplotlib.backends.backend_pdf.PdfPages("3dHitPositions.pdf")
        
        if self.args.fit_algorithm == "hesse":
            self._fit_algorithm = HesseRidgeDetection(
                None,
                (self.args.plot_centre_x - self.args.plot_size_x / 2.0, self.args.plot_centre_x + self.args.plot_size_x / 2.0),
                (self.args.plot_centre_y - self.args.plot_size_y / 2.0, self.args.plot_centre_y + self.args.plot_size_y / 2.0),
                (self.args.plot_centre_z - self.args.plot_size_z / 2.0, self.args.plot_centre_z + self.args.plot_size_z / 2.0)
            )

        elif self.args.fit_algorithm == "hough":
            self._fit_algorithm = HoughTransform(
                None,
            )

        elif self.args.fit_algorithm == "lmdbscan":
            self._fit_algorithm = LocalMeanDBSCAN(
                None,
            )


    def finalise(self):
        if self._corner_pdf is not None:
            self._corner_pdf.close()

        if self._3d_hits_pdf is not None:
            self._3d_hits_pdf.close()

        if self._fibers_pdf is not None:
            self._fibers_pdf.close()

        if self._fit_algorithm is not None:
            self._fit_algorithm.finalise()

    def process(self):

        self.initialise()

        ## loop through each event
        unique_event_ids = np.unique(self._event_ids)
        if self.args.max_n_events is not None:
            max_n = min(self.args.max_n_events, unique_event_ids.shape[0])
            unique_event_ids = unique_event_ids[:max_n]
        for event_id in tqdm(unique_event_ids):

            ## positions in the position, weight and time arrays for this event
            event_array_positions = np.where(self._event_ids == event_id)[0]

            ## values for hits belonging to this particular event
            event_positions = self._positions[event_array_positions, ...]
            event_weights   = self._weights[event_array_positions, ...]
            event_times     = self._times[event_array_positions, ...]

            x_fiber_hits, y_fiber_hits, z_fiber_hits = build_2d_hits(
                event_positions, event_weights, event_times,
                x_fiber_x_pos=self.args.x_fiber_x_pos,
                y_fiber_y_pos=self.args.y_fiber_y_pos,
                z_fiber_z_pos=self.args.z_fiber_z_pos
            )

            if self.args.find_2d_peaks:
                x_fiber_hits, y_fiber_hits, z_fiber_hits = find_2d_peaks(
                    x_fiber_hits,
                    y_fiber_hits,
                    z_fiber_hits
                )

            event_3d_hits = build_3d_hits(
                x_fiber_hits, y_fiber_hits, z_fiber_hits, 
                require_3_fibers=not self.args.allow_2_fiber_hits,
                min_2d_hit_weight=self.args.min_2d_hit_weight,
                n_required_peaks=self.args.n_required_peak_hits)

            if len(event_3d_hits) == 0:
                print("NO HITS!! SKIPPING EVENT!!")
                continue

            if self.args.normalise:
                local_normalisation(
                    event_3d_hits, 
                    self.args.norm_window_size, 
                    self.x_bins, self.y_bins, self.z_bins
                )
                
            event_3d_hits.sort(key = lambda h: h.weight)

            fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(5, 5))
            fig.suptitle("Homo-FGD 3D Hits")
            make_corner_plot(fig, axs, event_3d_hits, charge_cutoff=self.args.charge_cutoff)

            self._corner_pdf.savefig(fig)

            fig.clf()

            if self.args.make_fiber_hit_plots:
                fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(5, 5))
                fig.suptitle("Homo-FGD Hits - Fiber Hits")

                make_corner_plot_fiber_hits(
                    fig,
                    axs, 
                    x_fiber_hits,
                    y_fiber_hits,
                    z_fiber_hits,
                    charge_cutoff=self.args.charge_cutoff
                )

                self._fibers_pdf.savefig(fig)


            if self.args.make_3d_hit_plots:
                self.make_3d_plot(event_3d_hits)

            if self.args.make_gifs:
                self.make_gif(event_3d_hits, event_id)

            if self._fit_algorithm is not None:
                self._fit_algorithm._process(Event(event_3d_hits))

    def make_gif(self, event_hits, event_id):
        cmap = plt.get_cmap("coolwarm")
        c = [cmap(ev.weight / self.args.charge_cutoff) for ev in event_hits]
        s = [min(ev.weight, self.args.charge_cutoff) / self.args.charge_cutoff for ev in event_hits]
        
        fig = plt.figure(dpi=400)
        ax = fig.add_subplot(projection='3d')

        # Grab some example data and plot a basic wireframe.
        ax.scatter(
            [ev.x for ev in event_hits],
            [ev.y for ev in event_hits],
            [ev.z for ev in event_hits],
            c = c,
            s = s,
            #alpha = s
        )

        make_rotating_gif(fig, ax, f"3d_hits_event_{event_id}.gif")

    def make_3d_plot(self, event_hits):
        # make 3d plot
        cmap = plt.get_cmap("coolwarm")
        c = [cmap(ev.weight / self.args.charge_cutoff) for ev in event_hits]
        s = [0.1 * min(ev.weight, self.args.charge_cutoff) / self.args.charge_cutoff for ev in event_hits]
        
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(projection='3d')
        
        # Grab some example data and plot a basic wireframe.
        ax.scatter(
            [hit.x for hit in event_hits],
            [hit.y for hit in event_hits],
            [hit.z for hit in event_hits],
            c = c,
            s = s,
            #alpha = s
        )

        self._3d_hits_pdf.savefig(fig)
        plt.clf()