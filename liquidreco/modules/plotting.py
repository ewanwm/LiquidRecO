import matplotlib
from matplotlib import pyplot as plt

from liquidreco.plotting import make_corner_plot, make_corner_plot_fiber_hits, make_rotating_gif
from liquidreco.modules.module_base import ModuleBase
from liquidreco.event import Event

class HitPlotter2D(ModuleBase):
    def __init__(self):

        super().__init__()
          
        self.requirements = ["x_fiber_hits", "y_fiber_hits", "z_fiber_hits"]
        self.outputs = []

    def _help(self):
    
        return """Plots 2D (fiber) hits"""

    def _setup_cli_options(self, parser):

        parser.add_argument(
            "--file-name",
            help="Name of the pdf file to save plots to",
            default="fiberHits.pdf", required=False, type=str
        )
        parser.add_argument(
            "--charge-cutoff",
            help="Will cap charges at this value for the purpose of plotting",
            default=100.0, required=False, type=float
        )
        parser.add_argument(
            "--show-directions",
            help="show hit direction on plots for peak hits",
            action='store_true'
        )

    def _initialise(self):
        
        self._fibers_pdf = matplotlib.backends.backend_pdf.PdfPages(self.args.file_name)
        self.charge_cutoff = self.args.charge_cutoff
        self._show_directions = self.args.show_directions

    def _finalise(self):
          
        self._fibers_pdf.close()

    def _process(self, event: Event) -> None:
            
        fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(5, 5))
        fig.suptitle("Homo-FGD Hits - Fiber Hits")

        make_corner_plot_fiber_hits(
            fig,
            axs, 
            event["x_fiber_hits"],
            event["y_fiber_hits"],
            event["z_fiber_hits"],
            charge_cutoff=self.charge_cutoff,
            plot_directions=self._show_directions
        )

        self._fibers_pdf.savefig(fig)

        plt.close(fig)
        
class HitPlotter3D(ModuleBase):

    def __init__(self):

        super().__init__()

        self.requirements = ["3d_hits", "id"]
        self.outputs = []

    def _help(self):
    
        return """Plots 3D hits

default behaviour is to plot only the 3 individual 2D projections in "corner plots".
but can be configured to also plot fully 3D events."""

    def _setup_cli_options(self, parser):

        parser.add_argument(
            "--file-name",
            help="Name of the pdf file to save plots to",
            default="3Dhits.pdf", required=False, type=str
        )
        parser.add_argument(
            "--make-3d-plots",
            help="Whether to make 3D plots of the hits",
            action='store_true'
        )
        parser.add_argument(
            "--make-gifs",
            help="Whether to make animated gifs of hits",
            action='store_true'
        )
        parser.add_argument(
            "--charge-cutoff",
            help="Will cap charges at this value for the purpose of plotting",
            default=100.0, required=False, type=float
        )
        parser.add_argument(
            "--show-directions",
            help="show hit direction on plots for peak hits",
            action='store_true'
        )

    def _initialise(self):
        
        self.charge_cutoff = self.args.charge_cutoff
        self._pdf = matplotlib.backends.backend_pdf.PdfPages(self.args.file_name)
        self._show_directions = self.args.show_directions

    def _finalise(self):
        
        self._pdf.close()

    def _process(self, event):
        
        event_3d_hits = event["3d_hits"]

        event_3d_hits.sort(key = lambda h: h.weight)

        fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(5, 5))
        fig.suptitle("Homo-FGD 3D Hits")
        make_corner_plot(
            fig, 
            axs, 
            event_3d_hits, 
            charge_cutoff=self.charge_cutoff,
            plot_directions=self._show_directions
        )

        self._pdf.savefig(fig)

        fig.clf()        
        
        plt.close(fig)

        if self.args.make_3d_plots:
            self.make_3d_plot(event_3d_hits)

        if self.args.make_gifs:
            self.make_gif(event_3d_hits, event["id"])

    def make_gif(self, event_hits, event_id):
        cmap = plt.get_cmap("coolwarm")
        c = [cmap(ev.weight / self.charge_cutoff) for ev in event_hits]
        s = [min(ev.weight, self.charge_cutoff) / self.charge_cutoff for ev in event_hits]
        
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
        c = [cmap(ev.weight / self.charge_cutoff) for ev in event_hits]
        s = [0.1 * min(ev.weight, self.charge_cutoff) / self.charge_cutoff for ev in event_hits]
        
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

        self._pdf.savefig(fig)
        plt.close(fig)
