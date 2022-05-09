import xarray as xr

from tsdat import IngestPipeline, get_start_date_and_time_str, get_filename

# from utils import format_time_xticks
import matplotlib.pyplot as plt
from palettable.colorbrewer.sequential import YlOrRd_3 as my_colors01
from palettable.colorbrewer.qualitative import Set2_3 as qual_colors
from utils import format_time_xticks
from cmocean.cm import amp_r, dense, haline
import seaborn as sns
import cartopy.crs as ccrs
import geopy.distance
import numpy as np
from mhkit.wave.performance import capture_length_matrix

# DEVELOPER: Implement your pipeline and update its docstring.
class MbariWec(IngestPipeline):
    """---------------------------------------------------------------------------------
    This is an example ingestion pipeline meant to demonstrate how one might set up a
    pipeline using this template repository.

    ---------------------------------------------------------------------------------"""

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # DEVELOPER: (Optional) Use this hook to modify the dataset before qc is applied
        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # DEVELOPER: (Optional) Use this hook to modify the dataset after qc is applied
        # but before it gets saved to the storage area
        return dataset

    def hook_plot_dataset(self, dataset: xr.Dataset):
        Te_lims = [5,13]
        Hm0_lims = [0.5, 2.5]

        ds = dataset
        loc = self.dataset_config.attrs.location_id
        datastream: str = self.dataset_config.attrs.datastream

        date, time = get_start_date_and_time_str(dataset)

        plt.style.use("default")  # clear any styles that were set before
        plt.style.use("shared/styling.mplstyle")

        with self.storage.uploadable_dir(datastream) as tmp_dir:

            # spectrogram
            fig, ax = plt.subplots()
            ds.S.sel(freq=slice(1/4)).dropna('time').plot.contourf(ax=ax,
                                levels=12,
                                cmap=my_colors01.mpl_colormap)
            # fig.tight_layout()
            ax.autoscale(enable=True, axis='x', tight=True)

            plot_file = get_filename(ds, title="spectrogram", extension="png")
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)



            # time history
            vars_to_plot = ['J',
                'Hm0',
                'Te',
                'mean_dir'
                ]
            fig, ax = plt.subplots(nrows=len(vars_to_plot),
                                figsize=(8,12),
                                sharex=True)

            for axi, var in zip(ax, vars_to_plot):
                axi.set_prop_cycle('color', qual_colors.mpl_colors)
                qual_colors
                ds[var].plot(ax=axi,
                            marker='.',
                            )
                axi.label_outer()
                axi.spines['right'].set_visible(False)
                axi.spines['top'].set_visible(False)
                axi.autoscale(enable=True, axis='x', tight=True)

            ax[3].fill_between(ds.time.values, ds.mean_dir, ds.mean_dir + ds.mean_spread,
                            color=qual_colors.mpl_colors[0],
                            alpha=0.25,
                            )
            p1 = ax[3].fill_between(ds.time.values, ds.mean_dir, ds.mean_dir - ds.mean_spread,
                            color=qual_colors.mpl_colors[0],
                            alpha=0.25,
                            )

            p1.set_label('Mean directional spread')
            ax[3].legend(fontsize='small')
            for axi in ax:
                for item in ([axi.title, axi.xaxis.label, axi.yaxis.label] + axi.get_xticklabels() + axi.get_yticklabels()):
                    item.set_fontsize('x-small')

            plot_file = get_filename(ds, title="time_history", extension="png")
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)
            





