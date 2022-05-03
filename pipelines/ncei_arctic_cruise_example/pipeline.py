import xarray as xr
import cmocean
import matplotlib.pyplot as plt

from tsdat import IngestPipeline, get_start_date_and_time_str, get_filename
from utils import format_time_xticks


class NceiArcticCruiseExample(IngestPipeline):
    """---------------------------------------------------------------------------------
      NCEI ARCTIC CRUISE EXAMPLE INGESTION PIPELINE

      Historical marine data that are comprised of ship, buoy, and platform
      observations.
    ---------------------------------------------------------------------------------"""

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset before qc is applied
        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset after qc is applied
        # but before it gets saved to the storage area
        return dataset

    def hook_plot_dataset(self, dataset: xr.Dataset):
        location = self.dataset_config.attrs.location_id
        datastream: str = self.dataset_config.attrs.datastream

        date, time = get_start_date_and_time_str(dataset)

        plt.style.use("default")  # clear any styles that were set before
        plt.style.use("shared/styling.mplstyle")

        with self.storage.uploadable_dir(datastream) as tmp_dir:

            fig, ax = plt.subplots()
            dataset["pressure"].plot(ax=ax, x="time", c=cmocean.cm.deep_r(0.5))
            fig.suptitle(f"Pressure Observations from at {location} on {date} {time}")
            format_time_xticks(ax)

            plot_file = get_filename(dataset, title="example_plot", extension="png")
            fig.savefig(tmp_dir / plot_file)
            plt.close(fig)
