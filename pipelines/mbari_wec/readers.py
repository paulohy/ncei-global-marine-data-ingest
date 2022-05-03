from typing import Dict, Union
from pydantic import BaseModel, Extra
import xarray as xr
from tsdat import DataReader
import os
import getpass
from datetime import datetime
import pandas as pd
import numpy as np
from mhkit.wave.resource import energy_flux, energy_period
from mhkit.wave.performance import capture_length_matrix


class MbariWecDataReader(DataReader):
    """---------------------------------------------------------------------------------
    Custom Data Reader for MBARI WEC 2021 dataset.

    Implementation based on https://github.com/SNL-WaterPower/fbWecCntrl/blob/master/mbari_wec/mbari_wec_2021_example.py

    ---------------------------------------------------------------------------------"""

    class Parameters(BaseModel, extra=Extra.forbid):
        """
        density [km/m^3]: water density
        depth [m]: depth
        gravity [m/s^2]: earth's gravity acceleration
        """

        density: float = 1025.0
        depth: float = 100.0
        gravity: float = 9.81

    parameters: Parameters = Parameters()
    """Extra parameters that can be set via the retrieval configuration file. If you opt
    to not use any configuration parameters then please remove the code above."""

    def read(self, input_key: str) -> Union[xr.Dataset, Dict[str, xr.Dataset]]:
        file_name = input_key
        rho = self.parameters.density
        h = self.parameters.depth
        g = self.parameters.gravity

        #%% Read bulk parameters
        date_parser = lambda epoch: pd.to_datetime(epoch, unit="s")
        dat = pd.read_csv(
            file_name,
            index_col=3,
            usecols=np.insert(np.arange(13), -1, [364, 365, 366]),
            parse_dates=["Epoch Time"],
            date_parser=date_parser,
        )
        b = dat.to_xarray()

        #%% Frequency array
        dat1 = pd.read_csv(
            file_name,
            index_col=[],
            usecols=np.arange(13, 13 + 38 + 1),
        )
        freq_array = dat1.iloc[0].to_xarray()
        freq_array.name = "Frequency"

        dat2 = pd.read_csv(
            file_name,
            index_col=[],
            usecols=np.arange(13 + 38 + 1, 13 + 2 * (38 + 1)),
        )
        df_array = (
            dat2.iloc[0]
            .to_xarray()
            .assign_coords(dict(index=freq_array.values))
            .rename(dict(index="Frequency"))
        )
        df_array.name = "df"
        df_array.attrs["long_name"] = "Frequency spacing"
        df_array.attrs["units"] = "Hz"

        #%% a and b parameters
        names = ["a1", "b1", "a2", "b2"]
        tmp_list = []
        for idx, name in enumerate(names):
            dat_tmp = pd.read_csv(
                file_name,
                index_col=[0],
                usecols=np.insert(
                    np.arange(13 + (2 + idx) * (38 + 1), 13 + (3 + idx) * (38 + 1)),
                    0,
                    3,
                ),
                date_parser=date_parser,
            )
            tmp_da = dat_tmp.to_xarray().to_array(dim="Frequency", name=name)
            tmp_da = tmp_da.assign_coords({"Frequency": freq_array.values})
            tmp_list.append(tmp_da)

        ab_ds = xr.merge(tmp_list)

        #%% Spectral density, spreading, etc.
        dat_S = pd.read_csv(
            file_name,
            index_col=[0],
            usecols=np.insert(np.arange(13 + 6 * (38 + 1), 13 + 7 * (38 + 1)), 0, 3),
            date_parser=date_parser,
        )
        S = dat_S.to_xarray().to_array(dim="Frequency", name="Variance density")
        S = S.assign_coords({"Frequency": freq_array.values})

        dat_dir = pd.read_csv(
            file_name,
            index_col=[0],
            usecols=np.insert(np.arange(13 + 7 * (38 + 1), 13 + 8 * (38 + 1)), 0, 3),
            date_parser=date_parser,
        )
        Dir = dat_dir.to_xarray().to_array(dim="Frequency", name="Direction")
        Dir = Dir.assign_coords({"Frequency": freq_array.values})

        dat_spread = pd.read_csv(
            file_name,
            index_col=[0],
            usecols=np.insert(np.arange(13 + 8 * (38 + 1), 13 + 9 * (38 + 1)), 0, 3),
            date_parser=date_parser,
        )
        spread = dat_spread.to_xarray().to_array(
            dim="Frequency", name="Directional spread"
        )
        spread = spread.assign_coords({"Frequency": freq_array.values})

        #%% Combine, clean up

        da = xr.merge([b, ab_ds, S, Dir, spread, df_array])
        da["Battery Voltage (V)"].attrs["units"] = "V"
        da["Battery Voltage (V)"].attrs["long_name"] = "Battery voltage"

        da["Power (W)"].attrs["units"] = "W"
        da["Power (W)"].attrs["long_name"] = "Battery power"

        da["Humidity (%rel)"].attrs["units"] = "1"
        da["Humidity (%rel)"].attrs["standard_name"] = "relative_humidity"
        da["Humidity (%rel)"].attrs["long_name"] = "Relative humidity"

        da["Significant Wave Height (m)"].attrs["units"] = "m"
        da["Significant Wave Height (m)"].attrs[
            "standard_name"
        ] = "sea_surface_wave_significant_height"
        da["Significant Wave Height (m)"].attrs["long_name"] = "Significant wave height"

        da["Direction"].attrs["units"] = "degree"
        da["Direction"].attrs["long_name"] = ""

        da["Peak Period (s)"].attrs["units"] = "s"
        da["Peak Period (s)"].attrs[
            "standard_name"
        ] = "sea_surface_wave_period_at_variance_spectral_density_maximum"
        da["Peak Period (s)"].attrs["long_name"] = "Peak period"

        da["Mean Period (s)"].attrs["units"] = "s"
        da["Mean Period (s)"].attrs[
            "standard_name"
        ] = "sea_surface_wave_zero_upcrossing_period"
        da["Mean Period (s)"].attrs["long_name"] = "Mean period"

        da["Peak Direction (deg)"].attrs["units"] = "degree"
        da["Peak Direction (deg)"].attrs[
            "standard_name"
        ] = "sea_surface_wave_from_direction_at_variance_spectral_density_maximum"
        da["Peak Direction (deg)"].attrs["long_name"] = "Peak direction"

        da["Peak Directional Spread (deg)"].attrs["units"] = "degree"
        da["Peak Directional Spread (deg)"].attrs[
            "standard_name"
        ] = "sea_surface_wave_directional_spread_at_variance_spectral_density_maximum"
        da["Peak Directional Spread (deg)"].attrs[
            "long_name"
        ] = "Peak directional spread"

        da["Mean Direction (deg)"].attrs["units"] = "degree"
        da["Mean Direction (deg)"].attrs[
            "standard_name"
        ] = "sea_surface_wave_from_direction"
        da["Mean Direction (deg)"].attrs["long_name"] = "Mean direction"

        da["Mean Directional Spread (deg)"].attrs["units"] = "degree"
        da["Mean Directional Spread (deg)"].attrs[
            "long_name"
        ] = "Mean directional spread"

        da["Latitude (deg)"].attrs["units"] = "degree_north"
        da["Latitude (deg)"].attrs["standard_name"] = "latitude"
        da["Latitude (deg)"].attrs["long_name"] = "Latitude"

        da["Longitude (deg)"].attrs["units"] = "degree_east"
        da["Longitude (deg)"].attrs["standard_name"] = "longitude"
        da["Longitude (deg)"].attrs["long_name"] = "Longitude"

        da["Wind Speed (m/s)"].attrs["units"] = "m/s"
        da["Wind Speed (m/s)"].attrs["standard_name"] = "wind_speed"
        da["Wind Speed (m/s)"].attrs["long_name"] = "Wind speed"

        da["Wind Direction (deg)"].attrs["units"] = "degree"
        da["Wind Direction (deg)"].attrs["standard_name"] = "wind_from_direction"
        da["Wind Direction (deg)"].attrs["long_name"] = "Wind direction"

        da["Surface Temperature (°C)"] = 274.15 * da["Surface Temperature (°C)"]
        da["Surface Temperature (°C)"].attrs["units"] = "K"
        da["Surface Temperature (°C)"].attrs[
            "standard_name"
        ] = "sea_surface_temperature"
        da["Surface Temperature (°C)"].attrs["long_name"] = "Surface temperature"

        da["Frequency"].attrs["units"] = "Hz"
        da["Frequency"].attrs["standard_name"] = "wave_frequency"
        da["Frequency"].attrs["long_name"] = "Frequency"

        da["Variance density"].attrs["units"] = "m^2/Hz"
        da["Variance density"].attrs[
            "standard_name"
        ] = "sea_surface_wave_variance_spectral_density"
        da["Variance density"].attrs["long_name"] = "Spectral density"

        da["Directional spread"].attrs["units"] = "degree"
        da["Directional spread"].attrs[
            "standard_name"
        ] = "sea_surface_wave_directional_spread"
        da["Directional spread"].attrs["long_name"] = "Directional spreading"

        da = da.rename(
            {
                "Epoch Time": "time",
                "Frequency": "freq",
                "Battery Voltage (V)": "batter_voltage",
                "Variance density": "S",
                "Direction": "wave_dir",
                "Power (W)": "battery_power",
                "Humidity (%rel)": "humidity",
                "Significant Wave Height (m)": "Hm0",
                "Peak Period (s)": "Tp",
                "Mean Period (s)": "Tm",
                "Peak Direction (deg)": "peak_dir",
                "Peak Directional Spread (deg)": "peak_spread",
                "Mean Direction (deg)": "mean_dir",
                "Mean Directional Spread (deg)": "mean_spread",
                "Directional spread": "spread",
                "Latitude (deg)": "spot_lat",
                "Longitude (deg)": "spot_lon",
                "Wind Speed (m/s)": "wind_speed",
                "Wind Direction (deg)": "wind_dir",
                "Surface Temperature (°C)": "temperature",
            }
        )

        J = xr.DataArray(
            np.array(
                [
                    energy_flux(
                        da.isel(time=idx)["S"].to_pandas(), h=h, rho=rho
                    ).values[0][0]
                    for idx in range(len(da.time))
                ]
            ),
            dims="time",
            name="J",
        ).assign_coords(dict(time=da.time.values))
        J.attrs["units"] = "W/m"
        J.attrs["long_name"] = "Energy flux"

        Te = xr.DataArray(
            np.array(
                [
                    energy_period(
                        da.isel(time=idx)["S"].to_pandas(),
                    )
                    for idx in range(len(da.time))
                ]
            ).squeeze(),
            dims="time",
            name="Te",
        ).assign_coords(dict(time=da.time.values))
        Te.attrs["units"] = "s"
        Te.attrs["long_name"] = "Energy period"

        da = xr.merge([da, J, Te])

        da.time.attrs["long_name"] = "Epoch time"

        da.attrs[
            "institution"
        ] = "Sandia National Laboratories and Monterey Bay Aquarium Research Institute"
        da.attrs["Conventions"] = "CF-1.8"
        da.attrs["title"] = file_name
        da.attrs["source"] = "Sofar spotter buoy"
        da.attrs["history"] = "generated {:} by {:}".format(
            datetime.now().strftime("%Y-%m-%d @ %H:%M:%S"), getpass.getuser()
        )
        da.attrs[
            "references"
        ] = "https://content.sofarocean.com/hubfs/Technical_Reference_Manual.pdf"
        da = da.sortby("time")
        da = da.drop_isel(time=0)  # first sample appears anomalous

        ds = da

        return ds
