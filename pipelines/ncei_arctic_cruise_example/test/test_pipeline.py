import xarray as xr
from pathlib import Path
from tsdat import PipelineConfig, assert_close

# DEVELOPER: Update paths to your configuration(s), test input(s), and expected test
# results files.
def test_ncei_arctic_cruise_example_pipeline():
    config_path = Path("pipelines/ncei_arctic_cruise_example/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = "pipelines/ncei_arctic_cruise_example/test/data/input/arctic_ocean.sample_data.csv"
    expected_file = "pipelines/ncei_arctic_cruise_example/test/data/expected/arctic_ocean.ncei_arctic_cruise_example.a1.20150112.000000.nc"

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)
