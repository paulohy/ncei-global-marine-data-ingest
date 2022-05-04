import xarray as xr
from pathlib import Path
from tsdat import PipelineConfig, assert_close

# DEVELOPER: Update paths to your configuration(s), test input(s), and expected test
# results files.
def test_mbari_wec_pipeline():
    config_path = Path("pipelines/mbari_wec/config/pipeline.yaml")
    config = PipelineConfig.from_yaml(config_path)
    pipeline = config.instantiate_pipeline()

    test_file = "pipelines/mbari_wec/test/data/input/monterrey_bay.sample_data.csv"
    expected_file = (
        "pipelines/mbari_wec/test/data/expected/monterrey_bay.mbari_wec.a1.20210903.191201.nc"
    )

    dataset = pipeline.run([test_file])
    expected: xr.Dataset = xr.open_dataset(expected_file)  # type: ignore
    assert_close(dataset, expected, check_attrs=False)
