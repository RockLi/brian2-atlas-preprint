from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mam_native_v1_140_geometry import (
    PARAMETERS_SHA, SAMPLES, V1_NAMES, derive_groups,
)


def fixture():
    v1_counts = (47386, 13366, 70387, 17597, 20740, 4554, 19839, 4063)
    v1 = [dict(area="V1", population=name.removeprefix("mam_V1_"),
               name="V1-" + name.removeprefix("mam_V1_"), count=count)
          for name, count in zip(V1_NAMES, v1_counts, strict=True)]
    before = [dict(area=f"A{i:03}", population="23E",
                   name=f"A{i:03}-23E", count=2741840 if i == 0 else 1)
              for i in range(182)]
    after = [dict(area=f"Z{i:03}", population="23E",
                  name=f"Z{i:03}-23E", count=1189908 if i == 0 else 1)
             for i in range(64)]
    populations = v1 + before + after  # Native raw order puts V1 first.
    names = sorted(f"mam_{row['area']}_{row['population']}" for row in populations)
    offset = 2742021
    rust_groups = []
    for index, (name, count, wanted) in enumerate(zip(V1_NAMES, v1_counts, SAMPLES,
                                                       strict=True)):
        rust_groups.append(dict(name=name, population_index=182 + index,
                                global_offset=offset, simulated_neurons=count,
                                sample_count=wanted))
        offset += count
    parameters = dict(schema="b2-official-mam-parameters-v1", total_neurons=4129924,
                      N_scaling=1.0, K_scaling=1.0, populations=populations)
    series = dict(schema="b2-mam-modern-paper-time-series-v1",
                  identity=dict(simulator="NEST", seed=1730, ranks=48,
                                threads=4, dt_ms=0.1, nest_version="3.10.0",
                                duration_ms=100500, parameters_sha256=PARAMETERS_SHA),
                  actual_simulated_neurons=4129924, population_names=names)
    return parameters, dict(groups=rust_groups), series


def test_native_raw_offsets_are_not_rust_analysis_offsets():
    parameters, rust, series = fixture()
    result = derive_groups(parameters, rust, series, seed=1730)
    assert result["groups"][0] == (0, 47386, 34)
    assert result["groups"][-1] == (193869, 4063, 3)
    assert result["mapping"][0]["rust_global_offset"] == 2742021
    assert result["native_offsets_equal_rust_offsets"] is False


def test_wrong_rust_offset_or_series_order_rejected():
    parameters, rust, series = fixture()
    rust["groups"][0]["global_offset"] = 0
    with pytest.raises(ValueError, match="native/Rust V1"):
        derive_groups(parameters, rust, series, seed=1730)
    parameters, rust, series = fixture()
    series["population_names"][0], series["population_names"][1] = (
        series["population_names"][1], series["population_names"][0])
    with pytest.raises(ValueError, match="science ordering"):
        derive_groups(parameters, rust, series, seed=1730)
