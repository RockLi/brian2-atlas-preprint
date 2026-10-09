# Skaar NMDA experiment programs

These are the first-party experiment drivers and analysis scripts supporting manuscript §4.7 / §5.8.1, Figure 8 and Table 4. They were previously retained in an independent study directory and are now preserved here byte-for-byte. The original nested layout keeps sibling imports and study-relative paths intact.

Start with `benchmarks/published/nmda_skaar_2025/scripts/run_rust_fixture.py`, `run_cpp_timed_fixture.py`, `prepare_cpu_frozen_model.py`, `measure_cpu_frozen.py`, and `prepare_mpi_frozen_model.py`. Analysis includes `compare_compiled_outputs.py`, `summarize_scale5120.py`, `summarize_scale10240.py`, `summarize_scale20480_native.py`, and `summarize_cpu_threads.py`.

Use an explicitly selected Atlas source/environment or installed package. Historical `brian2_rust` imports remain supported; adding this directory does not rerun the recorded measurements on the latest engine. The scripts retain their original explicit paths and launch conventions. Configure inputs and destinations before running campaign scripts. Remote launch and GPU scripts are preserved as source; they are not executed by the coverage check.

The external authors' model is not vendored. Its exact source revision and acquisition command are in [SOURCE.md](benchmarks/published/nmda_skaar_2025/SOURCE.md) and [fetch_upstream.sh](benchmarks/published/nmda_skaar_2025/scripts/fetch_upstream.sh). The recorded upstream revision has no license file. Fetch it directly and retain its revision identity.

The paper's measured summaries remain in [published-model evidence](../../../paper/data/published_models/evidence.json). Additional GPU/decision-network scripts are retained with the study but do not add claims to the Atlas manuscript. Large output arrays, private operational logs and the full study directory are not part of this code import. File hashes and provenance are in [the import manifest](../../../migration/paper-experiment-code-import-20261009.json).
