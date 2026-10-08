import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .mapping import audit
from .stimuli import KINDS, MovieConfig, movie, on_off, sample_hexels


def main():
    p = argparse.ArgumentParser(description="Audit FlyWire visual coordinates and generate diagnostic inputs")
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=783)
    args = p.parse_args()
    report, eligible = audit(args.sources)
    config = MovieConfig()
    stimuli = np.stack([movie(kind, args.seed, config) for kind in KINDS])
    arrays = {"movies": stimuli}
    channels = {}
    for cell_type in ("Mi1", "Tm1"):
        rows = sorted((r for r in eligible if r["cell_type"] == cell_type), key=lambda r: int(r["root_id"]))
        if not rows:
            raise ValueError(f"no concordant mapped {cell_type} cells")
        pq = [[r["p"], r["q"]] for r in rows]
        arrays[cell_type+"_luminance"] = np.stack([sample_hexels(m, pq) for m in stimuli])
        arrays[cell_type+"_on"] = np.stack([sample_hexels(on_off(m)[0], pq) for m in stimuli])
        arrays[cell_type+"_off"] = np.stack([sample_hexels(on_off(m)[1], pq) for m in stimuli])
        channels[cell_type] = rows
    # Immutable run directory; never overwrite research evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"mapping-audit.json").write_text(json.dumps(report, indent=2)+"\n")
    (args.output/"channels.json").write_text(json.dumps(channels, indent=2)+"\n")
    np.savez_compressed(args.output/"diagnostic-inputs.npz", **arrays)
    manifest = dict(schema="flywire-vision-input-diagnostic-v1", seed=args.seed,
                    config=asdict(config), conditions=list(KINDS),
                    channels={k: len(v) for k, v in channels.items()},
                    array_sha256={k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},
                    frame_start_ticks=(np.arange(config.frames)*round(config.frame_ms/config.dt_ms)).tolist(),
                    completed=["source audit", "stimulus generation", "hexel input sampling"],
                    not_completed=["spike/current encoding", "neural simulation", "classification", "GPU validation"],
                    scope="diagnostic only; no accuracy or physiological claims")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"neurons":report["neurons"], "mapping_issues":report["issue_records"],
                      "channels":manifest["channels"], "conditions":len(KINDS), "output":str(args.output)}))


if __name__ == "__main__":
    main()
