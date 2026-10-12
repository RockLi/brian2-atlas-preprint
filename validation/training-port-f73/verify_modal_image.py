"""Build the public Atlas Modal image and check its installed backend imports."""
import argparse
import json
from pathlib import Path
import sys


def inspect_install():
    import brian2
    import brian2_atlas
    from brian2_rust._runtime import executable_path
    return {'brian_version':brian2.__version__, 'training_api':hasattr(brian2_atlas,'NativeLIFTrainer'),
            'simulation_binary':executable_path('b2-runner').is_file(),
            'training_binary':executable_path('b2-train').is_file()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--atlas',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.atlas.resolve();sys.path[:0]=[str(root/'brian2-rust/examples'),str(root/'brian2-rust/python'),str(root)]
    from modal_cuda_tests import native_image
    import modal
    image=native_image().add_local_file(Path(__file__),'/root/verify_modal_image.py')
    app=modal.App('atlas-training-f73-image-check',include_source=False)
    run=app.function(image=image,cpu=1,memory=2048,timeout=120,retries=0)(inspect_install)
    with modal.enable_output(),app.run():
        print('APP_ID='+app.app_id,flush=True);report=run.remote()
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    assert all(report[k] for k in ('training_api','simulation_binary','training_binary'))

if __name__=='__main__':main()
