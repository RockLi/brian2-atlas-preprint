"""Copy checksum-verified official MNIST test IDX files into static UI assets."""
import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path

FILES = {
    'images': ('t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3', (2051, 10000, 28, 28)),
    'labels': ('t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c', (2049, 10000)),
}

def package(data, output):
    files = {}
    payloads = {}
    for kind, (name, md5, shape) in FILES.items():
        packed = (data / name).read_bytes()
        if hashlib.md5(packed).hexdigest() != md5:
            raise ValueError('MNIST checksum mismatch: ' + name)
        raw = gzip.decompress(packed)
        header = 4 * len(shape)
        if struct.unpack('>' + 'I' * len(shape), raw[:header]) != shape:
            raise ValueError('Invalid IDX dimensions')
        count = 10000 * (784 if kind == 'images' else 1)
        if len(raw) != header + count or (kind == 'labels' and max(raw[header:]) > 9):
            raise ValueError('Invalid IDX payload')
        url = 'mnist-test-' + kind + '.bin'
        payloads[url] = packed
        files[kind] = dict(url=url, encoding='gzip', source='https://ossci-datasets.s3.amazonaws.com/mnist/' + name,
                           md5=md5, sha256=hashlib.sha256(packed).hexdigest(), bytes=len(packed),
                           raw_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=len(raw))
    output.mkdir(parents=True, exist_ok=True)
    for name, packed in payloads.items():
        (output / name).write_bytes(packed)
    manifest = dict(schema='mnist-test-idx-v1', split='test', count=10000, rows=28, columns=28,
                    source='Official MNIST test split', authors='Yann LeCun, Corinna Cortes and Christopher J. C. Burges',
                    sample_ids='Zero-based original test IDX position (0–9999)', files=files)
    (output / 'mnist-test.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package(args.data, args.output)
