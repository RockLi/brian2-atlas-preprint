"""Independently compare v4 output with v3, expanding only bounded spike blocks.

No reconstructed output file is created. All non-spike bytes are compared
unchanged; only the result version/declared size and event magic may differ.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

SIZES = {'bool': 1, 'f32': 4, 'f64': 8, 'i32': 4, 'i64': 8, 'u32': 4, 'u64': 8}
BLOCK = 2**20


class Reader:
    def __init__(self, path):
        self.path = path
        self.stream = path.open('rb')
        self.digest = hashlib.sha256()
        self.size = path.stat().st_size
        self.at = 0

    def take(self, size):
        if not 0 <= size <= BLOCK:
            raise ValueError('unbounded comparison block')
        raw = self.stream.read(size)
        if len(raw) != size:
            raise ValueError('truncated comparison input')
        self.digest.update(raw)
        self.at += size
        return raw

    def finish(self):
        if self.at != self.size or self.stream.read(1):
            raise ValueError('unexpected comparison trailer')
        return dict(bytes=self.size, sha256=self.digest.hexdigest())


def identical(a, b, size):
    while size:
        count = min(size, BLOCK)
        if a.take(count) != b.take(count):
            raise ValueError('non-spike bytes differ')
        size -= count


def spikes(a, b, count):
    while count:
        n = min(count, BLOCK // 16)
        wide = a.take(n * 16)
        narrow = np.frombuffer(b.take(n * 8), dtype='<u4')
        if wide != narrow.astype('<i8').tobytes():
            raise ValueError('canonical spike bytes differ')
        count -= n


def compare_pair(model, wide_dir, compact_dir):
    result = dict(schema='mam-compact-output-byte-comparison-v1', complete=False,
                  canonical_v3_bytes_exact=True, reconstructed_files_created=False, files={})
    for filename in ['results.bin', 'events.bin']:
        if filename == 'events.bin' and not any(p.get('events') for p in model['definition']['populations']):
            if (wide_dir/filename).exists() or (compact_dir/filename).exists():
                raise ValueError('unexpected event sidecar')
            continue
        a, b = Reader(wide_dir/filename), Reader(compact_dir/filename)
        count = 0
        try:
            if filename == 'results.bin':
                wh = struct.unpack('<8sIIQQQ', a.take(40))
                ch = struct.unpack('<8sIIQQQ', b.take(40))
                expected = (b'B2DMP001', 0x01020304, len(model['definition']['populations']), model['instance']['neuron_count'])
                if (wh[0], wh[2], wh[3], wh[4]) != expected or (ch[0], ch[2], ch[3], ch[4]) != expected or wh[1] != 3 or ch[1] != 4 or wh[5] != a.size or ch[5] != b.size:
                    raise ValueError('invalid result comparison header')
                for pop in model['definition']['populations']:
                    raw = a.take(64)
                    if raw != b.take(64):
                        raise ValueError('population headers differ')
                    fields = struct.unpack('<8Q', raw)
                    n, steps, recorded, variables, states, spike_count, last_count, flags = fields
                    if (n, steps, recorded, variables, states, flags) != (pop['count'], pop['steps'], len(pop['monitor']['record']), len(pop['monitor']['variables']), len(pop['states']), int(pop['refractory'] is not None)):
                        raise ValueError('population dimensions differ from model')
                    symbols = {s['name']:s for s in pop['states'] + pop['parameters'] + pop.get('linked_variables', [])}
                    trace = recorded * pop['monitor']['window_steps'] * sum(SIZES[symbols[name]['dtype']] for name in pop['monitor']['variables'])
                    identical(a, b, trace)
                    spikes(a, b, spike_count)
                    count += spike_count
                    rest = 8*n + 8*last_count + n*sum(SIZES[s['dtype']] for s in pop['states']) + (9*n if flags else 0)
                    identical(a, b, rest)
                if a.size-a.at != b.size-b.at:
                    raise ValueError('result tail length differs')
                identical(a, b, a.size-a.at)
            else:
                if a.take(8) != b'B2EVT001' or b.take(8) != b'B2EVT002':
                    raise ValueError('event version mismatch')
                raw = a.take(8)
                if raw != b.take(8):
                    raise ValueError('event stream count differs')
                streams = struct.unpack('<Q', raw)[0]
                pops = model['definition']['populations']
                if any(p.get('events', []) not in ([], ['spike']) or p.get('event_monitors') for p in pops) or streams != sum(len(p.get('events', [])) for p in pops):
                    raise ValueError('unsupported event streams')
                for _ in range(streams):
                    raw = a.take(8)
                    if raw != b.take(8):
                        raise ValueError('event lengths differ')
                    n = struct.unpack('<Q', raw)[0]
                    spikes(a, b, n)
                    count += n
                if a.take(8) != b'\0'*8 or b.take(8) != b'\0'*8:
                    raise ValueError('event monitors not supported')
                if a.take(8) != b'B2EEND01' or b.take(8) != b'B2EEND01':
                    raise ValueError('event trailer differs')
            if a.size-b.size != 8*count:
                raise ValueError('compact byte saving differs from record count')
            result['files'][filename] = dict(wide=a.finish(), compact=b.finish(), records=count,
                                             bytes_saved=8*count, canonical_v3_bytes_exact=True)
        finally:
            a.stream.close()
            b.stream.close()
    result['complete'] = True
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--wide', required=True, type=Path)
    parser.add_argument('--compact', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    report = compare_pair(json.loads(args.model.read_text()), args.wide, args.compact)
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
