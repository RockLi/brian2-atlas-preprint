"""Partition the pinned whole graph into disjoint static and MBON01 plastic edges."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from audit_connectome import audit, CSR

PREPARED_FILES = ('static.b2csr', 'plastic.npz', 'groups.npz',
                  'static_edge_ids.npy', 'subgraph.npz')


def write_csr(path, n, offsets, targets, contacts):
    with path.open('wb') as f:
        f.write(CSR['HEADER'].pack(CSR['MAGIC'], n, n, len(targets), 1))
        f.write(np.asarray(offsets, dtype='<u8').tobytes())
        f.write(np.asarray(targets, dtype='<u4').tobytes())
        f.write(np.asarray(contacts, dtype='<f8').tobytes())
    CSR['inspect_csr'](path)


def prepare(graph, annotations, output):
    graph, output = Path(graph).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit_report = audit(graph, annotations, output/'audit')
    with Path(annotations).open() as f:
        rows = sorted(csv.DictReader(f, delimiter='\t'), key=lambda r: int(r['root_id']))
    ids = np.load(graph/'root_ids.npy', allow_pickle=False)
    mbons = np.array([i for i,r in enumerate(rows) if r['cell_class']=='MBON' and r['hemibrain_type']=='MBON01'], dtype=np.int32)
    assert ids[mbons].tolist() == [720575940624117245, 720575940643309197]
    kc = np.array([i for i,r in enumerate(rows) if r['cell_class']=='Kenyon_Cell'], dtype=np.int32)
    orn = {name: np.array([i for i,r in enumerate(rows) if r['cell_class']=='olfactory' and
                           name in (r['cell_type'],r['hemibrain_type'])], dtype=np.int32)
           for name in ('ORN_DM1','ORN_DM2')}
    info = CSR['inspect_csr'](graph/'connectome.b2csr')
    offsets, targets, contacts = CSR['csr_arrays'](info)
    with np.load(output/'audit/kc_mbon_edges.npz') as edges:
        selected = np.isin(edges['target_index'], mbons)
        plastic = {key:edges[key][selected] for key in edges.files}
    edge_id = plastic['edge_id']
    assert len(edge_id) > 0 and np.all(plastic['signed_contacts'] > 0)
    keep = np.ones(info['edge_count'], dtype=bool)
    keep[edge_id] = False
    static_ids = np.flatnonzero(keep)
    degrees = np.diff(offsets).astype(np.int64)-np.bincount(plastic['source_index'], minlength=len(ids))
    static_offsets = np.r_[0,np.cumsum(degrees)]
    write_csr(output/'static.b2csr', len(ids), static_offsets, targets[keep], contacts[0,keep])
    np.savez(output/'plastic.npz', **plastic)
    np.savez(output/'groups.npz', root_ids=ids, kc=kc, mbon=mbons, **orn)
    np.save(output/'static_edge_ids.npy', static_ids)
    # Independent inverse reconstruction by original edge identity.
    recovered_targets = np.empty(info['edge_count'], dtype=np.uint32)
    recovered_contacts = np.empty(info['edge_count'])
    so, st, sw = CSR['csr_arrays'](CSR['inspect_csr'](output/'static.b2csr'))
    recovered_targets[static_ids], recovered_targets[edge_id] = st, plastic['target_index']
    recovered_contacts[static_ids], recovered_contacts[edge_id] = sw[0], plastic['signed_contacts']
    np.testing.assert_array_equal(recovered_targets, targets)
    np.testing.assert_array_equal(recovered_contacts, contacts[0])
    np.testing.assert_array_equal(np.diff(so)+np.bincount(plastic['source_index'],minlength=len(ids)),np.diff(offsets))
    assert len(static_ids)+len(edge_id)==info['edge_count'] and not np.any(keep[edge_id])
    # Real induced subgraph includes every KC supplying MBON01 and both outputs.
    nodes = np.unique(np.r_[plastic['source_index'],mbons]).astype(np.int32)
    local = np.full(len(ids), -1, dtype=np.int32)
    local[nodes] = np.arange(len(nodes))
    sub_pre, sub_post, sub_contacts = [], [], []
    for source in nodes:
        start, end = int(offsets[source]),int(offsets[source+1])
        selected = (local[targets[start:end]]>=0)&keep[start:end]
        sub_pre.extend([int(local[source])]*int(selected.sum()))
        sub_post.extend(local[targets[start:end][selected]].tolist())
        sub_contacts.extend(contacts[0,start:end][selected].tolist())
    np.savez(output/'subgraph.npz', nodes=nodes, source=np.array(sub_pre,dtype=np.int32),
             target=np.array(sub_post,dtype=np.int32), contacts=np.array(sub_contacts),
             plastic_source=local[plastic['source_index']], plastic_target=local[plastic['target_index']],
             plastic_contacts=plastic['signed_contacts'], mbon=local[mbons])
    report = dict(source=audit_report, neurons=len(ids), static_edges=len(static_ids),
                  plastic_edges=len(edge_id), plastic_contacts=int(plastic['signed_contacts'].sum()),
                  mbon_root_ids=ids[mbons].tolist(), subgraph_neurons=len(nodes),
                  subgraph_static_edges=len(sub_pre), partition_reconstructs_original_exactly=True,
                  selection='All positive KC inputs to both exact MBON01 annotations; pooled gamma5/beta-prime2a',
                  orn_counts={name:len(group) for name,group in orn.items()},
                  # Hash only scientific payloads, not filesystem metadata such
                  # as macOS AppleDouble ._* files on external volumes.
                  hashes={name:CSR['file_hash'](output/name) for name in PREPARED_FILES})
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source','hashes')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('graph','annotations','output'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.graph,a.annotations,a.output)
