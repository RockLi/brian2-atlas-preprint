"""Versioned random-key allocation for the reserved MAM confirmation replicates.

This allocates input keys, not simulation samples or scientific acceptance.
"""
import hashlib
import json

SCHEMA='b2-mam-replicate-random-keys-v1'
SEEDS=(1750,1751,1752,1753,1754)
ROLES=('initial_voltage','runtime_input')


def key(replicate,role,projection=None):
    if type(replicate) is not int or replicate not in SEEDS:
        raise ValueError('unreserved confirmation replicate')
    if role not in (*ROLES,'projection'):
        raise ValueError('unknown random domain')
    if (role=='projection' and (type(projection) is not int or not 0<=projection<8344)) or (role!='projection' and projection is not None):
        raise ValueError('invalid projection identity')
    payload=json.dumps([SCHEMA,replicate,role,projection],separators=(',',':')).encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')


def allocation(replicate,projections=8344):
    if type(projections) is not int or not 1<=projections<=8344:
        raise ValueError('invalid projection count')
    return dict(schema=SCHEMA,replicate=replicate,initial_voltage=key(replicate,'initial_voltage'),
        runtime_input=key(replicate,'runtime_input'),
        projections=[key(replicate,'projection',i) for i in range(projections)])


def registry():
    rows=[allocation(s) for s in SEEDS]
    seen=set();legacy=set(range(1729,1729+8344))
    for r in rows:
        for k in [r['initial_voltage'],r['runtime_input'],*r['projections']]:
            if k==0 or k in seen or k in legacy:
                raise ValueError('allocated random key collides with batch or legacy key')
            seen.add(k)
    encoded=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    return dict(schema=SCHEMA,replicates=list(SEEDS),keys=len(seen),all_keys_unique=True,
        legacy_key_overlap=0,allocation_sha256=hashlib.sha256(encoded).hexdigest(),
        algorithm='SHA-256 of compact ASCII JSON [schema,replicate,domain,projection-or-null]; first 8 bytes unsigned big endian',
        statistical_independence_proven=False,new_simulations=0)
