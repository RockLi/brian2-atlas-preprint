"""Select a pinned confirmation identity once, before importing stage modules.

The explicit environment value is propagated in guarded worker commands. It
selects immutable input identity and paths, never model or estimator settings.
Unsupported replicas fail before remote work. Missing later-stage receipts do
not authorize a launch, audit, or success claim.
"""
import os

_value=os.environ.get('B2_MAM_CONFIRMATION_REPLICATE','1750')
if _value not in ('1750','1751'):
    raise ValueError('unsupported confirmation profile')
REPLICATE=int(_value)
IDENTITY_SHA_1751='eaa1d4a12a666c32760ccb4660d0333e960c57f2b1c5a8417e274f7cf3178e19'
COUNTS_SHA_1751='5e2a5911388cebd11770d1e2554d5b61add8ff55f54abdad065479c4d58a3b97'
DEPLOYMENT_SHA_1751='91e9831747b3d7df8a48aaaef2687d84e31366cee7a9830fe5ed7c62d32211dc'
TOPOLOGY_SHA_1751='fac08d982c402e1ec55464dcf48cd1ebd3b92a1fa94cef370ba780d073751abf'
READINESS_SHA_1751='d6ae475ffc9445537bad4d8e5f03eb6d1c1fcce1b8e901e6aa32ae4452750617'
ADMISSION_SHA_1751='e59aa0c85cae2410040374e4d2c3d770122a80771b4d8482d600c9902f33d6bb'
SOURCE_CATALOG_1751={'mam_confirmation_terminal_sync.py': {'bytes': 10229, 'sha256': '8214031f9f4a85679729428a55c7f4b1ec1b8341641577aac77ba20c1644ad35'}, 'mpi_resource_guard.py': {'bytes': 6287, 'sha256': '630fc35f0b38729fc202925b09b4eea55cc965cfeee396ace6085a3a98004014'}, 'mam_rust_guard23.py': {'bytes': 668, 'sha256': '4ff7fcc2cd12f0b9448fb616c649b17c30ae3ed59f0cb345b05a03c2eca5fcfe'}, 'protocol.json': {'bytes': 5073, 'sha256': '0797530db98fe99f97bfec7d48fc284c877d0c4c4b8f8fa135e5af129e9e3054'}, 'identity.json': {'bytes': 236966, 'sha256': 'eaa1d4a12a666c32760ccb4660d0333e960c57f2b1c5a8417e274f7cf3178e19'}}
RAW_PREPARATION_SHA_1751='769a0e37fd4cf79d4ff8508171c8c54410523774763063283f76fde928b37909'


def selected(legacy,new):
    return legacy if REPLICATE==1750 else new


def worker_environment():
    return [] if REPLICATE==1750 else ['B2_MAM_CONFIRMATION_REPLICATE=1751']
