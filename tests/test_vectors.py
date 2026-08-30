import numpy as np
from idiosteer.vectors import build_md_vector, build_random_vector
def test_md_unit_norm():
    rng=np.random.RandomState(0); pos=rng.randn(50,16)+1; neg=rng.randn(50,16)
    v,raw=build_md_vector(pos,neg)
    assert abs(np.linalg.norm(v)-1.0)<1e-5 and raw>0
def test_random_seed_determinism():
    a=build_random_vector(32,7); b=build_random_vector(32,7); c=build_random_vector(32,8)
    assert np.allclose(a,b) and not np.allclose(a,c)
    assert abs(np.linalg.norm(a)-1.0)<1e-5
