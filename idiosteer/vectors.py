"""Steering-vector constructors: mean-difference, logistic-probe, random. fp32, unit-norm."""
import numpy as np
def build_md_vector(pos_acts, neg_acts):
    d = pos_acts.mean(0) - neg_acts.mean(0); rn = float(np.linalg.norm(d))
    return (d/(rn+1e-8)).astype(np.float32), rn
def build_probe_vector(acts, labels, C=1.0):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, C=C, solver="lbfgs").fit(acts, labels)
    w = clf.coef_[0].astype(np.float32); rn = float(np.linalg.norm(w))
    return (w/(rn+1e-8)), rn, float(clf.score(acts, labels))
def build_random_vector(dim, seed):
    rng = np.random.RandomState(seed); v = rng.randn(dim).astype(np.float32)
    return v/(np.linalg.norm(v)+1e-8)
