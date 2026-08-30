import os, json, numpy as np
V=os.path.expanduser("./concreteness/vectors/llama3b")
OUT=os.path.expanduser("./concreteness/tier3/vectors"); os.makedirs(OUT,exist_ok=True)
def u(x): return x/(np.linalg.norm(x)+1e-8)
v_idiom=np.load(f"{V}/v_IDIOM.npy").astype(np.float64)
v_gj=np.load(f"{V}/v_GJ.npy").astype(np.float64)
v_bry=np.load(f"{V}/v_BRY.npy").astype(np.float64)
# C = orthonormal basis (QR) of span{v_GJ, v_BRY}
A=np.column_stack([v_gj,v_bry])
Q,_=np.linalg.qr(A)                       # Q: [d,2] orthonormal
proj=Q@(Q.T@v_idiom)                      # projection of v_IDIOM onto span
resid=v_idiom-proj
n_proj=float(np.linalg.norm(proj)); n_resid=float(np.linalg.norm(resid))
v_par=u(proj); v_perp=u(resid)
def cos(a,b): return float(np.dot(u(a),u(b)))
res={
 "||proj_C(v_IDIOM)|| (before renorm)": round(n_proj,4),
 "||v_IDIOM - proj_C|| (before renorm)": round(n_resid,4),
 "check ||v_IDIOM||": round(float(np.linalg.norm(v_idiom)),4),
 "cos(v_PERP, v_IDIOM)": round(cos(v_perp,v_idiom),4),
 "cos(v_PAR, v_BRY)": round(cos(v_par,v_bry),4),
 "cos(v_PAR, v_GJ)": round(cos(v_par,v_gj),4),
 "cos(v_PERP, v_PAR) [should~0]": round(cos(v_perp,v_par),4),
 "cos(v_GJ,v_BRY)": round(cos(v_gj,v_bry),4),
 "v_PAR_positive_pole": "abstract (figurative-ward); literal steering = negative alpha",
}
np.save(f"{OUT}/v_PERP.npy", v_perp.astype(np.float32))
np.save(f"{OUT}/v_PAR.npy",  v_par.astype(np.float32))
np.save(f"{OUT}/C_basis.npy", Q.astype(np.float32))
json.dump({k:res[k] for k in res}, open(f"{OUT}/geometry.json","w"), indent=2)
print(json.dumps(res, indent=2))
print("saved:", OUT, "-> v_PERP.npy, v_PAR.npy, C_basis.npy, geometry.json")
