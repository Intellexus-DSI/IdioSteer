import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["ps.fonttype"]=42
CR=os.path.join(os.environ.get("IDIOSTEER_ROOT", os.getcwd()), "assets")
FLAG={"out of the blue","behind bars","spill the beans","in the pink"}
p7=pd.read_csv(f"{CR}/p7_per_idiom_components.csv").sort_values("vBRY_minus_vPERP")
x=np.arange(len(p7)); w=0.26
fig,ax=plt.subplots(figsize=(14,6))
LBL={"vIDIOM":r"$v_{\mathrm{MD}}$","vPERP":r"$v_{\perp}$","vBRY":r"$v_{\mathrm{conc}}$"}  # CSV cols -> paper notation
for off,col,c in [(-w,"vIDIOM","#4c72b0"),(0,"vPERP","#dd8452"),(w,"vBRY","#55a868")]:
    ax.bar(x+off,p7[col],w,label=LBL[col],color=c,yerr=[p7[col]-p7[col+"_lo"],p7[col+"_hi"]-p7[col]],capsize=2,ecolor="#333",error_kw={"lw":0.7})
ax.scatter(x,p7["baseline"],marker="_",s=220,color="black",zorder=6,label="baseline")
ax.set_xticks(x); labs=ax.set_xticklabels(p7["idiom"],rotation=55,ha="right",fontsize=8)
for t,idi in zip(labs,p7["idiom"]):
    if idi in FLAG: t.set_fontweight("bold"); t.set_color("#b22222")
ax.set_ylabel("literal rate (coherent-only)"); ax.set_ylim(0,1)
ax.legend(ncol=4,fontsize=11); ax.grid(axis="y",alpha=0.3); fig.tight_layout()
fig.savefig(f"{CR}/p7_per_idiom_components.pdf"); fig.savefig(f"{CR}/p7_per_idiom_components.png", dpi=140); print("saved",f"{CR}/p7_per_idiom_components.pdf")
