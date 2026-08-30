import os, sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"]=42; matplotlib.rcParams["ps.fonttype"]=42

KEY = sys.argv[1] if len(sys.argv) > 1 else 'llama3b'
D = os.path.join(os.environ.get('IDIOSTEER_ROOT', os.getcwd()), 'assets')
df = pd.read_csv(os.path.join(D, f'layer_vs_accuracy_{KEY}.csv'))
CANON = {'llama3b': 14, 'gemma2_2b': 20, 'llama8b': 16, 'gemma3_4b': 17, 'gemma3_12b': 24}

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(df['hidden_state_layer'], df['cv_accuracy'], '-o', color='#1f77b4', lw=2, ms=4, label='CV accuracy')
ax.fill_between(df['hidden_state_layer'], df['cv_accuracy'] - df['cv_std'],
                df['cv_accuracy'] + df['cv_std'], color='#1f77b4', alpha=0.15, label='±1 SD (folds)')
peak = df.loc[df['cv_accuracy'].idxmax()]
ax.scatter([peak['hidden_state_layer']], [peak['cv_accuracy']], color='crimson', zorder=5,
           label=f"peak hs[{int(peak['hidden_state_layer'])}] = {peak['cv_accuracy']:.3f}")
cl = CANON.get(KEY)
if cl is not None:
    rC = df[df.hidden_state_layer == cl]
    if len(rC):
        ax.scatter([cl], [rC['cv_accuracy'].iloc[0]], color='green', zorder=5, marker='s',
                   label=f"canonical hs[{cl}] = {rC['cv_accuracy'].iloc[0]:.3f}")
if KEY == 'llama3b':
    ax.axhline(0.975, ls='--', color='gray', lw=1, label='Table ref 0.975')
ax.set_xlabel('hidden_states layer index'); ax.set_ylabel('5-fold StratifiedGroupKFold-by-idiom accuracy')
ax.set_ylim(min(0.5, df.cv_accuracy.min() - 0.05), 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout()
out = os.path.join(D, f'layer_vs_accuracy_{KEY}.png')
fig.savefig(out, dpi=150); fig.savefig(out.replace('.png', '.pdf'))
print('saved', KEY, '| canonical hs[%s]' % cl)
