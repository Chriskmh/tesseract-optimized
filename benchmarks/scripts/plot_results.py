"""Render the published LER / average seconds-per-shot figure without decoding."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[2]
data=json.loads((ROOT/'benchmarks/results/2026-10/summary.json').read_text())
colors={'original':'#64748b','optimized':'#2563eb','gpu':'#e87522'}
labels={'original':'Original CPU','optimized':'Optimized CPU','gpu':'GPU'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(10.8,6.1))
fig.patch.set_facecolor('#ffffff');ax.set_facecolor('#ffffff')
for family,marker in [('bb','o'),('color','^')]:
    rows=data[family]['modes']
    xs=[rows[m]['average_seconds_per_shot'] for m in labels]
    ys=[rows[m]['ler'] for m in labels]
    ax.plot(xs,ys,color='#cbd5e1',lw=1.3,zorder=1)
    for mode in labels:
        r=rows[mode];lo,hi=r['ler_wilson95'];x=r['average_seconds_per_shot'];y=r['ler']
        ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt=marker,color=colors[mode],markersize=10,
                    capsize=5,elinewidth=1.5,label=labels[mode] if family=='bb' else None,zorder=3)
        ax.annotate(f'{x:.4f} s/shot',xy=(x,y),xytext=(0,-30),textcoords='offset points',ha='center',color=colors[mode],fontsize=10)
    text='BB [[288,12,18]]  |  LER 0.040%  |  20 errors' if family=='bb' else 'Superdense Color d=19  |  LER 2.528%  |  1,264 errors'
    ax.text(.025,ys[0]*(2.2 if family=='bb' else 1.55),text,fontsize=12,fontweight='bold',color='#0f172a')
ax.set_yscale('log');ax.set_ylim(.00014,.07);ax.set_xlim(.012,.43)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{100*y:g}%'))
ax.set_xlabel('Average decoding time (s/shot)  —  lower is faster',labelpad=13)
ax.set_ylabel('Logical error rate (LER)',labelpad=10)
ax.grid(axis='y',which='major',color='#e2e8f0',linewidth=.8);ax.grid(axis='x',color='#f1f5f9',linewidth=.7)
ax.set_title('Same observed accuracy, faster decoding',loc='left',fontweight='bold',fontsize=19,pad=26)
ax.legend(loc='upper right',frameon=False,ncol=3,bbox_to_anchor=(1,1.065),fontsize=10)
fig.text(.125,.025,'50,000 shots per configuration · SI1000 p = 0.001 · 1 CPU core · GTX 1660 Ti\nVertical bars: Wilson 95% LER intervals. All versions agree on every shot-level success/failure flag.',fontsize=10,color='#475569')
fig.subplots_adjust(left=.12,right=.98,top=.84,bottom=.19)
assets=ROOT/'docs/assets';assets.mkdir(parents=True,exist_ok=True)
fig.savefig(assets/'ler_vs_time.png',dpi=180,facecolor='white')
fig.savefig(assets/'ler_vs_time.svg',facecolor='white')
plt.close(fig)
print(assets/'ler_vs_time.png')
