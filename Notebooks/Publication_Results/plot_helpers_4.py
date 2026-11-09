"""Small reusable plotting routines; figure selection and captions live in the notebook."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from analysis_helpers_4 import METHODS, METHOD_LABELS, METRIC_LABELS, PERCENTAGES, DISPLAY_CLASSES

COLORS = dict(zip(METHODS, sns.color_palette('colorblind', len(METHODS))))
METHOD_COLORS = {(model, method): sns.color_palette('colorblind', 8)[i]
                 for model, methods in METHODS.items() for i, method in enumerate(methods)}


def empty_panel(ax, text='Pending eligible results'):
    ax.text(.5,.5,text,ha='center',va='center',transform=ax.transAxes,color='#596574',wrap=True)


def intervals(frame, metrics):
    if frame.empty:
        fig,ax=plt.subplots(figsize=(8,3));empty_panel(ax);return fig
    fig,axes=plt.subplots(1,len(metrics),figsize=(15,max(4,.43*len(frame)+1.8)),sharey=True,squeeze=False)
    y=np.arange(len(frame))
    for ax,m in zip(axes[0],metrics):
        for position,(_,row) in enumerate(frame.iterrows()):
            ax.plot([row[m+'_low'],row[m+'_high']],[position,position],color=COLORS[row.model],lw=1.5)
            ax.scatter(row[m],position,color=COLORS[row.model],s=30,zorder=3)
        ax.set_title(METRIC_LABELS[m]);ax.set_xlim(0,1);ax.grid(axis='x',alpha=.25)
        ax.set_xlabel('Estimate and 95% patient bootstrap CI')
    axes[0,0].set_yticks(y,frame.label);axes[0,0].invert_yaxis()
    fig.tight_layout();return fig


def learning_panels(frame, metric, counts):
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
    for ax,model in zip(axes.flat,METHODS):
        subset=frame[frame.model==model]
        ax.set_title(model,weight='bold')
        if subset.empty:empty_panel(ax)
        for (method,family),g in subset.groupby(['method','source_family'],sort=False):
            # Missing percentages are NaN so lines never bridge missing runs.
            g=g.set_index('percentage').reindex(PERCENTAGES)
            x=np.array([counts[p] for p in PERCENTAGES])
            color=METHOD_COLORS[(model,method)]
            multiple=subset[subset.method==method].source_family.nunique()>1
            label=METHOD_LABELS[method]+(f' [{family}]' if multiple else '')
            ax.plot(x,g[metric],marker='o',ms=4,label=label,color=color)
            ax.vlines(x,g[metric+'_low'],g[metric+'_high'],color=color,alpha=.6,lw=.8)
            ax.fill_between(x,g[metric+'_low'].astype(float),g[metric+'_high'].astype(float),color=color,alpha=.12)
        if not subset.empty:ax.legend(fontsize=7,loc='lower right')
        ax.set_ylim(0,1);ax.grid(alpha=.2);ax.set_xticks(list(counts.values()))
        ax.tick_params(axis='x',rotation=35)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,.85,'Development pools are nested.\nPerforation: 40 at every size.\nTest cohort: fixed across sizes.\n\nShading: 95% pointwise CIs.\nGaps: missing eligible results.',va='top',fontsize=11)
    fig.supxlabel('Actual development patients (all five folds combined)')
    fig.supylabel(METRIC_LABELS[metric]);fig.tight_layout();return fig


def difference_plot(frame, label_column='comparison'):
    if frame.empty:
        fig,ax=plt.subplots(figsize=(9,3));empty_panel(ax);return fig
    fig,ax=plt.subplots(figsize=(12,max(4,len(frame)*.4+1.5)))
    for i,(_,r) in enumerate(frame.iterrows()):
        ax.plot([r.ci_low,r.ci_high],[i,i],color='#365c8d',lw=2)
        ax.scatter(r.difference,i,color='#365c8d',s=30)
    ax.axvline(0,color='#666',ls='--',lw=1)
    ax.set_yticks(range(len(frame)),frame[label_column]);ax.invert_yaxis()
    ax.set_xlabel('Macro F1 difference (A − B), paired 95% bootstrap CI')
    ax.grid(axis='x',alpha=.2);fig.tight_layout();return fig


def confusion_panels(frame, matrices, normalize=False):
    n=max(1,len(frame));cols=3;rows=int(np.ceil(n/cols))
    fig,axes=plt.subplots(rows,cols,figsize=(14,rows*4.1),squeeze=False)
    for ax,(_,r) in zip(axes.flat,frame.iterrows()):
        cm=matrices[r.run_id]
        values=cm/cm.sum(1,keepdims=True) if normalize else cm
        sns.heatmap(values,annot=True,fmt='.2f' if normalize else 'd',cmap='Blues',
                    vmin=0,vmax=1 if normalize else None,cbar=False,ax=ax,
                    xticklabels=DISPLAY_CLASSES,yticklabels=DISPLAY_CLASSES)
        ax.set_title(r.label,fontsize=9);ax.set_xlabel('Predicted');ax.set_ylabel('Reference')
        ax.tick_params(axis='x',rotation=40,labelsize=8);ax.tick_params(axis='y',rotation=0,labelsize=8)
    for ax in list(axes.flat)[len(frame):]:ax.axis('off')
    fig.tight_layout();return fig


PERCENTAGES=[20,40,60,80,100]
SIZES=[248,456,664,872,1080]
PARTIAL=[
    ('3DINO','LoRA','LoRA'),
    ('COLIPRI','PrimusMFinalBlockFineTune','Final-block fine-tuning'),
    ('FlexiCT','FinalBlockFineTune','Final-block fine-tuning'),
    ('Merlin','ConvLoRA_LLRD','ConvLoRA + LLRD'),
    ('Pillar0','AtlasFinalBlockFineTune','Final-block fine-tuning'),
]
FROZEN={'3DINO':('LinearProbe','Linear probe'),
        'COLIPRI':('PoolingOptimisedLinearProbe','Frozen + native pooling'),
        'FlexiCT':('LinearProbe','Linear probe'),
        'Merlin':('LinearProbe','Linear probe'),
        'Pillar0':('LinearProbe','Linear probe')}
FULL=[('Merlin','FullFineTune','Merlin — full fine-tuning','#8d4f98','^'),
      ('DenseNet','FromScratch','DenseNet — from scratch','#3279a8','o'),
      ('MedicalNet','FullFineTune','MedicalNet — full fine-tuning','#18877c','s')]
CAPTIONS={
 'Figure_3a':
 'Figure 3a. Partial adaptation and frozen-encoder performance across development-set sizes. '
 'Macro-F1 is shown for 3DINO, COLIPRI, FlexiCT, Merlin, and Pillar-0 using five-fold CV ensembles '
 'at actual development-set sizes of 248, 456, 664, 872, and 1080 patients. Partial adaptation comprises '
 '3DINO LoRA, final-block fine-tuning for COLIPRI, FlexiCT, and Pillar-0, and corrected Merlin ConvLoRA '
 'with layer-wise learning-rate decay (LLRD). Frozen-encoder comparators use linear probing, except '
 'COLIPRI, which uses its native pooling and classifier procedure. The completed 3DINO linear-probe '
 'results are included. Points are observed ensemble estimates, and lines connect them for readability; '
 'no monotonic model or extrapolation was fitted. The same 278-patient test cohort was used throughout. '
 'Perforation support in development remained 40 patients at every size, so both total sample size '
 'and class composition changed. Curves show point estimates without confidence bands. '
 'Full-network Merlin fine-tuning is reported separately in Figure 3b.',
 'Figure_3b':
 'Figure 3b. Full-network fine-tuning and training from scratch across development-set sizes. '
 'Macro-F1 is compared for pretrained Merlin and MedicalNet with full-network fine-tuning and '
 'DenseNet trained from scratch, using five-fold CV ensembles at actual development-set sizes of '
 '248, 456, 664, 872, and 1080 patients. The Merlin curve is the retrospective publication-derived '
 'full-network addition (FullFineTune CV_2). Points are observed ensemble estimates, and lines connect '
 'them for readability; no monotonic model or extrapolation was fitted. The same 278-patient test '
 'cohort was used throughout. Perforation support in development remained 40 patients at every size, '
 'so both total sample size and class composition changed. Curves show point estimates without '
 'confidence bands. These are comparisons of complete model-specific training recipes, not an '
 'isolated test of architecture or pretraining.',
 'Figure_3_combined':
 'Figure 3. Ensemble macro-F1 across development-set sizes. (a) Partial adaptation versus '
 'frozen-encoder comparators for 3DINO, COLIPRI, FlexiCT, Merlin, and Pillar-0; the 3DINO linear probe '
 'is included, and COLIPRI uses its native frozen-encoder pooling and classifier procedure. '
 '(b) Full-network fine-tuning of Merlin and MedicalNet versus DenseNet trained from scratch, '
 'including the retrospective publication-derived full-network Merlin addition. The actual '
 'development-set sizes were 248, 456, 664, 872, and 1080 patients; the same 278-patient test cohort '
 'was used throughout. Perforation support in development remained 40 patients, so both total '
 'sample size and class composition changed. Points are observed five-fold ensemble estimates, '
 'and lines connect them for readability; no monotonic model or extrapolation was fitted. '
 'Observed point estimates are displayed without confidence bands.'
}


def style_axis(ax):
    ax.set_xlim(205,1123);ax.set_ylim(0,100)
    ax.set_xticks(SIZES);ax.set_yticks(np.arange(0,101,20))
    ax.grid(True,color='#e5e8eb',linewidth=.7,zorder=0)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#39434b');ax.spines['bottom'].set_color('#39434b')
    ax.tick_params(length=3,width=.8,labelsize=9)
    for text in ax.get_xticklabels():text.set_rotation(35);text.set_ha('right')

def values(data,model,method):
    rows=data[(data.model==model)&(data.method==method)].sort_values('development_n')
    assert rows.development_n.tolist()==SIZES
    return rows.macro_f1_percent.to_numpy()

def draw_a(data):
    fig=plt.figure(figsize=(11.2,7.4),facecolor='white')
    gs=fig.add_gridspec(2,6,left=.075,right=.985,bottom=.125,top=.875,hspace=.53,wspace=.70)
    slots=[gs[0,0:2],gs[0,2:4],gs[0,4:6],gs[1,1:3],gs[1,3:5]]
    for (model,method,label),slot in zip(PARTIAL,slots):
        ax=fig.add_subplot(slot);style_axis(ax)
        ax.set_title('Pillar-0' if model=='Pillar0' else model,fontsize=13,fontweight='bold',pad=9)
        ax.plot(SIZES,values(data,model,method),color='#397fb0',marker='o',markersize=4.5,lw=1.8,label=label,zorder=3)
        frozen,frozen_label=FROZEN[model]
        ax.plot(SIZES,values(data,model,frozen),color='#d78b28',marker='s',markersize=4.3,lw=1.8,label=frozen_label,zorder=3)
        ax.legend(loc='lower right',fontsize=8,frameon=False,handlelength=2,handletextpad=.6)
    fig.text(.075,.965,'a',fontsize=19,fontweight='bold',va='top')
    fig.text(.111,.962,'Partial adaptation and frozen-encoder comparators',fontsize=14,va='top',fontweight='medium')
    fig.supylabel('Macro-F1 (%)',x=.016,fontsize=12)
    fig.supxlabel('Actual development patients across all five folds',y=.035,fontsize=12)
    return fig

def draw_b(data):
    fig,ax=plt.subplots(figsize=(8.2,5.5),facecolor='white')
    fig.subplots_adjust(left=.12,right=.975,bottom=.18,top=.84)
    style_axis(ax)
    for model,method,label,color,marker in FULL:
        ax.plot(SIZES,values(data,model,method),color=color,marker=marker,markersize=6,lw=2,label=label,zorder=3)
    ax.legend(loc='lower right',fontsize=10,frameon=False,handlelength=2.5,labelspacing=.7)
    ax.set_xlabel('Actual development patients across all five folds',fontsize=12,labelpad=11)
    ax.set_ylabel('Macro-F1 (%)',fontsize=12,labelpad=10)
    fig.text(.12,.965,'b',fontsize=19,fontweight='bold',va='top')
    fig.text(.167,.960,'Full-network fine-tuning and training from scratch',fontsize=13,va='top')
    return fig

