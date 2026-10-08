"""Native-node diagnostic sections; no interpolation of posterior samples."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import TwoSlopeNorm


def source_centroid(points, weights, flux, threshold):
    active = weights * np.where(flux >= threshold, flux, 0.)
    if active.sum() <= 0:
        raise ValueError('No active source for displacement')
    return np.sum(points * active[:, None], axis=0) / active.sum()


def section_nodes(points, center, direction, half_length=.002):
    """Nearest native nodes to a geometric line; retain offsets for interpretation."""
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm <= 0:
        raise ValueError('Undefined section direction')
    direction = direction / norm
    s = np.linspace(-half_length, half_length, 401)
    targets = center + s[:, None] * direction
    _, ids = cKDTree(points).query(targets)
    ids = np.unique(ids)
    along = (points[ids] - center) @ direction
    offset = (points[ids] - center) @ np.array([-direction[1], direction[0]])
    order = np.argsort(along, kind='stable')
    return ids[order], along[order], offset[order]


def write_diagnostics(folder, name, data, direction, *, error_limit, labels=None):
    """Truth-selected peak sections are visualization only, never inference inputs."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    labels = labels or [('native_B', 'B: physics mean + RBF'),
                        ('native_C', 'C: physics mean + sequential heat')]
    points, truth = data['points'], data['truth']
    tri = Triangulation(points[:, 0]*1000, points[:, 1]*1000, data['triangles'])
    center = points[np.argmax(truth)]
    direction = direction / np.linalg.norm(direction)
    sections = [('Along flux-centroid displacement', direction),
                ('Across flux-centroid displacement', np.array([-direction[1], direction[0]]))]
    fields = [('Physics forecast', data['native_physics_mean'])] + [(label, data[key+'_mean']) for key,label in labels]
    fig, axes = plt.subplots(1, len(fields), figsize=(5*len(fields), 4.6), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0, vmax=error_limit)
    for ax,(label,field) in zip(np.atleast_1d(axes), fields):
        im=ax.tripcolor(tri, field-truth, shading='gouraud', cmap='RdBu_r', norm=norm)
        if np.any(data['camera']==data['ceiling']) and not np.all(data['camera']==data['ceiling']):
            ax.tricontour(tri, (data['camera']==data['ceiling']).astype(float), levels=[.5], colors='black', linewidths=.7)
        for _,vec in sections:
            line = center + np.array([-.002,.002])[:,None]*vec
            ax.plot(line[:,0]*1000,line[:,1]*1000,color='black',lw=.7,ls='--')
        ax.set(title=label,xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), label='Signed error: estimate - truth (K)', shrink=.85)
    fig.suptitle(name+': red = overprediction; blue = underprediction\nBlack contour: observed saturation; dashed lines: section locations')
    fig.savefig(folder/f'signed_errors_{name}.png',dpi=180);plt.close(fig)
    fig, axes=plt.subplots(2,len(labels),figsize=(6*len(labels),8),sharey=True,constrained_layout=True,squeeze=False)
    rows=[]
    for j,(section,vec) in enumerate(sections):
        ids,along,offset=section_nodes(points,center,vec)
        for i,(key,label) in enumerate(labels):
            ax=axes[j,i]; x=along*1000
            ax.fill_between(x,data[key+'_lower'][ids],data[key+'_upper'][ids],color='#0072B2',alpha=.22,label='Pointwise 95% interval')
            ax.plot(x,truth[ids],color='black',lw=1.7,label='Native truth')
            ax.plot(x,data['native_physics_mean'][ids],color='#D55E00',ls='--',label='Physics forecast')
            ax.plot(x,data[key+'_mean'][ids],color='#0072B2',lw=1.3,marker='.',ms=2,label='Posterior mean')
            ax.axhline(float(data['ceiling']),color='#777777',ls=':',label='Camera ceiling')
            ax.set(title=label+'\n'+section,xlabel='Distance from true peak along section (mm)',ylabel='Temperature (K)')
            ax.grid(alpha=.15);ax.legend(fontsize=8,loc='upper left')
            for node,a,o in zip(ids,along,offset):
                rows.append(dict(trajectory=name,section=section,model=key,node=int(node),
                    along_mm=a*1000,normal_offset_mm=o*1000,x_mm=points[node,0]*1000,y_mm=points[node,1]*1000,
                    truth_K=truth[node],physics_K=data['native_physics_mean'][node],
                    mean_K=data[key+'_mean'][node],lower_K=data[key+'_lower'][node],upper_K=data[key+'_upper'][node],
                    ceiling_K=float(data['ceiling'])))
    fig.suptitle(name+': same native nodes for B/C; sections through true peak (evaluation only)\nNodewise marginal intervals; connecting lines/bands are visual guides, not off-mesh credible intervals',fontsize=11)
    fig.savefig(folder/f'cross_sections_{name}.png',dpi=180);plt.close(fig)
    pd.DataFrame(rows).to_csv(folder/f'cross_sections_{name}.csv',index=False)
    summary=[]; weights=data['area_weights']
    for label,field in fields:
        error=field-truth
        summary.append(dict(trajectory=name,field=label,bias_K=float(np.sum(weights*error)/weights.sum()),
                            positive_error_area_fraction=float(np.sum(weights*(error>0))/weights.sum()),
                            negative_error_area_fraction=float(np.sum(weights*(error<0))/weights.sum()),
                            error_min_K=float(error.min()),error_max_K=float(error.max())))
    return summary


def advection_comparison(folder):
    folder=Path(folder)
    df=pd.read_csv(folder/'results.csv')
    models=['B','B_mean_advection','C','C_mean_advection','C_uncertainty_advection','C_both_advection']
    labels=['B','B + mean adv.','C','C + mean adv.','C + uncertainty adv.','C + both adv.']
    colors=['#757575','#009E73','#333333','#0072B2','#CC79A7','#D55E00']
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    metrics=[('whole_area','rmse_K','Whole-field RMSE (K)'),('hot1pct_area','rmse_K','Hot-area RMSE (K)'),
             ('hot1pct_area','crps_K','Hot-area CRPS (K)'),('hot1pct_area','bias_K','Hot-area signed bias (K)'),
             ('hot1pct_area','coverage95','Hot-area 95% coverage'),('hot1pct_area','width95_K','Hot-area 95% width (K)')]
    names=list(df.trajectory.unique());x=np.arange(len(names));width=.12
    for ax,(region,metric,title) in zip(axes.flat,metrics):
        for i,(model,label,color) in enumerate(zip(models,labels,colors)):
            values=[float(df[(df.trajectory==name)&(df.region==region)&(df.model==model)][metric].iloc[0]) for name in names]
            ax.bar(x+(i-2.5)*width,values,width=width,label=label,color=color)
        ax.set_xticks(x,[n.replace('ScanPath_',' ') for n in names]);ax.set_title(title)
        ax.axhline(0,color='black',lw=.6);ax.grid(axis='y',alpha=.15)
        if metric=='coverage95':ax.axhline(.95,color='black',ls='--',lw=.8);ax.set_ylim(0,1)
    handles,leglabels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,leglabels,loc='outside lower center',ncol=3)
    fig.suptitle('Native-input advection pilot: two development paths, frozen camera and parameters\nB: RBF residual; C: Q + propagated previous uncertainty; no joint previous/current likelihood')
    fig.savefig(folder/'comparison.png',dpi=170);plt.close(fig)
