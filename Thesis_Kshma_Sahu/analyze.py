"""Statistics and graphs for the kinesiophobia–SCT thesis from the master chart."""
import json, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
df = pd.read_excel(src, sheet_name='Masterchart', header=2, keep_default_na=False, na_values=[''])
df.columns = ['sno', 'id', 'age', 'gender', 'bmi', 'side', 'weeks', 'tsk', 'sct', 'handrail', 'gait', 'device']
df = df[df['id'].astype(str).str.startswith('TKR')].copy()
for c in ['age', 'bmi', 'weeks', 'tsk', 'sct']:
    df[c] = df[c].astype(float)
df['tskgrp'] = np.where(df.tsk > 37, 'High', 'Low')
n = len(df)
R = {'n': n}

def desc(s):
    q1, q3 = s.quantile([.25, .75])
    return dict(mean=s.mean(), sd=s.std(ddof=1), min=s.min(), max=s.max(), median=s.median(), q1=q1, q3=q3,
                se=s.std(ddof=1) / np.sqrt(len(s)), skew=stats.skew(s, bias=False), kurt=stats.kurtosis(s, bias=False))
R['desc'] = {c: desc(df[c]) for c in ['age', 'bmi', 'weeks', 'tsk', 'sct']}
R['freq'] = {c: df[c].value_counts().to_dict() for c in ['gender', 'side', 'handrail', 'gait', 'device', 'tskgrp']}
R['age_bands'] = pd.cut(df.age, [49, 59, 69, 75], labels=['50–59', '60–69', '70–75']).value_counts().sort_index().to_dict()
R['bmi_bands'] = pd.cut(df.bmi, [0, 18.49, 24.99, 29.99, 100], labels=['Underweight (<18.5)', 'Normal (18.5–24.9)', 'Overweight (25–29.9)', 'Obese (≥30)']).value_counts().sort_index().to_dict()
R['week_bands'] = pd.cut(df.weeks, [3, 8, 12, 24], labels=['4–8 weeks', '9–12 weeks', '13–24 weeks']).value_counts().sort_index().to_dict()

R['shapiro'] = {c: dict(zip(['W', 'p'], stats.shapiro(df[c]))) for c in ['age', 'bmi', 'weeks', 'tsk', 'sct']}
pr = stats.pearsonr(df.tsk, df.sct)
ci = pr.confidence_interval(0.95)
sr = stats.spearmanr(df.tsk, df.sct)
lr = stats.linregress(df.tsk, df.sct)
R['pearson'] = dict(r=pr.statistic, p=pr.pvalue, lo=ci.low, hi=ci.high, r2=pr.statistic ** 2)
R['spearman'] = dict(rho=sr.statistic, p=sr.pvalue)
R['reg'] = dict(slope=lr.slope, intercept=lr.intercept, se=lr.stderr, p=lr.pvalue)
R['other_corr'] = {}
for c in ['age', 'bmi', 'weeks']:
    a, b = stats.pearsonr(df[c], df.tsk), stats.pearsonr(df[c], df.sct)
    R['other_corr'][c] = dict(r_tsk=a.statistic, p_tsk=a.pvalue, r_sct=b.statistic, p_sct=b.pvalue)

def two_group(col, g1, g2):
    out = {}
    for v in ['tsk', 'sct']:
        a, b = df[df[col] == g1][v], df[df[col] == g2][v]
        t = stats.ttest_ind(a, b)
        lev = stats.levene(a, b)
        sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
        u = stats.mannwhitneyu(a, b)
        out[v] = dict(n1=len(a), m1=a.mean(), s1=a.std(), n2=len(b), m2=b.mean(), s2=b.std(), t=t.statistic, p=t.pvalue,
                      df=len(a) + len(b) - 2, lev_F=lev.statistic, lev_p=lev.pvalue, d=(a.mean() - b.mean()) / sp, U=u.statistic, pu=u.pvalue)
    return out
R['by_tskgrp'] = two_group('tskgrp', 'Low', 'High')
R['by_gender'] = two_group('gender', 'Male', 'Female')
R['by_side'] = two_group('side', 'Right', 'Left')
R['by_gait'] = two_group('gait', 'Reciprocal', 'Step-by-step')

def anova(col):
    out = {}
    groups = [g for g in df[col].unique()]
    for v in ['tsk', 'sct']:
        arrs = [df[df[col] == g][v] for g in groups]
        f = stats.f_oneway(*arrs)
        out[v] = dict(F=f.statistic, p=f.pvalue, groups={g: dict(n=len(a), m=a.mean(), s=a.std()) for g, a in zip(groups, arrs)})
    return out
R['by_handrail'] = anova('handrail')
R['by_device'] = anova('device')
# Cross-tab kinesiophobia group vs gait pattern / handrail
for col in ['gait', 'handrail']:
    ct = pd.crosstab(df.tskgrp, df[col])
    chi = stats.chi2_contingency(ct)
    R[f'ct_{col}'] = dict(table=ct.to_dict(), chi2=chi.statistic, p=chi.pvalue, dof=int(chi.dof))
json.dump(R, open(f'{out}/stats.json', 'w'), indent=1, default=float)

# ---- graphs ----
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 11, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False})

def save(fig, name):
    fig.tight_layout(); fig.savefig(f'{out}/{name}.png', dpi=200, facecolor='white'); plt.close(fig)

def bars(name, labels, values, ylabel, colors=None, pct=True, title=None, err=None):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    colors = colors or [BLUE] * len(values)
    b = ax.bar(labels, values, color=colors, width=0.55, edgecolor='white', linewidth=2, yerr=err, capsize=6 if err else 0,
               error_kw=dict(ecolor=INK2, lw=1.2))
    top = max(v + (e or 0) for v, e in zip(values, err or [0] * len(values)))
    for rect, v, e in zip(b, values, err or [0] * len(values)):
        lab = f'{v:.0f} ({v / n * 100:.0f}%)' if pct else f'{v:.2f}'
        ax.text(rect.get_x() + rect.get_width() / 2, v + (e or 0) + top * 0.02, lab, ha='center', va='bottom', color=INK, fontsize=10)
    ax.set_ylabel(ylabel); ax.set_ylim(0, top * 1.18)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    if title: ax.set_title(title, color=INK, fontsize=12)
    save(fig, name)

f = R['freq']
bars('g_gender', ['Male', 'Female'], [f['gender'].get('Male', 0), f['gender'].get('Female', 0)], 'Number of participants', [BLUE, ORANGE])
bars('g_side', ['Right knee', 'Left knee'], [f['side'].get('Right', 0), f['side'].get('Left', 0)], 'Number of participants', [BLUE, ORANGE])
bars('g_age', list(R['age_bands'].keys()), list(R['age_bands'].values()), 'Number of participants')
bars('g_bmi', [k.split(' (')[0] for k in R['bmi_bands']], list(R['bmi_bands'].values()), 'Number of participants')
bars('g_tskgrp', ['Low (≤ 37)', 'High (> 37)'], [f['tskgrp'].get('Low', 0), f['tskgrp'].get('High', 0)], 'Number of participants', [BLUE, ORANGE])
bars('g_gait', ['Reciprocal', 'Step-by-step'], [f['gait'].get('Reciprocal', 0), f['gait'].get('Step-by-step', 0)], 'Number of participants', [BLUE, ORANGE])
bars('g_handrail', ['None', 'Unilateral', 'Bilateral'], [f['handrail'].get(k, 0) for k in ['None', 'Unilateral', 'Bilateral']], 'Number of participants', [BLUE, ORANGE, AQUA])
dk = ['None', 'Single Cane', 'Elbow Crutch', 'Walker']
bars('g_device', dk, [f['device'].get(k, 0) for k in dk], 'Number of participants', [BLUE, ORANGE, AQUA, YELLOW])
g = R['by_tskgrp']['sct']
bars('g_sct_by_grp', ['Low kinesiophobia', 'High kinesiophobia'], [g['m1'], g['m2']], 'Mean SCT time (seconds)', [BLUE, ORANGE], pct=False, err=[g['s1'], g['s2']])
d = R['desc']
bars('g_means', ['TSK-17 score', 'SCT time (s)'], [d['tsk']['mean'], d['sct']['mean']], 'Mean value', [BLUE, ORANGE], pct=False, err=[d['tsk']['sd'], d['sct']['sd']])

# scatter with regression line
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.scatter(df.tsk, df.sct, s=46, color=BLUE, edgecolor='white', linewidth=1.5, zorder=3)
xs = np.linspace(df.tsk.min() - 1, df.tsk.max() + 1, 50)
ax.plot(xs, lr.intercept + lr.slope * xs, color=ORANGE, lw=2, zorder=2)
ax.axvline(37.5, color=INK2, lw=1, ls='--')
ax.text(37.8, df.sct.min(), 'TSK cut-off (> 37)', color=INK2, fontsize=9)
ax.set_xlabel('TSK-17 score'); ax.set_ylabel('Stair Climb Test time (seconds)')
ax.text(0.03, 0.95, f"r = {pr.statistic:.3f}, p < 0.001\nR² = {pr.statistic**2:.3f}\ny = {lr.intercept:.2f} + {lr.slope:.3f}x",
        transform=ax.transAxes, va='top', color=INK, fontsize=10, bbox=dict(fc='white', ec=GRID))
ax.grid(True, color=GRID); ax.set_axisbelow(True)
save(fig, 'g_scatter')

# histograms with normal curve
for v, lab in [('tsk', 'TSK-17 score'), ('sct', 'SCT time (seconds)')]:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(df[v], bins=8, color=BLUE, edgecolor='white', linewidth=2)
    xs = np.linspace(df[v].min(), df[v].max(), 100)
    bw = (df[v].max() - df[v].min()) / 8
    ax.plot(xs, stats.norm.pdf(xs, df[v].mean(), df[v].std()) * n * bw, color=ORANGE, lw=2)
    ax.set_xlabel(lab); ax.set_ylabel('Frequency'); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    save(fig, f'g_hist_{v}')
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    (osm, osr), (sl, ic, _) = stats.probplot(df[v], dist='norm')
    ax.scatter(osm, osr, s=26, color=BLUE, edgecolor='white', linewidth=1)
    ax.plot(osm, sl * osm + ic, color=ORANGE, lw=2)
    ax.set_xlabel('Theoretical quantiles'); ax.set_ylabel(f'Observed {lab}'); ax.grid(True, color=GRID); ax.set_axisbelow(True)
    save(fig, f'g_qq_{v}')
print(json.dumps({k: R[k] for k in ['desc', 'freq', 'shapiro', 'pearson', 'spearman', 'reg', 'other_corr', 'age_bands', 'bmi_bands', 'week_bands']}, indent=1, default=lambda x: round(float(x), 4)))
print(json.dumps({k: R[k] for k in ['by_tskgrp', 'by_gender', 'by_side', 'by_gait', 'by_handrail', 'by_device', 'ct_gait', 'ct_handrail']}, default=lambda x: round(float(x), 4)))
