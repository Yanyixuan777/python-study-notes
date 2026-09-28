"""Reproducible IBM Telco EDA, preprocessing and hypothesis tests."""
from pathlib import Path
import hashlib
import json
import platform
import urllib.request
import importlib.metadata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
SEED = 42


def wilson(successes, n):
    z = stats.norm.ppf(.975)
    p = successes / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half, center+half


def run():
    for sub in ['data', 'figures', 'results']:
        (ROOT/sub).mkdir(exist_ok=True)
    source_path = ROOT/'data/Telco-Customer-Churn.csv'
    if not source_path.exists():
        urllib.request.urlretrieve(SOURCE, source_path)
    raw = pd.read_csv(source_path)
    df = raw.copy()
    for c in df.select_dtypes(include=['object', 'str']).columns:
        df[c] = df[c].str.strip()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing = df['TotalCharges'].isna()
    assert df['customerID'].is_unique
    assert df['Churn'].isin(['Yes', 'No']).all()
    assert df.loc[missing, 'tenure'].eq(0).all()
    df['ChurnFlag'] = df['Churn'].map({'No': 0, 'Yes': 1})
    # Separate exploratory discovery from a reserved confirmation sample.
    discovery_idx, confirm_idx = train_test_split(df.index, test_size=.30, random_state=SEED)
    discovery = df.loc[discovery_idx].copy()
    confirm = df.loc[confirm_idx].copy()
    numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
    summary = {
        'source': SOURCE, 'sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'rows': len(df), 'columns': len(raw.columns), 'duplicates': int(raw.duplicated().sum()),
        'missing_total_charges': int(missing.sum()), 'missing_fraction': float(missing.mean()),
        'churn_count': int(df.ChurnFlag.sum()), 'churn_rate': float(df.ChurnFlag.mean()),
        'discovery_n': len(discovery), 'confirmation_n': len(confirm),
        'discovery_churn_rate': float(discovery.ChurnFlag.mean()),
        'confirmation_churn_rate': float(confirm.ChurnFlag.mean()),
        'seed': SEED,
    }
    raw.dtypes.astype(str).rename('dtype').to_csv(ROOT/'results/schema.csv')
    df[numeric].describe().T.to_csv(ROOT/'results/descriptive_statistics.csv')
    pd.DataFrame({'raw_missing': raw.isna().sum(), 'after_parsing': df[raw.columns].isna().sum()}).to_csv(ROOT/'results/missingness.csv')
    sns.set_theme(style='whitegrid', font_scale=1.0)
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 180, 'axes.spines.top': False, 'axes.spines.right': False})
    palette = ['#168575', '#cb5266']
    def save(name):
        plt.tight_layout()
        plt.savefig(ROOT/'figures'/name, bbox_inches='tight')
        plt.close()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, c in zip(axs, numeric):
        sns.histplot(data=discovery, x=c, hue='Churn', bins=25, multiple='layer', alpha=.55, palette=palette, ax=ax)
        ax.set_title(c)
    fig.suptitle('Discovery sample: distributions by churn status', y=1.04)
    save('01_distributions.png')
    tables = {}
    for col in ['Contract', 'PaymentMethod', 'InternetService']:
        t = discovery.groupby(col, observed=True).ChurnFlag.agg(['sum', 'count', 'mean'])
        bounds = [wilson(s, n) for s, n in zip(t['sum'], t['count'])]
        t['lower'] = [a for a, b in bounds]
        t['upper'] = [b for a, b in bounds]
        t.to_csv(ROOT/f'results/{col}_discovery.csv')
        tables[col] = t.reset_index().to_dict(orient='records')
    summary['segments'] = tables
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, col in zip(axs, ['Contract', 'PaymentMethod']):
        t = pd.DataFrame(tables[col])
        yy = np.arange(len(t))
        ax.barh(yy, t['mean']*100, color='#168575')
        ax.errorbar(t['mean']*100, yy, xerr=np.vstack([t['mean']-t['lower'], t['upper']-t['mean']])*100, fmt='none', color='#303030', capsize=3)
        ax.set_yticks(yy, t[col].str.replace(' (automatic)', '\n(automatic)', regex=False))
        ax.set_xlim(0, 65)
        ax.set_xlabel('Churn rate (%) with 95% Wilson interval')
        ax.set_title(col)
        for i, row in t.iterrows():
            ax.text(row['upper']*100+1, i, f"{row['mean']:.1%}\nn={row['count']:,}", va='center', fontsize=8)
    save('02_segments.png')
    discovery['TenureBand'] = pd.cut(discovery.tenure, [-1, 12, 24, 48, 72], labels=['0-12', '13-24', '25-48', '49-72'])
    heat = discovery.pivot_table(index='Contract', columns='TenureBand', values='ChurnFlag', aggfunc='mean', observed=False)
    counts = discovery.pivot_table(index='Contract', columns='TenureBand', values='ChurnFlag', aggfunc='size', observed=False).fillna(0)
    annotations = heat.copy().astype(object)
    for i in heat.index:
        for j in heat.columns:
            annotations.loc[i,j] = f'{heat.loc[i,j]:.1%}\nn={counts.loc[i,j]:.0f}' if pd.notna(heat.loc[i,j]) else ''
    plt.figure(figsize=(9,3.5))
    sns.heatmap(heat, annot=annotations, fmt='', cmap='YlGnBu', vmin=0, vmax=.65, cbar_kws={'label':'Churn proportion'})
    plt.title('Contract and tenure together: discovery sample')
    plt.xlabel('Tenure in months')
    save('03_contract_tenure.png')
    summary['contract_tenure'] = [{'contract':i, 'band':j, 'n':int(counts.loc[i,j]), 'rate':float(heat.loc[i,j]) if pd.notna(heat.loc[i,j]) else None} for i in heat.index for j in heat.columns]
    fig, axs = plt.subplots(1, 2, figsize=(10,3.7))
    sns.boxplot(data=discovery, x='Churn', y='MonthlyCharges', ax=axs[0], color='#7ebcac')
    sns.heatmap(discovery[numeric+['ChurnFlag']].corr(), annot=True, fmt='.2f', cmap='vlag', center=0, vmin=-1, vmax=1, ax=axs[1])
    axs[0].set_title('Monthly charges by churn status')
    axs[1].set_title('Numeric correlations')
    save('04_charges_correlations.png')
    summary['correlations'] = discovery[numeric+['ChurnFlag']].corr().round(4).to_dict()
    outliers = {}
    for c in numeric:
        q1, q3 = discovery[c].quantile([.25,.75])
        outliers[c] = int(((discovery[c]<q1-1.5*(q3-q1)) | (discovery[c]>q3+1.5*(q3-q1))).sum())
    summary['iqr_flags_discovery'] = outliers

    # Hypotheses fixed before inspecting the reserved confirmation sample.
    a = confirm.loc[confirm.Contract.eq('Month-to-month'), 'ChurnFlag']
    b = confirm.loc[~confirm.Contract.eq('Month-to-month'), 'ChurnFlag']
    contingency = np.array([[a.sum(), len(a)-a.sum()], [b.sum(), len(b)-b.sum()]], dtype=int)
    chi = stats.chi2_contingency(contingency, correction=False)
    p1, p2 = a.mean(), b.mean()
    rd = p1-p2
    se = np.sqrt(p1*(1-p1)/len(a)+p2*(1-p2)/len(b))
    rr = p1/p2
    se_log_rr = np.sqrt(1/a.sum()-1/len(a)+1/b.sum()-1/len(b))
    fisher = stats.fisher_exact(contingency)
    h1 = {'id':'H1', 'test':'Pearson chi-square (2x2)', 'statistic':float(chi.statistic), 'p_value':float(chi.pvalue),
          'df':int(chi.dof), 'min_expected':float(chi.expected_freq.min()), 'table':contingency.tolist(),
          'month_to_month_n':len(a), 'long_term_n':len(b), 'month_to_month_rate':float(p1), 'long_term_rate':float(p2),
          'risk_difference':float(rd), 'rd_ci95':[float(rd-1.96*se), float(rd+1.96*se)],
          'rd_ci_bonferroni':[float(rd-stats.norm.ppf(1-.05/6)*se), float(rd+stats.norm.ppf(1-.05/6)*se)],
          'risk_ratio':float(rr), 'rr_ci95':np.exp(np.log(rr)+np.array([-1,1])*1.96*se_log_rr).tolist(),
          'fisher_p_value':float(fisher.pvalue)}
    yes = confirm.loc[confirm.ChurnFlag.eq(1), 'MonthlyCharges']
    no = confirm.loc[confirm.ChurnFlag.eq(0), 'MonthlyCharges']
    welch = stats.ttest_ind(yes, no, equal_var=False)
    delta = yes.mean()-no.mean()
    sed = np.sqrt(yes.var()/len(yes)+no.var()/len(no))
    h2 = {'id':'H2','test':'Welch t-test (two-sided)', 'statistic':float(welch.statistic), 'df':float(welch.df),
          'p_value':float(welch.pvalue), 'difference':float(delta),
          'ci95':[float(delta-stats.t.ppf(.975,welch.df)*sed),float(delta+stats.t.ppf(.975,welch.df)*sed)],
          'mean_churn':float(yes.mean()),'mean_retained':float(no.mean()),'n_churn':len(yes),'n_retained':len(no)}
    payment_table = pd.crosstab(confirm.PaymentMethod, confirm.ChurnFlag)
    pc = stats.chi2_contingency(payment_table, correction=False)
    h3 = {'id':'H3','test':'Pearson chi-square (4x2)', 'statistic':float(pc.statistic), 'df':int(pc.dof),
          'p_value':float(pc.pvalue), 'min_expected':float(pc.expected_freq.min()),
          'cramers_v':float(np.sqrt(pc.statistic/payment_table.to_numpy().sum()))}
    tests = [h1,h2,h3]
    for t in tests:
        t['bonferroni_p'] = min(1., 3*t['p_value'])
        t['reject_at_family_alpha_05'] = bool(t['p_value'] < .05/3)
    summary['tests'] = tests
    pd.DataFrame([{k:v for k,v in t.items() if not isinstance(v,(dict,list))} for t in tests]).to_csv(ROOT/'results/hypothesis_tests.csv',index=False)
    pd.DataFrame(contingency,index=['Month-to-month','One/two year'],columns=['Churn','Retained']).to_csv(ROOT/'results/confirmation_contingency.csv')
    payment_table.to_csv(ROOT/'results/payment_confirmation_contingency.csv')

    # Deterministic engineering; fitted imputation/scaling sees discovery only.
    X = df.drop(columns=['customerID','Churn','ChurnFlag']).copy()
    X['TotalChargesMissing'] = missing.astype(int)
    X['SeniorCitizen'] = X.SeniorCitizen.astype(str)
    X['NewCustomer'] = (X.tenure<=12).astype(int)
    X['LogTenure'] = np.log1p(X.tenure)
    X['LogTotalCharges'] = np.log1p(X.TotalCharges)
    X['MonthlyCharges_x_NewCustomer'] = X.MonthlyCharges * X.NewCustomer
    categorical = X.select_dtypes(include=['object','str']).columns.tolist()
    continuous = [c for c in X.columns if c not in categorical and c not in ['NewCustomer','TotalChargesMissing']]
    binary = ['NewCustomer','TotalChargesMissing']
    preprocessor = ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),continuous),
        ('category',OneHotEncoder(handle_unknown='ignore', sparse_output=False,drop=None),categorical),
        ('binary','passthrough',binary)])
    train = preprocessor.fit_transform(X.loc[discovery_idx])
    test = preprocessor.transform(X.loc[confirm_idx])
    names = preprocessor.get_feature_names_out()
    assert np.isfinite(train).all() and np.isfinite(test).all()
    assert set(discovery_idx).isdisjoint(confirm_idx)
    assert len(discovery_idx)+len(confirm_idx)==len(df)
    assert not any('customerID' in n or 'ChurnFlag' in n for n in names)
    pd.DataFrame(train,columns=names,index=discovery_idx).to_csv(ROOT/'data/X_train.csv',index_label='source_row')
    pd.DataFrame(test,columns=names,index=confirm_idx).to_csv(ROOT/'data/X_confirmation.csv',index_label='source_row')
    df.loc[discovery_idx,['ChurnFlag']].to_csv(ROOT/'data/y_train.csv',index_label='source_row')
    df.loc[confirm_idx,['ChurnFlag']].to_csv(ROOT/'data/y_confirmation.csv',index_label='source_row')
    df.drop(columns='customerID').to_csv(ROOT/'data/cleaned_telco.csv',index=False)
    summary['preprocessing'] = {'train_shape':list(train.shape),'confirmation_shape':list(test.shape),'features':names.tolist(),
        'numeric_columns':continuous,'categorical_columns':categorical,
        'imputation_medians':dict(zip(continuous,preprocessor.named_transformers_['numeric'].named_steps['impute'].statistics_.tolist()))}
    sample = pd.DataFrame(train[:5,:6],columns=names[:6]).round(3)
    fig, ax = plt.subplots(figsize=(11,2.5)); ax.axis('off')
    tbl = ax.table(cellText=sample.values,colLabels=[s.replace('numeric__','') for s in sample.columns], loc='center',cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.6)
    ax.set_title(f'Executed preprocessing output: {train.shape[0]:,} rows x {train.shape[1]} features (first five rows, six columns)',pad=20)
    save('05_preprocessing_output.png')
    summary['environment'] = {p:importlib.metadata.version(p) for p in ['pandas','numpy','scipy','scikit-learn','matplotlib','seaborn','reportlab']}
    summary['environment']['python'] = platform.python_version()
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (ROOT/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in summary['environment'].items() if k!='python')+'\nnbformat\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k in ['rows','columns','missing_total_charges','churn_rate','tests','discovery_n','confirmation_n','preprocessing']},indent=2))
    return summary


if __name__ == '__main__':
    run()
