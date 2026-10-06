import pandas as pd, numpy as np, sys, os
V='/private/tmp/claude-501/-Users-yimingzhang-Github-Sweden-dikes/3865b26b-616b-4c68-b473-673d942898fa/scratchpad/verify'
C='/Users/yimingzhang/Github/Grenville_seminar/data/pmag_compilation'
def num(x):
    try: return float(str(x).replace('+','').strip())
    except: return np.nan
def cmp(study, cfile=None):
    c=pd.read_csv(f'{C}/{cfile or study}.csv',encoding='utf-8-sig')
    c=c.rename(columns={'k':'dir_k','a95':'dir_alpha95','N':'dir_n_samples'})
    t=pd.read_csv(f'{V}/{study}/sites_transcribed.csv',dtype=str)
    for col in ['dec','inc','k','a95','n']: t[col+'_f']=t[col].map(num)
    used=set(); out=[]
    for i,r in c.iterrows():
        cand=t[(np.isclose(t.dec_f,r.dir_dec))&(np.isclose(t.inc_f,r.dir_inc))&(~t.index.isin(used))]
        if len(cand)==0:
            # nearest
            d=np.hypot(t.dec_f-r.dir_dec,t.inc_f-r.dir_inc); j=d.idxmin()
            out.append(f'row {i} site {r.site}: NO exact dec/inc match ({r.dir_dec},{r.dir_inc}); nearest transcribed site {t.site[j]} ({t.dec[j]},{t.inc[j]})'); used.add(j); continue
        j=cand.index[0]; used.add(j)
        for cc,tc in [('dir_k','k_f'),('dir_alpha95','a95_f'),('dir_n_samples','n_f')]:
            if cc in c.columns:
                a,b=r[cc],t.at[j,tc]
                if not ((pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and np.isclose(a,b))):
                    out.append(f'row {i} site {r.site} ({r.get("dir_comp_name","")}): {cc} compiled {a} vs transcribed {t.at[j,tc.replace("_f","")]} [{t.site[j]}]')
        if str(r.site)!=str(t.site[j]): out.append(f'row {i}: site name compiled {r.site} vs transcribed {t.site[j]}')
        if 'dir_comp_name' in c.columns and str(r.dir_comp_name)!=str(t.comp_compiled[j]) and not (pd.isna(r.dir_comp_name) and pd.isna(t.comp_compiled[j])):
            out.append(f'row {i} site {r.site}: label compiled {r.dir_comp_name} vs transcription comp_compiled {t.comp_compiled[j]}')
    extra=[f'{t.site[j]} {t.comp_compiled[j]} {t.dec[j]}/{t.inc[j]}' for j in t.index if j not in used]
    print(f'===== {study}: compiled {len(c)}, transcribed {len(t)}')
    for o in out: print('  ',o)
    if extra: print('   transcribed rows not in compiled:', '; '.join(extra))
if __name__=='__main__':
    for s in sys.argv[1:]: cmp(s)
