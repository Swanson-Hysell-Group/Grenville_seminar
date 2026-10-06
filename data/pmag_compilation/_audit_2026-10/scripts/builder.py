"""Generic builder: verify/<Study>/{sites,means}_transcribed.csv + META -> MagIC 3 folder."""
import pandas as pd, numpy as np, os, re, math
import pmagpy.pmag as pmag
V='/private/tmp/claude-501/-Users-yimingzhang-Github-Sweden-dikes/3865b26b-616b-4c68-b473-673d942898fa/scratchpad/verify'
QC=[]
def num(x):
    if x is None: return np.nan
    s=str(x).strip().replace('+','').replace('−','-')
    if s in ('','nan','NaN','None','-','—','...'): return np.nan
    try: return float(s)
    except: return np.nan
def angdist(lat1,lon1,lat2,lon2):
    a=np.radians([lat1,lon1,lat2,lon2]); 
    c=np.sin(a[0])*np.sin(a[2])+np.cos(a[0])*np.cos(a[2])*np.cos(a[1]-a[3])
    return np.degrees(np.arccos(np.clip(c,-1,1)))
def fmt(x):
    if isinstance(x,float):
        if np.isnan(x): return ''
        return ('%.4f'%x).rstrip('0').rstrip('.')
    return '' if x is None else str(x)
def vgp_fix(dec,inc,lat,lon,plat,plon,tag):
    """return (vgp_lat, vgp_lon, note) for the VGP of the listed direction, starting from the printed pole."""
    if any(np.isnan(v) for v in [plat,plon]): return np.nan,np.nan,''
    plon=plon%360
    if any(np.isnan(v) for v in [dec,inc,lat,lon]): return plat,plon,''
    v=pmag.dia_vgp(dec,inc,0,lat,lon); clon,clat=v[0],v[1]
    d=angdist(plat,plon,clat,clon); note=''
    if d>90:
        plat,plon=-plat,(plon+180)%360; d=180-d
        note='printed pole is the antipode of the VGP of the listed direction; vgp_lat/vgp_lon converted to the VGP (antipode of printed value)'
    if d>5: QC.append(f'{tag}: printed VGP differs from VGP recomputed with listed dec/inc/lat/lon by {d:.1f} deg')
    return plat,plon,note
def write_magic(folder, table, df, cols):
    os.makedirs(folder,exist_ok=True)
    df=df[[c for c in cols if c in df.columns and df[c].map(lambda x: fmt(x)!='').any()]]
    with open(os.path.join(folder,f'{table}.txt'),'w') as f:
        f.write(f'tab\t{table}\n'); f.write('\t'.join(df.columns)+'\n')
        for _,r in df.iterrows(): f.write('\t'.join(fmt(r[c]).replace('\t',' ').replace('\n',' ') for c in df.columns)+'\n')
SITE_COLS=['site','location','result_type','result_quality','method_codes','citations','geologic_classes','geologic_types','lithologies','lat','lon','age','age_low','age_high','age_unit',
 'dir_tilt_correction','dir_dec','dir_inc','dir_alpha95','dir_k','dir_r','dir_n_samples','dir_n_total_samples','dir_n_specimens','dir_comp_name','vgp_lat','vgp_lon','vgp_dp','vgp_dm','vgp_alpha95','description']
LOC_COLS=['location','location_type','result_name','result_type','method_codes','citations','geologic_classes','lithologies','lat_s','lat_n','lon_w','lon_e','age','age_low','age_high','age_unit',
 'dir_tilt_correction','dir_dec','dir_inc','dir_alpha95','dir_k','dir_r','dir_n_sites','dir_n_samples','dir_n_specimens','pole_comp_name','pole_lat','pole_lon','pole_dp','pole_dm','pole_alpha95','pole_k','pole_n_sites','description']
SAMP_COLS=['sample','site','result_type','result_quality','method_codes','citations','geologic_classes','geologic_types','lithologies','lat','lon','dir_tilt_correction','dir_dec','dir_inc','dir_alpha95','dir_k','dir_r','dir_n_specimens','dir_comp_name','description']
SPEC_COLS=['specimen','sample','result_quality','method_codes','citations','geologic_classes','geologic_types','lithologies','dir_tilt_correction','dir_dec','dir_inc','dir_alpha95','dir_k','dir_mad_free','dir_comp','description']
def build(study, M, outdirs):
    """M: meta dict. Keys: citation(doi), ref (short text), location, lat, lon, coord_note, age...(optional), method_codes, tilt (0/100/None),
       litho(unit)->(lithologies, geologic_classes, geologic_types), site_filter(row)->bool, site_name(row)->str, comp(row)->str, quality(row)->'g'/'b'/'',
       extra_desc(row)->str, means_filter(row)->bool"""
    sfile=f'{V}/{study}/sites_transcribed.csv'; mfile=f'{V}/{study}/'+M.get('means_file','means_transcribed.csv')
    out={}
    if os.path.exists(sfile) and M.get('sites',True):
        t=pd.read_csv(sfile,dtype=str).fillna('')
        rows=[]; samp_rows=[]; spec_rows=[]
        for i,r in t.iterrows():
            if 'site_filter' in M and not M['site_filter'](r): continue
            lat=num(r.site_lat); lon=num(r.site_lon); cnote=''
            if M.get('force_coords') or np.isnan(lat) or np.isnan(lon):
                ll=M['coords'](r) if callable(M.get('coords')) else (M['lat'],M['lon'],M['coord_note'])
                lat,lon,cnote=ll
            else: cnote='site coordinates as published'
            if not np.isnan(lon) and lon>180: lon-=360
            dec,inc=num(r.dec),num(r.inc)
            vl,vo,vnote=vgp_fix(dec,inc,lat,lon,num(r.vgp_lat),num(r.vgp_lon),f'{study} site {r.site}')
            nt=r.n_type.lower(); n=num(r.n)
            nsamp=nspec=np.nan; nnote=''
            if 'specimen treat' in nt or (nt.startswith('specimen')) : nspec=n
            elif 'sample' in nt or 'core' in nt: nsamp=n
            elif not np.isnan(n): nnote=f'N={r.n} ({r.n_type})'
            lith,gcl,gty=M['litho'](r)
            comp=M['comp'](r) if 'comp' in M else r.comp_compiled
            desc=[f"{M['ref']} {r.table_ref}".strip()]
            if r.comp_paper: desc.append(f'paper component/polarity: {r.comp_paper}')
            if r.polarity_note and r.polarity_note not in r.comp_paper: desc.append(r.polarity_note)
            if r.unit: desc.append(f'unit: {r.unit}')
            if nnote: desc.append(nnote)
            if num(r.N_total)==num(r.N_total): desc.append(f'N collected/total: {r.N_total}')
            if vnote: desc.append(vnote+f' (printed {r.vgp_lat}, {r.vgp_lon})')
            if r.notes: desc.append(r.notes)
            if 'extra_desc' in M: 
                e=M['extra_desc'](r)
                if e: desc.append(e)
            desc.append('coordinates: '+cnote)
            if M.get('age_note'): desc.append(M['age_note'])
            a95=num(r.a95)
            lvl=M['level'](r) if 'level' in M else 'site'
            if lvl=='skip': continue
            if 'age_fn' in M:
                ag=M['age_fn'](r)
            else: ag=(M.get('age',np.nan),M.get('age_low',np.nan),M.get('age_high',np.nan))
            if lvl in ('sample','specimen'):
                base=dict(method_codes=M['method_codes'], citations=M['citation'], result_quality=M['quality'](r) if 'quality' in M else '',
                    geologic_classes=gcl, geologic_types=gty, lithologies=lith, dir_tilt_correction=M.get('tilt',np.nan), dir_dec=dec, dir_inc=inc,
                    dir_alpha95=a95, dir_k=num(r.k), dir_r=num(r.r), description='; '.join(d for d in desc if d))
                if lvl=='sample':
                    base.update(sample=M['sample_name'](r), site=M['sample_site'](r), lat=lat, lon=lon, result_type='i', dir_comp_name=comp,
                        dir_n_specimens=(n if ('specimen' in nt) else np.nan))
                    samp_rows.append(base)
                else:
                    base.update(specimen=M['sample_name'](r), sample=M['sample_site'](r), dir_comp=comp, dir_mad_free=M['mad'](r) if 'mad' in M else np.nan)
                    spec_rows.append(base)
                continue
            rows.append(dict(site=M['site_name'](r) if 'site_name' in M else r.site, location=M.get('location_fn',lambda r:M['location'])(r), result_type=M.get('result_type','i'),
                result_quality=M['quality'](r) if 'quality' in M else '', method_codes=M['method_codes'], citations=M['citation'],
                geologic_classes=gcl, geologic_types=gty, lithologies=lith, lat=lat, lon=lon, age=ag[0], age_low=ag[1], age_high=ag[2], age_unit=(M.get('age_unit','Ma') if any(not np.isnan(x) for x in ag) else ''),
                dir_tilt_correction=M.get('tilt',np.nan), dir_dec=dec, dir_inc=inc, dir_alpha95=a95, dir_k=num(r.k), dir_r=num(r.r), dir_n_samples=nsamp,
                dir_n_total_samples=(num(r.N_total) if not np.isnan(nsamp) else np.nan), dir_n_specimens=nspec, dir_comp_name=comp,
                vgp_lat=vl, vgp_lon=vo, vgp_dp=num(r.dp), vgp_dm=num(r.dm), description='; '.join(d for d in desc if d)))
        if rows: out['sites']=pd.DataFrame(rows)
        if samp_rows: out['samples']=pd.DataFrame(samp_rows)
        if spec_rows: out['specimens']=pd.DataFrame(spec_rows)
    if os.path.exists(mfile) and M.get('means',True):
        t=pd.read_csv(mfile,dtype=str).fillna('')
        rows=[]
        for i,r in t.iterrows():
            if 'means_filter' in M and not M['means_filter'](r): continue
            lat=num(r.lat); lon=num(r.lon)
            if np.isnan(lat): lat,lon,cn=(M['lat'],M['lon'],M['coord_note']) if not callable(M.get('coords')) else M['mean_coords'](r)
            else: cn='coordinate as published'
            if not np.isnan(lon) and lon>180: lon-=360
            nt=r.n_type.lower(); n=num(r.N)
            ns=nsa=nsp=np.nan
            if 'site' in nt or 'stud' in nt or 'intrusion' in nt or 'unit' in nt or 'region' in nt: ns=n
            elif 'sample' in nt or 'core' in nt: nsa=n
            elif 'specimen' in nt: nsp=n
            plat,plon=num(r.pole_lat),num(r.pole_lon)
            if not np.isnan(plon): plon%=360
            desc=[f"{M['ref']} {r.table_ref}".strip(), f"result: {r['name']}"]
            if r.comp_paper: desc.append(f'paper component: {r.comp_paper}')
            if r.comp_compiled: desc.append(f'compilation label: {r.comp_compiled}')
            if r.n_type: desc.append(f'N = {r.N} {r.n_type}')
            if r.notes: desc.append(r.notes)
            desc.append('coordinates: '+cn)
            if M.get('age_note'): desc.append(M['age_note'])
            A95=num(r.A95)
            lag=M['mean_age_fn'](r) if 'mean_age_fn' in M else (M.get('age',np.nan),M.get('age_low',np.nan),M.get('age_high',np.nan))
            rows.append(dict(location=M.get('loc_mean_name',lambda r: f"{M['location']}: {r['name']}")(r), location_type=M.get('location_type','Region'), result_name=r['name'], result_type='a',
                method_codes=M['method_codes'], citations=M['citation'], geologic_classes=M.get('loc_classes',''), lithologies=M.get('loc_lith',''),
                lat_s=lat, lat_n=lat, lon_w=lon, lon_e=lon, age=lag[0], age_low=lag[1], age_high=lag[2], age_unit=('Ma' if any(not np.isnan(x) for x in lag) else ''),
                dir_tilt_correction=M.get('tilt',np.nan), dir_dec=num(r.dec), dir_inc=num(r.inc), dir_alpha95=num(r.a95), dir_k=num(r.k),
                dir_n_sites=ns, dir_n_samples=nsa, dir_n_specimens=nsp, pole_comp_name=r.comp_compiled or r.comp_paper, pole_lat=plat, pole_lon=plon,
                pole_dp=num(r.dp), pole_dm=num(r.dm), pole_alpha95=A95, pole_k=num(r.K), pole_n_sites=(ns if not np.isnan(plat) else np.nan), description='; '.join(desc)))
        if rows: out['locations']=pd.DataFrame(rows)
    for od in outdirs:
        if 'sites' in out: write_magic(od,'sites',out['sites'],SITE_COLS)
        if 'locations' in out: write_magic(od,'locations',out['locations'],LOC_COLS)
        if 'samples' in out: write_magic(od,'samples',out['samples'],SAMP_COLS)
        if 'specimens' in out: write_magic(od,'specimens',out['specimens'],SPEC_COLS)
    return out
