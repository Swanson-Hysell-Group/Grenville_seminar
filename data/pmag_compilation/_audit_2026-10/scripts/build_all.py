"""Build MagIC 3 folders for every compiled Grenville pmag study.
Inputs: verify/<Study>/{sites,means}_transcribed.csv (transcribed from the source publications; see MASTER_NOTES.md)
Output: <OUT>/<Folder>/{sites,samples,specimens,locations}.txt
"""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, '/private/tmp/claude-501/-Users-yimingzhang-Github-Sweden-dikes/3865b26b-616b-4c68-b473-673d942898fa/scratchpad/build')
from builder import build, QC, num

OUT = sys.argv[1] if len(sys.argv) > 1 else '/Users/yimingzhang/Github/Grenville_seminar/data/pmag_compilation'
ONLY = sys.argv[2].split(',') if len(sys.argv) > 2 else None

def L(lith, cls='Intrusive:Igneous', typ='Pluton'):
    return lambda r: (lith, cls, typ)

def kw_litho(r, default=('', '', '')):
    u = (str(r.unit) + ' ' + str(r.notes)).lower()
    table = [('impactite', ('Impactite', 'Impact', 'Impact Structure')),
             ('charnockite', ('Charnockite', 'Metamorphic', 'Outcrop')),
             ('anorthositic gabbro', ('Gabbro', 'Intrusive:Igneous', 'Pluton')),
             ('gabbroic anorthosite', ('Anorthosite', 'Intrusive:Igneous', 'Pluton')),
             ('meta-anorthosite', ('Anorthosite', 'Igneous:Metamorphic', 'Pluton')),
             ('metamorphosed anorthosite', ('Anorthosite', 'Igneous:Metamorphic', 'Pluton')),
             ('anorthosite', ('Anorthosite', 'Intrusive:Igneous', 'Pluton')),
             ('microcline gneiss', ('Gneiss', 'Metamorphic', 'Outcrop')),
             ('tonalite gneiss', ('Gneiss', 'Metamorphic', 'Outcrop')),
             ('gneiss', ('Gneiss', 'Metamorphic', 'Outcrop')),
             ('granite', ('Granite', 'Intrusive:Igneous', 'Pluton')),
             ('monzonite', ('Monzonite', 'Intrusive:Igneous', 'Pluton')),
             ('syenite', ('Syenite', 'Intrusive:Igneous', 'Pluton')),
             ('alaskite', ('Alaskite', 'Intrusive:Igneous', 'Dike')),
             ('diabase dike', ('Diabase', 'Intrusive:Igneous', 'Dike')),
             ('dike', ('Diabase', 'Intrusive:Igneous', 'Dike')),
             ('dyke', ('Diabase', 'Intrusive:Igneous', 'Dike')),
             ('pyroxenite', ('Pyroxenite', 'Intrusive:Igneous', 'Intrusion')),
             ('peridotite', ('Peridotite', 'Intrusive:Igneous', 'Sill')),
             ('norite', ('Norite', 'Intrusive:Igneous', 'Pluton')),
             ('diorite', ('Diorite', 'Intrusive:Igneous', 'Pluton')),
             ('gabbro', ('Gabbro', 'Intrusive:Igneous', 'Pluton')),
             ('metavolcanic', ('Metavolcanic', 'Metamorphic', 'Outcrop')),
             ('metasediment', ('Metasediment', 'Metamorphic', 'Outcrop')),
             ('ultramafic', ('Ultramafic', 'Metamorphic', 'Outcrop')),
             ('hemo-ilmenite', ('Fe-Ti oxide ore', 'Intrusive:Igneous', 'Pluton'))]
    for k, v in table:
        if k in u:
            return v
    return default

HALLS = 'Halls (2015, Geology 43:1051-1054, GSA Data Repository 2015352, Table DR2/DR4)'
def C(lat, lon, note):
    return lambda r: (lat, lon, note)

META = {}

META['Alvarez1998a'] = dict(
    ref='Costanzo-Alvarez & Dunlop (1998) EPSL 157:89-103,', citation='10.1016/S0012-821X(98)00028-4',
    location='Central Gneiss Belt, Ontario', lat=45.13, lon=-79.6,
    coord_note=f'45.13, -79.6 = Muskoka (MU) coordinate listed by {HALLS}; Alvarez & Dunlop print no coordinates; regions span ~45-46.5N, so this is only an approximate study-area reference',
    litho=L('Gneiss:Amphibolite', 'Metamorphic', 'Outcrop'), method_codes='LP-DIR-AF:LP-DIR-T', result_type='a',
    age_note='magnetization interpreted ~980-920 Ma by APWP comparison (p. 99)')

META['Brett2008a'] = dict(
    ref='Brett & Dunlop (2008) EPSL 266:125-139,', citation='10.1016/j.epsl.2007.11.005',
    location_fn=lambda r: 'Central Metasedimentary Belt, ' + str(r.unit).split(' terrane')[0] + ' terrane',
    location='Central Metasedimentary Belt, Ontario', lat=44.7, lon=-76.75,
    coord_note=f'44.7, -76.75 = CMB coordinate listed by {HALLS}; no site coordinates printed (Fig. 4 map only)',
    litho=lambda r: kw_litho(r, ('', 'Metamorphic', 'Outcrop')), method_codes='LP-DIR-AF:LP-DIR-T',
    loc_mean_name=lambda r: f"Central Metasedimentary Belt, {r['name']} terrane ({r.comp_paper})")

META['Brown2012a'] = dict(
    ref='Brown & McEnroe (2012) Precambrian Res. 212-213:57-74,', citation='10.1016/j.precamres.2012.04.012',
    location='Adirondack Highlands', lat=44.2, lon=-74.0,
    coord_note=f'44.2, -74.0 = AH coordinate listed by {HALLS}; no site coordinates printed in Brown & McEnroe (2012)',
    site_name=lambda r: str(r.site).replace(' ', ''),
    litho=lambda r: kw_litho(r, ('', 'Metamorphic', 'Outcrop')), method_codes='LP-DIR-AF:LP-DIR-T',
    age_note='remanence age ~970 Ma (paper)')

META['Buchan1973a'] = dict(
    ref='Buchan & Dunlop (1973) Nature Phys. Sci. 246:28-30,', citation='10.1038/physci246028a0',
    location='Haliburton intrusions', lat=44.95, lon=-78.44,
    coord_note=f'44.95, -78.44 = HI coordinate (44 57 N, 78 26.5 W) listed by {HALLS}; not printed in Buchan & Dunlop (1973)',
    litho=lambda r: kw_litho(r, ('Diorite', 'Intrusive:Igneous', 'Pluton')), method_codes='LP-DIR-AF:LP-DIR-T',
    level=lambda r: 'sample' if str(r.comp_paper).startswith('B') else 'site',
    sample_name=lambda r: str(r.site), sample_site=lambda r: '',
    extra_desc=lambda r: 'NOTE: same collection re-reported with revised values by Buchan & Dunlop (1976); do not combine both studies')

def b76_level(r):
    s = str(r.site)
    return 'sample' if '(sample' in s else 'site'
META['Buchan1976a'] = dict(
    ref='Buchan & Dunlop (1976) JGR 81:2951-2967,', citation='10.1029/JB081i017p02951',
    location='Haliburton intrusions', lat=44.95, lon=-78.44,
    coord_note=f'44.95, -78.44 = HI coordinate listed by {HALLS}; no coordinates printed (Fig. 2 graticule 45N, 78 30W)',
    litho=lambda r: kw_litho(r, ('Diorite', 'Intrusive:Igneous', 'Pluton')), method_codes='LP-DIR-AF:LP-DIR-T',
    level=b76_level,
    sample_name=lambda r: re.sub(r'.*\(sample\s*([^)]*)\).*', r'\1', str(r.site)),
    sample_site=lambda r: str(r.site).split(' (')[0],
    comp=lambda r: ('C' if str(r.comp_paper).startswith('C') else ('AR' if str(r.site) == 'G4' and str(r.comp_paper).startswith('A') else r.comp_compiled)))

META['Buchan1978a'] = dict(
    ref='Buchan (1978) CJES 15:1407-1421,', citation='10.1139/e78-148',
    location='Thanet gabbro complex', lat=45.0, lon=-78.0,
    coord_note='45.0, -78.0 = compiler value; no coordinates printed (Fig. 1 map puts Thanet near 44.9N, 78.0W); approximate',
    litho=L('Gabbro'), method_codes='LP-DIR-AF:LP-DIR-T',
    level=lambda r: 'sample' if '(sample' in str(r.site) else 'site',
    sample_name=lambda r: re.sub(r'.*\(sample\s*([^)]*)\).*', r'\1', str(r.site)),
    sample_site=lambda r: str(r.site).split(' (')[0],
    comp=lambda r: ('AN' if str(r.comp_paper).startswith('A') else 'B'))

META['Buchan1983a'] = dict(
    ref='Buchan, Fahrig, Freda & Frith (1983) CJES 20:246-258,', citation='10.1139/e83-022',
    location='Lac St-Jean anorthosite', lat=48.333, lon=-70.95,
    coord_note=f'48.333, -70.95 (48 20 N, 70 57 W) = LA coordinate listed by {HALLS}; Buchan et al. print no site coordinates (Fig. 1 map, sites span ~48.0-49.0N, 70.0-73.0W)',
    litho=lambda r: kw_litho(r, ('Anorthosite', 'Intrusive:Igneous', 'Pluton')), method_codes='LP-DIR-AF:LP-DIR-T',
    age_low=900, age_high=950, age_note='magnetization age 950-900 Ma from calibrated Grenville track (p. 256-257); contact aureole Rb/Sr 1451+-163 Ma (Frith & Doig 1973)',
    loc_lith='Anorthosite:Granite:Monzonite:Diabase', loc_classes='Intrusive:Igneous')

META['Dubois1962a'] = dict(
    ref='Du Bois (1962) GSC Bulletin 71,', citation='10.4095/100589',
    location='Grenville Province, Ontario-Quebec (Du Bois 1962)', lat=45.0, lon=-78.5, force_coords=True,
    coord_note='directions are printed "corrected to 45N, 78.5W" (Table XXI), so lat/lon = 45.0, -78.5 (reference point), NOT the collecting locality',
    litho=lambda r: kw_litho(r, ('', '', '')), method_codes='LP-DIR-AF',
    level=lambda r: 'specimen', sample_name=lambda r: str(r.site).replace(' ', ''), sample_site=lambda r: str(r.site).replace(' ', ''),
    comp=lambda r: 'AR', age_note='magnetization related to Grenville metamorphism ~1050 m.y. (pp. 71-72)')

def d85a_age(r):
    c = str(r.comp_compiled)
    if c == 'C': return (np.nan, 445, 450)
    return (np.nan, np.nan, np.nan)
META['Dunlop1985a'] = dict(
    ref='Dunlop & Stirling (1985) Geophys. J. R. astr. Soc. 81:521-550,', citation='10.1111/j.1365-246X.1985.tb06420.x',
    location='Cordova gabbro', lat=44.5, lon=-77.75,
    coord_note=f'44.5, -77.75 = CG coordinate listed by {HALLS}; no coordinates printed (Fig. 2 grid 44 30-35 N, 77 45-50 W)',
    litho=L('Gabbro'), method_codes='LP-DIR-AF:LP-DIR-T', age_fn=d85a_age,
    quality=lambda r: 'b' if (str(r.comp_compiled) == 'C' and str(r.site) in ('13', '15')) else '',
    age_note='intrusion ~1180-1200 Ma; A ~900 Ma and B ~850-800 Ma by APWP comparison; C 445-450 Ma (40Ar/39Ar plagioclase 446, 447 Ma) (p. 523, 541)')

META['Dunlop1985b'] = dict(
    ref='Dunlop, Hyodo, Knight & Steele (1985) Geophys. J. R. astr. Soc. 83:699-720,', citation='10.1111/j.1365-246X.1985.tb04333.x',
    location='Tudor gabbro', lat=44.75, lon=-77.683,
    coord_note=f'44.75, -77.683 (44 45 N, 77 41 W) = TG coordinate listed by {HALLS}; no coordinates printed (Fig. 2 grid 44 42-45 N, 77 35-40 W)',
    litho=L('Gabbro'), method_codes='LP-DIR-AF:LP-DIR-T', age=1110,
    age_note='A interpreted as primary TRM; best age ~1110 Ma (hornblende 40Ar/39Ar cooling, pp. 699, 702)')

META['Fahrig1972a'] = dict(
    ref='Fahrig & Larochelle (1972) CJES 9:1287-1296,', citation='10.1139/e72-112',
    location='Michael gabbro, Labrador', lat=54.5, lon=-59.0,
    coord_note='54.5, -59.0 = compiler approximation; NOT printed in the paper (Fig. 1 map, sites ~54.2-54.8N, 58-61W)',
    litho=L('Gabbro', 'Intrusive:Igneous', 'Sill'), method_codes='LP-DIR-AF', age=1500,
    age_note='radioisotopic age ~1500 m.y.; magnetization interpreted as acquired on cooling (abstract)')

def f74_coords(r):
    if 'Shabogamo' in str(r.unit):
        return (53.5, -65.0, 'compiler approximation 53.5, -65.0; NOT printed in the paper (Fig. 6 map)')
    return (53.1, -60.7, f'53.1, -60.7 = Mealy Mountains (MM) coordinate listed by {HALLS}; no site coordinates printed (Fig. 1 map); the original compiled csv used 53,-59 (NW comp.) and 53,-61 (east comp.)')
def f74_mean_coords(r):
    if 'Shabogamo' in r['name']:
        return (53.5, -65.0, 'compiler approximation; not printed')
    return (53.1, -60.7, f'Mealy Mountains coordinate from {HALLS}')
def f74_dir(r):
    return r
META['Fahrig1974a'] = dict(
    ref='Fahrig, Christie & Schwarz (1974) CJES 11:18-29,', citation='10.1139/e74-002',
    location_fn=lambda r: 'Shabogamo Gabbro' if 'Shabogamo' in str(r.unit) else 'Mealy Mountain anorthosite suite',
    location='Mealy Mountains / Shabogamo', coords=f74_coords, mean_coords=f74_mean_coords,
    loc_mean_name=lambda r: r['name'],
    litho=lambda r: ('Gabbro', 'Intrusive:Igneous', 'Sill') if 'Shabogamo' in str(r.unit) else ('Anorthosite', 'Intrusive:Igneous', 'Pluton'),
    method_codes='LP-DIR-AF', age_note='no radiometric ages; Mealy and Shabogamo inferred ~1500-1600 m.y. (pp. 20, 24); NW component = primary TRM, east component = later thermal overprint (p. 22-23)')

META['Halls2015b'] = dict(
    ref='Halls (2015) Geology 43:1051-1054, GSA Data Repository 2015352,', citation='10.1130/G37188.1',
    location='Grenville Province', sites=False, lat=np.nan, lon=np.nan, coord_note='', method_codes='',
    means_filter=lambda r: not str(r.table_ref).startswith('"Halls et al. 2015 PR') and not str(r.table_ref).startswith('Halls et al. 2015 PR'),
    loc_mean_name=lambda r: f"{r['name']} [{str(r.comp_paper).split(' (')[0]}]",
    mean_coords=lambda r: (np.nan, np.nan, 'not printed'), coords='x')

META['Hargraves1967a'] = dict(
    ref='Hargraves & Burt (1967) CJES 4:357-369,', citation='10.1139/e67-018',
    location_fn=lambda r: 'Allard Lake anorthosite suite, ' + str(r.site), location='Allard Lake anorthosite suite',
    lat=51.0, lon=-63.0, coord_note='51N, 63W = study area (p. 358); no per-locality coordinates',
    site_name=lambda r: f"{r.site} {r.unit}",
    litho=lambda r: kw_litho(r, ('Anorthosite', 'Intrusive:Igneous', 'Pluton')), method_codes='LP-DIR-AF:LP-DIR-T',
    age_note='K-Ar ~1000 m.y. (Leech et al. 1963); magnetization related to Grenville metamorphism (p. 368)',
    loc_mean_name=lambda r: 'Allard Lake anorthosite suite: ' + r['name'])

META['Hargraves1974a'] = dict(
    ref='Hargraves & Roy (1974) CJES 11:854-859,', citation='10.1139/e74-085',
    location='St-Urbain anorthosite / Charlevoix structure', lat=47.5, lon=-70.33,
    coord_note='47.5, -70.33 = Fig. 1 graticule ticks (N 47 30, 70 20 W) near Mont des Eboulements, used by the compiler; no site coordinates printed; sites lie within ~40 km',
    litho=lambda r: kw_litho(r, ('Anorthosite', 'Intrusive:Igneous', 'Pluton')), method_codes='LP-DIR-AF',
    quality=lambda r: 'b' if str(r.site) in ('90', '91', '92', '98', '99', '100') else '',
    age_note='impact structure K-Ar ~350 m.y.; primary RM of anorthosite Grenvillian (pole similar to other Grenville poles)')

META['Irving1972a'] = dict(
    ref='Irving, Park & Roy (1972) Nature 236:344-346,', citation='10.1038/236344a0',
    location='Ottawa intrusions', lat=np.nan, lon=np.nan, coord_note='',
    litho=L('Diorite:Gabbro'), method_codes='LP-DIR-AF',
    mean_coords=lambda r: (45.7, -76.24, f'OI coordinate listed by {HALLS}'), coords=lambda r: (np.nan, np.nan, ''))

def irv_lvl(r):
    return 'site'
META['Irving1974b'] = dict(
    ref='Irving, Park & Emslie (1974) JGR 79:5482-5490,', citation='10.1029/JB079i035p05482',
    location='Morin anorthosite complex', lat=46.0, lon=-74.25,
    coord_note=f'46.0, -74.25 (46 00 N, 74 15 W) = MA coordinate listed by {HALLS}; paper gives UTM grid refs (zone 18, 6-digit, Table 3) without 100-km square IDs, not converted',
    litho=L('Anorthosite:Leucogabbro', 'Igneous:Metamorphic', 'Pluton'), method_codes='LP-DIR-AF:LP-DIR-T',
    quality=lambda r: 'b' if 'omitted by authors' in str(r.notes) else '',
    site_name=lambda r: str(r.site) + (' mcf' if 'mcf' in str(r.comp_paper) else ''),
    age_note='hcf: acquired during cooling after high-grade metamorphism ~1124+-27 m.y.; mcf: ~1000 m.y. (abstract)')

META['McWilliams1975a'] = dict(
    ref='McWilliams & Dunlop (1975) Science 190:269-272,', citation='10.1126/science.190.4211.269',
    location='Magnetawan', lat=45.75, lon=-79.667, coord_note='45 45 N, 79 40 W as printed (p. 270)',
    litho=L('Metasediment', 'Metamorphic', 'Outcrop'), method_codes='LP-DIR-AF:LP-DIR-T', tilt=0,
    mean_coords=lambda r: (45.75, -79.667, 'as printed'), coords=lambda r: (45.75, -79.667, 'as printed'))

def mur_loc(r):
    return 'Steel Mountain anorthosite' if 'Steel' in str(r.unit) else 'Indian Head anorthosite'
META['Murthy1976a'] = dict(
    ref='Murthy & Rao (1976) CJES 13:75-83,', citation='10.1139/e76-007',
    location_fn=mur_loc, location='Steel Mountain / Indian Head anorthosites', lat=48.5, lon=-58.5,
    coord_note='48.5N, 58.5W as printed for both inliers (p. 76); no per-site coordinates',
    site_name=lambda r: ('SM' if 'Steel' in str(r.unit) else 'IH') + '-' + str(r.site).replace(', ', '+'),
    comp=lambda r: (np.nan if str(r.comp_compiled) in ('nan', '') else str(r.comp_compiled).rstrip('?')),
    quality=lambda r: 'b' if ('Steel' in str(r.unit) and str(r.site) == '2') else '',
    litho=L('Anorthosite', 'Igneous:Metamorphic', 'Pluton'), method_codes='LP-DIR-AF',
    age_note='Indian Head: Grenvillian (~1000 m.y.) magnetization; Steel Mountain sites 3-10: lower Paleozoic (~451 m.y., K-Ar chlorite) secondary magnetization (p. 82)',
    loc_mean_name=lambda r: r['name'])

def p73_filter(r):
    return 'after 0 Oe' not in str(r.comp_paper)
def p73_coords(r):
    if 'Tudor' in str(r.unit):
        return (44.75, -77.683, f'Tudor gabbro: no coordinates printed; 44.75, -77.683 (44 45 N, 77 41 W) = TG coordinate listed by {HALLS}. The original compiled csv had 46.33,-80.40 (= St. Charles anorthosite) in error')
    return (np.nan, np.nan, '')
META['Palmer1973a'] = dict(
    ref='Palmer & Carmichael (1973) CJES 10:1175-1190,', citation='10.1139/e73-104',
    location_fn=lambda r: str(r.unit).replace('Tudor Gabbro', 'Tudor gabbro'), location='Grenville Province (Palmer & Carmichael 1973)',
    coords=p73_coords, mean_coords=lambda r: (np.nan, np.nan, 'not printed'), site_filter=p73_filter,
    site_name=lambda r: {'Wilberforce pyroxenite': 'WP', 'Tudor Gabbro': 'TG', 'River Valley anorthosite': 'RVA', 'Fall Lake Complex': 'FLC'}.get(str(r.unit), str(r.unit)[:3].upper()) + ('_block' if str(r.site) == 'block samples' else str(r.site)),
    quality=lambda r: 'b' if (str(r.comp_compiled) in ('nan', '')) else '',
    litho=lambda r: kw_litho(r, ('', 'Intrusive:Igneous', 'Pluton')), method_codes='LP-DIR-AF',
    age_note='southern units magnetized probably before the ~950 m.y. K-Ar event; anorthosite poles close to ~1050 m.y. North American poles (p. 1188)')

def p79_quality(r):
    return 'b' if 'not used in any mean' in str(r.notes) else ''
META['Palmer1979a'] = dict(
    ref='Palmer, Hayatsu, Waboso & Pullan (1979) CJES 16:459-471,', citation='10.1139/e79-042',
    location='Umfraville gabbro', lat=44.94, lon=-77.8,
    coord_note=f'44.94, -77.8 = UG coordinate listed by {HALLS} (Table DR4); no coordinates printed (Fig. 1 grid 44 56-57 N, 77 45-50 W)',
    litho=lambda r: kw_litho(r, ('Gabbro', 'Intrusive:Igneous', 'Pluton')), method_codes='LP-DIR-AF:LP-DIR-T', quality=p79_quality,
    level=lambda r: 'sample' if str(r.site).startswith('U') else 'site',
    sample_name=lambda r: str(r.site), sample_site=lambda r: 'U_blocks',
    age_note='zircon (related syenite) minimum emplacement age 1160 Ma; K-Ar whole-rock isochron 911+-11 Ma, to which the magnetization may be more closely related')

META['Park1972a'] = dict(
    ref='Park & Irving (1972) CJES 9:763-765,', citation='10.1139/e72-063',
    location='Frontenac Axis dikes', lat=np.nan, lon=np.nan, coord_note='',
    coords=lambda r: (np.nan, np.nan, ''), mean_coords=lambda r: (44.3, -76.25, f'KD (Kingston dykes) coordinate listed by {HALLS}'),
    litho=L('Diabase', 'Intrusive:Igneous', 'Dike'), method_codes='LP-DIR-AF',
    quality=lambda r: 'b' if str(r.site) in ('12', '4*') else '',
    age_note='NW-trending dikes: K-Ar whole rock 751+-75 and biotite 817+-70 m.y. (site 5), minimum age ~800 m.y.; NE-trending dikes early Paleozoic K-Ar (minimum ages)')

META['Park1996a'] = dict(
    ref='Park & Gower (1996) CJES 33:746-756,', citation='10.1139/e96-057',
    location='NE Grenville Province, Labrador (Groswater Bay terrane)', lat=54.5, lon=-58.5,
    coord_note='mean site locality 54.5N, 058.5W (Table 1 notes)', litho=L(''), method_codes='LP-DIR-AF:LP-DIR-T', loc_lith='Gabbro',
    loc_classes='Intrusive:Igneous',
    age_note='C: Grenville overprint ~970 Ma; B: probably ~615 Ma Lake Melville rift overprint (or Grenvillian); A: latest Grenville or Neoproterozoic (abstract)')

META['Robertson1979a'] = dict(
    ref='Robertson & Roy (1979) CJES 16:1842-1856,', citation='10.1139/e79-168',
    location='St-Urbain anorthosite / Charlevoix structure', lat=47.5, lon=-70.33,
    coord_note='47.5, -70.33 = compiler value (Fig. 1 of Hargraves & Roy 1974 graticule, 47 30 N 70 20 W); no site coordinates printed',
    litho=lambda r: kw_litho(r, ('Anorthosite', 'Intrusive:Igneous', 'Pluton')), method_codes='LP-DIR-AF:LP-DIR-T',
    quality=lambda r: 'b' if str(r.site) in ('2', '4') else '',
    age_note='pole falls on the ~950 Ma calibration of the Grenville track; ilmenite dykes cutting anorthosite ~890 Ma (Rose 1961); impact 360+-25 Ma')

def u76_coords(r):
    s = str(r.site)
    d = {'66': (49.5, -74.0), '51': (50.0, -74.0), '54': (50.0, -74.0), '70': (49.75, -74.0), '72': (49.75, -74.0)}
    la, lo = d.get(s, (np.nan, np.nan))
    return (la, lo, 'compiler approximation from the Fig. 4 map (map frame 49 30-50 00 N, 74-75 W); NOT printed in the paper')
META['Ueno1976a'] = dict(
    ref='Ueno & Irving (1976) Precambrian Res. 3:303-315,', citation='10.1016/0301-9268(76)90024-3',
    location='Chibougamau', coords=u76_coords, mean_coords=lambda r: (np.nan, np.nan, 'not printed'),
    litho=lambda r: kw_litho(r, ('', '', '')), method_codes='LP-DIR-AF:LP-DIR-T', tilt=0, age=1000,
    age_note='CH magnetization acquired during Grenvillian post-orogenic uplift ~1000 m.y. (abstract)')

def w_filter(r):
    return 'belongs to Buchan1976a' not in str(r.comp_compiled)
def w_age(r):
    return (1015, 1000, 1030) if str(r.comp_compiled) == 'AR' else (600, np.nan, np.nan)
META['Warnock2000a'] = dict(
    ref='Warnock, Kodama & Zeitler (2000) JGR 105(B8):19435-19453,', citation='10.1029/2000JB900114',
    location='Haliburton intrusions (Glamorgan gabbro lobe)', lat=44.8875, lon=-78.3875,
    coord_note="locality 44 53'15\"N, 78 23'15\"W as printed (p. 19,436)",
    litho=L('Gabbro'), method_codes='LP-DIR-T:LP-LT:DE-BFL', site_filter=w_filter, tilt=0,
    level=lambda r: 'specimen', sample_name=lambda r: str(r.site), sample_site=lambda r: str(r.site),
    mad=lambda r: num(re.search(r'MAD=([0-9.]+)', str(r.notes)).group(1)) if re.search(r'MAD=([0-9.]+)', str(r.notes)) else np.nan,
    mean_age_fn=lambda r: (1015, 1000, 1030) if 'HbA' in r['name'] else ((600, np.nan, np.nan) if 'combined' in r['name'] else (np.nan, np.nan, np.nan)),
    age_note='HbA age 1015+-15 Ma; HbB ~600 Ma (Table 11)')

def seg_comp(r):
    c = str(r.comp_compiled)
    if c in ('AN',): return 'AN'
    return str(r.comp_paper).replace('*', '')
META['Seguin1984a'] = dict(
    ref='Seguin & Brun (1984) Precambrian Res. 26:307-331,', citation='10.1016/0301-9268(84)90006-8',
    location='Cheneaux metagabbro', lat=45.583, lon=-76.667,
    coord_note='no coordinates printed (Fig. 2 graticule misprinted); 45.583, -76.667 = Cheneaux Falls locality printed by Irving, Park & Roy (1972, Nature 236:345) for the same intrusion; Table II poles recompute at ~45.6N 76.7W. The original compiled csv had 46.58,-77.75 (misprinted Fig. 2 graticule)',
    site_filter=lambda r: not str(r.site).startswith('(extra'),
    comp=seg_comp, litho=lambda r: kw_litho(r, ('Marble', 'Metamorphic', 'Outcrop')) if 'marble' not in str(r.unit).lower() else ('Marble', 'Metamorphic', 'Outcrop'),
    method_codes='LP-DIR-AF:LP-DIR-T',
    age_note='no radiometric age; APWP ages: W+ ~1300 Ma, W- ~1075 Ma, SE ~950 Ma (abstract; 975 Ma p. 326), NE ~750 Ma; SE fails baked-contact test (overprint)')

META['Stupavsky1982a'] = dict(
    ref='Stupavsky & Symons (1982) CJES 19:819-828,', citation='10.1139/e82-068', sites=False,
    location='French River anorthosites', lat=46.1, lon=-80.6, means_file='means_combined.csv',
    coord_note='no coordinates printed; compiled 46.25,-80.75 is the NW-corner graticule tick of Fig. 1; sites lie ~46.0-46.2N, 80.5-80.7W; lat/lon here = 46.1, -80.6 approximate centre (not published)',
    litho=L('Anorthosite', 'Igneous:Metamorphic', 'Pluton'), method_codes='LP-DIR-AF:LP-DIR-T', loc_lith='Anorthosite',
    loc_mean_name=lambda r: 'French River anorthosites: ' + r['name'],
    age_note='A: Grenvillian overprint ~975 Ma (magnetite); B ~1725 Ma; C ~1800 Ma; D ~1250 Ma (paper)')

META['Symons1978a'] = dict(
    ref='Symons (1978) CJES 15:956-962,', citation='10.1139/e78-103',
    location='Umfraville gabbro', lat=44.94, lon=-77.79, coord_note='mean site location 44.94N, 77.79W as printed (Table 1 footnote)',
    site_filter=lambda r: str(r.dec) not in ('', 'nan'),
    quality=lambda r: '' if str(r.site) in ('1', '3', '4', '6', '8', '9', '11', '14', '16', '17') else 'b',
    litho=L('Gabbro'), method_codes='LP-DIR-AF:LP-DIR-T', age=1180, age_low=1160, age_high=1200,
    age_note='U-Pb zircon (syenite phase) 1180+-20 Ma; remanence interpreted as primary TRM')

META['Ueno1975a'] = dict(
    ref='Ueno, Irving & McNutt (1975) CJES 12:209-226,', citation='10.1139/e75-019',
    location='Whitestone anorthosite and diorite', lat=45.658, lon=-79.867,
    coord_note=f'45.658, -79.867 (45 39.5 N, 79 52 W) = WA/WD coordinate listed by {HALLS}; Ueno et al. print per-site UTM refs with zone "18T" (probably a misprint for 17T), not converted here',
    litho=lambda r: ('Diorite', 'Intrusive:Igneous', 'Pluton') if 'diorite' in str(r.unit).lower() else ('Anorthosite', 'Intrusive:Igneous', 'Pluton'),
    method_codes='LP-DIR-AF:LP-DIR-T',
    age_note='TRMs acquired during slow cooling after metamorphism (1100-1000 m.y.); WD hornblende 40Ar/39Ar ~980 Ma (Dallmeyer & Sutter 1980, cited by Buchan et al. 1983)')

META['Hyodo1993a'] = dict(
    ref='Hyodo & Dunlop (1993) JGR 98(B5):7997-8017,', citation='10.1029/92JB02915',
    location='Mattawa', lat=46.25, lon=-78.18,
    coord_note=f'46.25, -78.18 = MAT coordinate listed by {HALLS}; no coordinates printed (Fig. 1 ~46 14-16 N, 78 05-15 W)',
    site_filter=lambda r: str(r.dec) not in ('', 'nan'), level=lambda r: 'sample',
    sample_name=lambda r: str(r.site), sample_site=lambda r: 'GD8', comp=lambda r: 'AR',
    litho=L('Gneiss', 'Metamorphic', 'Outcrop'), method_codes='LP-DIR-T:DE-BFL:DA-AC-ANI', age=1009, age_low=1007, age_high=1011,
    age_note='hornblende 40Ar/39Ar 1009+-2 Ma; directions are anisotropy-corrected (Table 4)', loc_lith='Gneiss',
    loc_mean_name=lambda r: r['name'])

META['Park1983a'] = dict(
    ref='Park & Emslie (1983) CJES 20:1818-1833 [UNVERIFIED: source not accessible],', citation='10.1139/e83-173',
    location='Mealy dykes, Labrador', lat=53.7, lon=-59.0, coord_note='53.7, -59.0 = compiler value, NOT VERIFIED',
    site_name=lambda r: str(r.unit).split('compiled ')[1].split(' (')[0].replace(' ', '') + '_site' + str(r.site),
    litho=L('Diabase', 'Intrusive:Igneous', 'Dike'), method_codes='')

if __name__ == '__main__':
    for study, M in META.items():
        if ONLY and study not in ONLY:
            continue
        src = 'Halls2015a' if study == 'Halls2015b' else study
        out = build(src, M, [f'{OUT}/{study}'])
        print(study, {k: len(v) for k, v in out.items()})
    print('\nQC:')
    for q in QC:
        print(' ', q)
