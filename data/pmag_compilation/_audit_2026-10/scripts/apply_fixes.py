"""Apply audited corrections to the ORIGINAL compiled csv files, cell by cell, preserving BOM / CRLF / formatting.
Every change asserts the old value first. Writes fixes_log.json (used for the CHANGELOG)."""
import csv, io, json, sys
D = '/Users/yimingzhang/Github/Grenville_seminar/data/pmag_compilation'
LOG = '/private/tmp/claude-501/-Users-yimingzhang-Github-Sweden-dikes/3865b26b-616b-4c68-b473-673d942898fa/scratchpad/build/fixes_log.json'
DRY = '--dry' in sys.argv

FIX = []   # (file, row(0-based data row), column, old, new, evidence)
def f(file, row, col, old, new, ev):
    FIX.append((file, row, col, old, new, ev))

# ---- Brown2012a
f('Brown2012a', 23, 'k', '46', '49', 'Brown & McEnroe 2012 Table 1 p.65, site ADH34 k=49')
f('Brown2012a', 25, 'vgp_lat', '-3.4', '-33.4', 'Brown & McEnroe 2012 Table 1 p.65, site ADH8 VGP lat -33.4')
# ---- Buchan1973a
f('Buchan1973a', 3, 'dir_k', '27', '28', 'Buchan & Dunlop 1973 Table 1 p.29, site D5 k=28')
# ---- Buchan1976a
f('Buchan1976a', 4, 'dir_inc', '-75.1', '-75', 'Buchan & Dunlop 1976 Table 1 p.2957, site B9 I=-75.0')
# ---- Buchan1983a: dir_n_samples held N (cores collected) instead of n (cores used)
b83 = {0: ('7', '4'), 5: ('7', '5'), 6: ('8', '3'), 7: ('7', '3'), 8: ('6', '5'), 10: ('7', '5'), 11: ('8', '3'), 12: ('7', '6'), 13: ('8', '7'),
       14: ('7', '5'), 17: ('8', '7'), 21: ('6', '5'), 22: ('7', '6'), 23: ('7', '5'), 25: ('6', '4'), 28: ('8', '3'), 29: ('6', '3'), 32: ('8', '7')}
for r, (o, n) in b83.items():
    f('Buchan1983a', r, 'dir_n_samples', o, n, 'Buchan et al. 1983 Tables 1-2 pp.253-254: column n (stable cores used; sums to "All cores") instead of N (cores collected)')
# ---- Dubois1962a
f('Dubois1962a', 13, 'dir_inc', '-54.5', '-54', 'Du Bois 1962 GSC Bull. 71 Table XXI p.67, W2 I=-54.0')
for r in range(16):
    lat_old = {**{i: '45.78' for i in range(10)}, 10: '45.08', 11: '45.08', 12: '45.68', 13: '45.68', 14: '45.68', 15: '45.68'}[r]
    lon_old = {**{i: '-78.92' for i in range(10)}, 10: '-78.25', 11: '-78.25', 12: '-75.88', 13: '-75.88', 14: '-75.88', 15: '-75.88'}[r]
    f('Dubois1962a', r, 'lat', lat_old, '45', 'Du Bois 1962 Table XXI title: directions "corrected to 45N and 78.5W" -> VGPs must be computed at the reference point, not the locality')
    f('Dubois1962a', r, 'lon', lon_old, '-78.5', 'idem')
# ---- Dunlop1985a
f('Dunlop1985a', 9, 'dir_dec', '297', '297.5', 'Dunlop & Stirling 1985 Table 1 p.530, site 13 A D=297.5')
f('Dunlop1985a', 19, 'dir_alpha95', '8', '8.5', 'Dunlop & Stirling 1985 Table 3 p.533, site 2 B a95=8.5')
# ---- Fahrig1972a: N -> n
f72 = {0: ('6', '5'), 1: ('6', '5'), 2: ('7', '6'), 8: ('6', '4'), 9: ('7', '4'), 11: ('7', '5'), 13: ('5', '3'), 15: ('5', '4'), 17: ('5', '4')}
for r, (o, n) in f72.items():
    f('Fahrig1972a', r, 'dir_n_samples', o, n, 'Fahrig & Larochelle 1972 Table 1 p.1290: n (samples used after rejecting theta>20 cores) instead of N (collected)')
# ---- Fahrig1974a: N -> n ; site 33 label
f74 = {0: ('7', '6'), 1: ('7', '3'), 2: ('6', '5'), 3: ('7', '5'), 4: ('7', '3'), 7: ('6', '5'), 9: ('7', '6'), 10: ('6', '5'), 11: ('7', '2'), 14: ('5', '2'),
       15: ('7', '2'), 18: ('6', '3'), 19: ('8', '6'), 20: ('6', '4'), 21: ('7', '3'), 22: ('6', '4'), 24: ('6', '4'), 26: ('7', '5'), 29: ('6', '5'), 34: ('7', '3'),
       35: ('7', '5'), 36: ('7', '4')}
for r, (o, n) in f74.items():
    f('Fahrig1974a', r, 'dir_n_samples', o, n, 'Fahrig et al. 1974 Tables 1-2 pp.22, 28: n (samples used) instead of N (cores collected)')
f('Fahrig1974a', 10, 'dir_comp_name', 'AN', 'AR', 'Fahrig et al. 1974 Table 1 p.22 lists site 33 (302/+37) in the Mealy Mountains NORTHWEST component (= compiled AR), not the east component')
# ---- Halls2015a_compilation
f('Halls2015a_compilation', 17, 'vgp_lon', '33', '327', 'Buchan et al. 1983 Table 2: north pole 213W 19S -> antipode 19N 33W = 327E (W longitude had been copied as E); Halls 2015 DR3 gives 19.3N 326.1E')
f('Halls2015a_compilation', 18, 'vgp_lon', '13', '347', 'Buchan et al. 1983 Table 1: south pole 193W 8S -> antipode 8N 13W = 347E (W longitude had been copied as E)')
f('Halls2015a_compilation', 15, 'dp', '9.9', '9.2', 'Irving et al. 1974 Table 4 p.5486: Model A pole error (dm, dp) = 9.9, 9.2 (dp/dm were swapped)')
f('Halls2015a_compilation', 15, 'dm', '9.2', '9.9', 'idem')
f('Halls2015a_compilation', 16, 'dir_k', '44', '17', 'Irving et al. 1974 Table 4 p.5486: M2 precision k=17')
f('Halls2015a_compilation', 16, 'dp', '12', '9', 'Irving et al. 1974 Table 4: M2 error (dm, dp) = 12, 9 (dp/dm were swapped)')
f('Halls2015a_compilation', 16, 'dm', '9', '12', 'idem')
for col, o, n in [('B', '8', '6'), ('dir_dec', '123.4', '121.8'), ('dir_inc', '32.5', '30.3'), ('dir_k', '79', '141'), ('dir_alpha95', '6.3', '5.1'),
                  ('vgp_lat', '-9', '-9.2'), ('vgp_lon', '334.6', '336.7'), ('dp', '4.2', '3'), ('dm', '7.3', '5.6'), ('pole_alpha95', '6.3', '5.1')]:
    f('Halls2015a_compilation', 28, col, o, n, 'Halls 2015 Geology GSA DR2015352 Table DR1 p.3, row BI (the compiled row had copied the adjacent SL = Seal Lake row)')
# ---- Hargraves1974a labels (rows are dropped by the notebook anyway)
f('Hargraves1974a', 2, 'dir_comp_name', 'AR', '', 'Hargraves & Roy 1974 p.857: site 90 direction anomalous (shocked, 12-14 km from centre), not a reversed Grenville A')
f('Hargraves1974a', 3, 'dir_comp_name', 'AN', '', 'Hargraves & Roy 1974 Table 1/p.858: site 91 = impactite (K-Ar ~350 m.y.), not Grenville A')
f('Hargraves1974a', 13, 'dir_comp_name', 'AR', 'AN', 'Hargraves & Roy 1974 p.857: site 101 (112/-10) = primary (SE) vector tectonically rotated, like sites 96, 97')
# ---- Irving1972a
f('Irving1972a', 8, 'dir_comp_name', 'AN', 'AR', 'Irving et al. 1972 Table 1 p.345: G18 (R) 267/-74 is reversed (NW-up)')
# ---- Irving1974b
f('Irving1974b', 19, 'dir_dec', '198', '298', 'Irving et al. 1974 Table 3 p.5486: site 57 = 298, -70 (4.71)')
f('Irving1974b', 19, 'dir_inc', '0', '-70', 'idem')
# ---- Murthy1976a
f('Murthy1976a', 1, 'dir_comp_name', 'AR', '', 'Murthy & Rao 1976 p.78: Steel Mountain site 2 (143/-14.5) "entirely anomalous", excluded; not a reversed A direction')
# ---- Palmer1973a
f('Palmer1973a', 0, 'dir_k', '58', '56.8', 'Palmer & Carmichael 1973 Table 1 p.1179: block samples K=56.8')
f('Palmer1973a', 5, 'dir_dec', '333.7', '337.7', 'Palmer & Carmichael 1973 Table 2 p.1182: Tudor site 2 D=337.7 (printed VGP 24.0N 125.8E recomputes from 337.7)')
for r in range(4, 14):
    f('Palmer1973a', r, 'lat', '46.33', '44.75', 'Tudor gabbro: 46.33,-80.40 is the St. Charles anorthosite locality (p.1180-1181); Tudor coordinate 44 45 N 77 41 W from Halls 2015 GSA DR2 (TG); Palmer & Carmichael Fig. 3 grid 44 43-45 N, 77 35-40 W')
    f('Palmer1973a', r, 'lon', '-80.40', '-77.683', 'idem')
# ---- Palmer1979a
f('Palmer1979a', 20, 'vgp_lon', '161', '-161', 'Palmer et al. 1979 Table 2 p.463: SP12 VGP printed 161W 36S')
# ---- Park1972a
f('Park1972a', 8, 'vgp_lon', '163', '197', 'Park & Irving 1972 Table 1 p.763: dike 12 pole printed 20N, 163W (= 197E)')
f('Park1972a', 8, 'dir_comp_name', 'AN', '', 'dike 12 has a westerly direction (283/+30) with K-Ar 407-411 m.y.; not Grenville A normal')
# ---- Park1996a
for r in range(8):
    f('Park1996a', r, 'lon', '-59', '-58.5', 'Park & Gower 1996 Table 1 notes: mean site locality 54.5N, 058.5W')
f('Park1996a', 7, 'dir_comp_name', 'AR', '', 'Park & Gower 1996 p.749: component E is a recent overprint (reverse of PEF), not Grenville A')
# ---- Seguin1984a
for r in range(17):
    f('Seguin1984a', r, 'lat', '46.58', '45.583', 'Seguin & Brun 1984 print no coordinates (Fig. 2 graticule misprinted); Cheneaux Falls 45 35 N 76 40 W (Irving et al. 1972 p.345); Table II poles recompute at ~45.6N 76.7W')
    f('Seguin1984a', r, 'lon', '-77.75', '-76.667', 'idem')
for r in range(11, 17):
    f('Seguin1984a', r, 'dir_comp_name', 'AR', 'W-', 'Seguin & Brun 1984 Table I: these rows are component W- (WSW, intermediate up, ~1075 Ma), not the reverse of SE (= compiled AN)')
# ---- Stupavsky1982a: k not published
for r, o in enumerate(['136', '162', '100', '2178']):
    f('Stupavsky1982a', r, 'dir_k', o, '', 'Stupavsky & Symons 1982 (CJES 19:819-828) Table 2 p.824 prints no k; compiled k = (140/a95)^2 (invalid)')
# ---- Symons1978a
for r in range(16):
    f('Symons1978a', r, 'lat', '44.95', '44.94', 'Symons 1978 Table 1 footnote p.958: mean site location 44.94N, 77.79W')
    f('Symons1978a', r, 'lon', '-77.75', '-77.79', 'idem')
# ---- Warnock2000a
for r in range(23):
    f('Warnock2000a', r, 'lat', '44.95', '44.8875', 'Warnock et al. 2000 p.19,436: locality 44 53\'15"N, 78 23\'15"W')
    f('Warnock2000a', r, 'lon', '-78.44', '-78.3875', 'idem')

ADD_COL = [('Hyodo1993a', 'dir_comp_name', 'AR', 'Hyodo & Dunlop 1993 Table 4 p.8010 / Fig. 12: Grenvillian host NRM (NW-up, plotted as reversed); column was missing')]

def load(name):
    raw = open(f'{D}/{name}.csv', 'rb').read()
    bom = raw.startswith(b'\xef\xbb\xbf')
    txt = raw.decode('utf-8-sig')
    final_nl = txt.endswith('\n')
    rows = list(csv.reader(io.StringIO(txt)))
    return rows, bom, final_nl
def save(name, rows, bom, final_nl):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\r\n')
    w.writerows(rows)
    s = buf.getvalue()
    if not final_nl:
        s = s[:-2]
    data = s.encode('utf-8')
    if bom:
        data = b'\xef\xbb\xbf' + data
    if not DRY:
        open(f'{D}/{name}.csv', 'wb').write(data)

files = sorted({x[0] for x in FIX} | {x[0] for x in ADD_COL})
log = []
for name in files:
    rows, bom, fnl = load(name)
    hdr = rows[0]
    for (fn, r, col, old, new, ev) in [x for x in FIX if x[0] == name]:
        j = hdr.index(col)
        cur = rows[r + 1][j]
        try:
            same = (cur == old) or (float(cur) == float(old))
        except ValueError:
            same = cur == old
        assert same, f'{name} row {r} {col}: expected {old!r}, found {cur!r}'
        rows[r + 1][j] = new
        log.append(dict(file=f'{name}.csv', row=r, site=rows[r + 1][hdr.index('site') if 'site' in hdr else 1], column=col, old=cur, new=new, evidence=ev))
    for (fn, col, val, ev) in [x for x in ADD_COL if x[0] == name]:
        if col not in hdr:
            hdr.append(col)
            for rr in rows[1:]:
                rr.append(val)
            log.append(dict(file=f'{name}.csv', row='all', site='', column=col, old='(column absent)', new=val, evidence=ev))
    save(name, rows, bom, fnl)
json.dump(log, open(LOG, 'w'), indent=1)
print(len(log), 'changes', '(dry run)' if DRY else 'APPLIED')
