"""Transcriptions made by the lead auditor (values read visually from the PDFs; see MASTER_NOTES.md).
Writes verify/<Study>/sites_transcribed.csv and means_transcribed.csv in the SPEC schema."""
import csv, os
V='/private/tmp/claude-501/-Users-yimingzhang-Github-Sweden-dikes/3865b26b-616b-4c68-b473-673d942898fa/scratchpad/verify'
SC=['table_ref','site','unit','comp_paper','comp_compiled','dec','inc','k','a95','n','n_type','N_total','r','site_lat','site_lon','vgp_lat','vgp_lon','dp','dm','tilt','polarity_note','notes']
MC=['table_ref','name','comp_paper','comp_compiled','N','n_type','dec','inc','k','a95','lat','lon','pole_lat','pole_lon','dp','dm','A95','K','tilt','notes']
def w(study, sites, means):
    os.makedirs(f'{V}/{study}',exist_ok=True)
    for fn,cols,rows in [('sites_transcribed.csv',SC,sites),('means_transcribed.csv',MC,means)]:
        with open(f'{V}/{study}/{fn}','w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=cols); wr.writeheader()
            for r in rows: wr.writerow({k:r.get(k,'') for k in cols})

# ---------------- Buchan1983a ----------------
T1='Table 1, p. 253'; T2='Table 2, p. 254'
s=[]
for site,N,n,af,D,I,k,a in [(1,7,4,25,103,24,130,8.1),(2,7,7,25,105,42,156,4.9),(5,7,7,35,106,42,119,5.5),(6,8,8,50,92,48,90,5.9),(7,7,7,35,82,55,161,4.8),
    (8,7,5,40,111,32,213,5.3),(10,8,3,25,111,33,234,8.1),(11,7,3,15,125,41,99,12.4),(25,6,5,25,101,70,28,14.8),(31,7,7,20,110,35,51,8.6),(32,7,5,25,114,40,62,9.8),
    (33,8,3,25,110,43,157,9.9),(34,7,6,60,109,60,41,10.6),(35,8,7,60,111,41,151,4.9),(37,7,5,40,118,47,193,5.5),(38,10,10,60,90,58,108,4.7),(39,9,9,60,105,42,257,3.2),
    (40,8,7,60,112,48,97,6.2),(49,5,5,30,107,36,134,6.6),(53,4,4,30,154,45,234,6.0)]:
    s.append(dict(table_ref=T1,site=site,unit='Lac St-Jean anorthosite',comp_paper='reversed',comp_compiled='AN',dec=D,inc=I,k=k,a95=a,n=n,n_type='samples (cores)',N_total=N,tilt='n.a. (intrusive, no correction)',
      polarity_note='paper: reversed (E dec, +inc; south paleopole)',notes=f'AF cleaning {af} mT'+('; mixed-polarity site, reversed direction isolated in some samples' if site in (8,10,11) else '')))
s.append(dict(table_ref=T1,site=22,unit='monzonite',comp_paper='reversed',comp_compiled='AN',dec=122,inc=57,k=57,a95=9.0,n=6,n_type='samples (cores)',N_total=6,vgp_lat=-11,vgp_lon=150,tilt='n.a. (intrusive, no correction)',polarity_note='paper: reversed',notes='AF 25 mT; printed virtual (south) pole 210W 11S (= 150E 11S, as printed)'))
for site,unit,N,n,af,D,I,k,a in [(9,'Lac St-Jean anorthosite',6,5,60,281,-66,255,4.8),(12,'Lac St-Jean anorthosite',7,6,60,271,-61,48,9.8),
    (3,'granite',7,5,50,321,-63,48,11.1),(4,'granite',6,6,50,284,-73,86,7.3),(14,'granite',6,4,60,318,-44,78,10.5),(15,'granite',6,6,50,312,-76,116,6.2),(17,'granite',6,6,50,268,-68,323,3.7),
    (26,'monzonite',8,3,40,305,-66,40,19.7),(29,'monzonite',6,3,40,325,-66,105,12.1),(42,'monzonite',6,6,25,298,-68,133,5.8),(45,'monzonite',7,7,25,320,-67,36,10.2),
    (36,'diabase dike',8,7,50,295,-55,69,7.3)]:
    r=dict(table_ref=T2,site=site,unit=unit,comp_paper='normal',comp_compiled='AR',dec=D,inc=I,k=k,a95=a,n=n,n_type='samples (cores)',N_total=N,tilt='n.a. (intrusive, no correction)',polarity_note='paper: normal (NW dec, -inc; north paleopole)',notes=f'AF cleaning {af} mT')
    if site==36: r.update(vgp_lat=-12,vgp_lon=158,notes=r['notes']+'; printed virtual (north) pole 202W 12S = 158E 12S')
    s.append(r)
m=[dict(table_ref=T1,name='Anorthosite, reversed component, all sites',comp_paper='reversed',comp_compiled='AN',N=20,n_type='sites',dec=109,inc=45,k=32,a95=5.9,pole_lat=-8,pole_lon=168,dp=4.7,dm=7.4,notes='printed mean (south) pole 192W 8S (=168E 8S); all cores n=117: 107/46 k28 a95 2.5'),
   dict(table_ref=T1,name='Reversed polarity (anorthosite + monzonite), all sites',comp_paper='reversed',comp_compiled='AN',N=21,n_type='sites',dec=110,inc=45,k=32,a95=5.7,pole_lat=-8,pole_lon=167,dp=4.6,dm=7.3,notes='printed mean (south) pole 193W 8S (=167E 8S); all cores n=123: 108/46 k28 a95 2.5'),
   dict(table_ref=T2,name='Anorthosite, normal component, all sites',comp_paper='normal',comp_compiled='AR',N=2,n_type='sites',dec=276,inc=-64,k=310,a95=14.2,pole_lat=-29,pole_lon=163,dp=18.0,dm=22.6,notes='printed mean (north) pole 197W 29S; all cores n=11: 275/-64 k72 a95 5.4'),
   dict(table_ref=T2,name='Granite, normal component, all sites',comp_paper='normal',comp_compiled='AR',N=5,n_type='sites',dec=305,inc=-66,k=26,a95=15.2,pole_lat=-20,pole_lon=144,dp=20.4,dm=25.0,notes='printed mean (north) pole 216W 20S; all cores n=27: 302/-68 k27 a95 5.4'),
   dict(table_ref=T2,name='Monzonite, normal component, all sites',comp_paper='normal',comp_compiled='AR',N=4,n_type='sites',dec=312,inc=-67,k=257,a95=5.7,pole_lat=-16,pole_lon=139,dp=7.9,dm=9.5,notes='printed mean (north) pole 221W 16S; all cores n=19: 312/-67 k55 a95 4.6'),
   dict(table_ref=T2,name='Normal polarity (all rock types), all sites',comp_paper='normal',comp_compiled='AR',N=12,n_type='sites',dec=301,inc=-66,k=46.2,a95=6.5,pole_lat=-19,pole_lon=147,dp=8.5,dm=10.5,notes='printed mean (north) pole 213W 19S (=147E 19S); all cores n=64: 299/-66 k32.9 a95 3.1')]
w('Buchan1983a',s,m)

# ---------------- Hargraves1974a ----------------
T='Table 1, p. 856 (AC demagnetized columns)'
rows=[(84,'Anorthosite',4,300,120.2,67.3,260.7,4.7,'AN',''),(85,'Anorthosite',4,300,142.8,43.4,459.0,4.3,'AN',''),
 (90,'Anorthosite',4,300,218.8,-4.3,43.8,14.0,'','anomalous direction (shocked anorthosite 12-14 km from structure centre); compiled label was AR'),
 (91,'Impactite',4,500,336.9,39.6,22.3,26.7,'','impactite (K-Ar ~350 Ma); NRM inconclusive; not Grenvillian; compiled label was AN'),
 (92,'Shocked anorthosite',4,500,325.8,-68.2,13.2,35.3,'AR','highly shocked anorthosite, crest of Mont des Eboulements; anomalous (paper: almost antiparallel to primary vector)'),
 (93,'Anorthosite',4,300,131.4,52.2,80.1,10.3,'AN',''),(94,'Anorthosite',4,300,113.8,65.0,23.0,19.6,'AN',''),(95,'Anorthosite',5,500,132.6,52.6,15.9,19.8,'AN',''),
 (96,'Anorthosite',4,300,109.4,23.1,61.1,11.8,'AN','~20 km from centre; interpreted as tectonically rotated primary vector'),
 (97,'Anorthosite',4,300,80.2,27.0,16.9,23.0,'AN','~20 km from centre; interpreted as tectonically rotated primary vector'),
 (98,'Anorthosite',5,'2/300',288.9,51.5,2.4,64.7,'','failed to respond to treatment (scattered)'),
 (99,'Anorthosite',4,300,209.7,-18.8,31.4,16.7,'','anomalous direction (shocked anorthosite 12-14 km from centre)'),
 (100,'Charnockite, with shatter cones',4,100,11.7,56.3,1.6,90.0,'','failed to respond to treatment (scattered)'),
 (101,'Anorthosite',4,300,112.2,-10.4,109.2,8.8,'AN','15 km from centre; interpreted as rotated primary vector; compiled label was AR')]
s=[dict(table_ref=T,site=a,unit=b,comp_paper='primary RM' if c else '',comp_compiled=i,dec=d,inc=e,k=f,a95=g,n=c,n_type='samples',tilt='n.a. (intrusive/metamorphic; no correction)',notes=f'AC demagnetization {h} Oe'+('; '+j if j else '')) for a,b,c,h,d,e,f,g,i,j in rows]
m=[dict(table_ref='Table 1, p. 856; text p. 857',name='Locality mean, sites 84, 85, 93, 94, 95 (>20 km from structure centre)',comp_paper='primary RM',comp_compiled='AN',N=5,n_type='sites',dec=127.9,inc=57.0,k=48.8,a95=11.1,pole_lat=7,pole_lon=328,dp=12,dm=16,notes='pole printed 7N 32W; error printed as alpha_lat 12, alpha_long 16 (text: dp 12, dm 16)')]
w('Hargraves1974a',s,m)

# ---------------- Robertson1979a ----------------
T='Table 1, pp. 1844-1845 (Combined columns)'
rows=[('1',6,7,134,49,6.95,138,5,2,148,'AN','St-Urbain anorthosite; = site 93 of Hargraves & Roy (1974)'),
 ('2',8,9,204,72,8.98,402,3,'','','','St-Urbain anorthosite; in situ position doubtful; no VGP; excluded'),
 ('3',9,13,98,8,12.88,101,4,2,191,'AN','St-Urbain anorthosite (inner margin of peripheral trough; possibly tilted); included only in some averages'),
 ('4',8,9,342,80,8.87,66,6,'','','','St-Urbain anorthosite; in situ position doubtful; no VGP; excluded'),
 ('8',11,13,107,39,12.83,71,5,-6,173,'AN','St-Urbain anorthosite; NRM scattered (dual polarity)'),
 ('9',5,6,133,65,5.95,91,7,-13,141,'AN','St-Urbain anorthosite'),
 ('10',8,9,127,39,8.67,24,11,6,158,'AN','St-Urbain anorthosite'),
 ('11',15,15,295,-62,14.24,18,9,-17,153,'AR','shocked anorthosite of central uplift; reversed polarity remanence (paper); = site 92 of Hargraves & Roy (1974)'),
 ('13',7,5,120,49,4.84,24,16,-4,158,'AN','central uplift (diorite/gabbroic anorthosite); = site 90 of Hargraves & Roy (1974)'),
 ('14a',8,10,126,23,9.63,24,10,13,164,'AN','margin of St-Urbain pluton; reliability questioned by authors'),
 ('15',9,9,142,50,8.89,71,6,4,142,'AN','St-Urbain anorthosite; NRM scattered (dual polarity)'),
 ('16',8,9,116,53,8.95,175,4,-9,159,'AN','St-Urbain anorthosite')]
s=[dict(table_ref=T,site=a,unit='St-Urbain anorthosite / Charlevoix impact structure',comp_paper='normal (high-Tub hematite)' if i!='AR' else 'reverse',comp_compiled=i,dec=d,inc=e,k=g,a95=h,n=n4,n_type='specimen treatments (N4 = N2 thermal + N3 af; cores counted once per treatment)',N_total=n1,r=r,vgp_lat=vl,vgp_lon=vo,tilt='n.a.',notes=f'N1 (cores treated) = {n1}; printed pole (E, lat) = {vo}, {vl}' + ('; printed pole is antipode of the VGP of this direction (southern/Pacific convention)' if (i=='AN') else '') + '; '+j) for a,n1,n4,d,e,r,g,h,vl,vo,i,j in rows]
m=[dict(table_ref='Table 2, p. 1854',name='(3) All sites (site 3 excluded)',comp_paper='normal',comp_compiled='AN',N=9,n_type='sites',dec=124,inc=48,k=31,a95=9,pole_lat=-2,pole_lon=156,dp=8,dm=12,notes='R 8.74; pole printed (E, S) 156, 02; considered a best estimate'),
   dict(table_ref='Table 2, p. 1854',name='(7) Average 3 + Hargraves & Roy (1974) sites',comp_paper='normal',comp_compiled='AN',N=14,n_type='sites',dec=126,inc=51,k=34,a95=7,pole_lat=-3,pole_lon=153,dp=7,dm=10,notes='R 13.62; pole printed (E, S) 153, 03; = St-Urbain pole StU of Roy & Robertson (1979)')]
w('Robertson1979a',s,m)

# ---------------- Ueno1976a ----------------
T='Table I, p. 310'
rows=[('66','Tonalite suite gneiss (metamorphosed in Grenvillian orogeny), within Grenville Province','tonalite gneiss',5,'',300,129,55,43,'A'),
 ('51','Bourbeau sill','pyroxenite',4,1,'200, 300',137,45,14,'B'),('54','Roberge sill','serpentinized dunite/peridotite',5,2,'200, 300',116,72,41,'B'),
 ('70','Dore Lake Complex','gabbroic anorthosite',5,'',200,90,54,24,'B'),('72','Dore Lake Complex','anorthositic gabbro',4,1,300,117,23,144,'B')]
s=[dict(table_ref=T,site=a,unit=b,comp_paper='CH',comp_compiled='CH',dec=d,inc=e,k=k,n=n,n_type='samples (cores used)',N_total=(n+(dn or 0)) if dn!='' else n,tilt='in situ (no correction; tilt correction increases dispersion)',notes=f'lithology: {c}; a.f. {af} Oe; thermal type {t}'+(f'; {dn} core(s) discarded' if dn!='' else '')+'; a95 not printed') for a,b,c,n,dn,af,d,e,k,t in rows]
m=[dict(table_ref='Table II, p. 310',name='CH magnetization (sites 51, 54, 66, 70, 72)',comp_paper='CH',comp_compiled='CH',N=5,n_type='sites',dec=119,inc=51,k=15,pole_lat=-7,pole_lon=155,dp=19,dm=28,notes='pole printed 07S, 155E; dm, dp printed 28, 19'),
   dict(table_ref='Table II, p. 310',name='CS magnetization (sites 53, 64, 78\', 79\')',comp_paper='CS',N=4,n_type='sites',dec=186,inc=-38,k=70,pole_lat=-61,pole_lon=273,dp=8,dm=13,notes='pole printed 61S, 087W'),
   dict(table_ref='Table II, p. 310',name='CV magnetization (sites 78, 79)',comp_paper='CV',N=2,n_type='sites',dec=60,inc=-24,k=64,pole_lat=-8,pole_lon=226,dp=18,dm=34,notes='pole printed 08S, 134W')]
w('Ueno1976a',s,m)

# ---------------- Fahrig1972a ----------------
T='Table 1, p. 1290 (cleaned columns)'
rows=[(1,6,5,136,35,7.3,124,3.3),(2,6,5,139,31,7.3,119,3.3),(3,7,6,140,38,5.5,216,2.2),(5,5,5,134,27,8.2,98,3.7),(6,6,6,131,29,6.0,180,2.5),(7,4,4,134,23,3.3,593,1.7),
 (11,6,6,138,36,3.7,468,1.5),(12,7,7,146,33,8.4,92,3.2),(13,6,4,140,32,6.0,184,3.0),(29,7,4,141,26,11,50,5.7),(30,6,6,143,22,8.6,90,3.5),(32,7,5,131,38,14,33,6.3),
 (57,5,5,142,35,3.5,528,1.6),(58,5,3,131,20,4.1,392,2.4),(59,5,5,140,36,11,58,4.7),(60,5,4,138,31,7.7,112,3.8),(61,5,5,143,30,4.5,324,2.0),(62,5,4,138,30,4.1,382,2.1),
 (63,5,5,136,34,3.0,731,1.3),(64,5,5,139,28,9.3,77,4.1),(65,5,5,138,38,6.4,163,2.8)]
s=[dict(table_ref=T,site=a,unit='Michael gabbro',comp_paper='stable (reversed) magnetization',comp_compiled='AN',dec=d,inc=e,k=k,n=n,n_type='samples (cores; after rejecting cores with theta > 20 deg)',N_total=N,tilt='n.a.',notes=f'AF 350 Oe; angular standard deviation {sd} deg; standard error of mean deltaR {se} deg; alpha95 not printed') for a,N,n,d,e,sd,k,se in rows]
m=[dict(table_ref='Table 1, p. 1290',name='Accepted sites (21)',comp_paper='reversed',comp_compiled='AN',N=21,n_type='sites',dec=138,inc=31,k=161,pole_lat=10.2,pole_lon=162.5,A95=2,K=226,notes='site-mean angular SD 6.1, deltaR 1.4; mean pole from 21 site poles 162.5E 10.2N, k=226, alpha95=2, deltaR=5.4 (as printed)')]
w('Fahrig1972a',s,m)

# ---------------- Fahrig1974a ----------------
TA='Table 1, p. 22 (Mealy Mountains, northwest component, cleaned)'; TB='Table 1, p. 22 (Mealy Mountains, east component, cleaned)'; TC='Table 2, p. 28 (Shabogamo Gabbro, cleaned)'
nw=[('1',7,6,100,271,-49,8.5,90,3.5),('5',7,3,800,320,-35,13,37,7.7),('6',6,5,150,290,-10,15,28,6.9),('10',7,5,200,310,24,9.4,74,4.2),('16',7,3,350,318,-31,5.9,187,3.4),
 ('17',6,6,500,290,-2,4.9,277,2.0),('18',7,7,200,323,-15,4.0,418,1.5),('19',6,5,800,302,-38,9.3,76,4.1),('24',5,5,800,287,-37,8.5,91,3.8),('26',7,6,100,276,-34,15,28,6.2),('33',6,5,350,302,37,13,36,6.0)]
ea=[('1',7,'2*',400,32,69,13,39,9.2),("2N'",7,7,150,90,51,4.1,394,1.5),("3N'",9,9,500,84,70,5.6,206,1.9),('8',5,2,150,88,49,21,15,15),('10',7,'2*',150,116,81,12,47,8.4),
 ('11',7,7,500,92,62,4.5,330,1.7),('12',7,7,500,68,66,5.4,228,2.0),('13',6,3,500,61,68,11,54,1.4),('23',8,6,350,131,67,9.4,75,3.8),("25N'",6,4,250,104,58,8.4,97,4.2),
 ('26',7,'3*',500,112,66,14,36,7.8),('27',6,4,200,102,56,12,44,6.1),('34',6,6,500,70,67,3.4,572,1.4),('35',6,4,350,72,50,4.8,288,2.4)]
sh=[('40',6,6,350,145,26,11,59,4.3),('41',7,5,350,120,30,9.0,82,4.0),('42',5,5,800,109,23,3.2,643,1.4),('43',6,6,350,112,5,7.4,119,3.0),('44',6,5,800,98,24,15,30,6.6),
 ('45',7,7,350,77,51,6.0,184,2.3),('46',6,6,350,83,57,5.1,249,2.1),('47',7,7,500,96,38,5.6,208,2.1),('48',5,5,800,122,61,8.5,91,3.8),('49',7,3,150,78,28,26,10,15),
 ('50',7,5,250,85,7,16,27,6.9),('51',7,4,100,66,-7,25,10,13),('52',6,6,350,75,21,5.4,227,2.2)]
s=[]
for a,N,n,af,d,e,sd,k,se in nw: s.append(dict(table_ref=TA,site=a,unit='Mealy Mountain anorthosite suite',comp_paper='northwest component (normal: north pole in Pacific)',comp_compiled='AR',dec=d,inc=e,k=k,n=n,n_type='samples (cores)',N_total=N,tilt='n.a.',notes=f'AF {af} Oe; angular SD {sd}; deltaR {se}; alpha95 not printed'+('; listed with NW component in paper although inc positive (compiled file had it as AN)' if a=='33' else '')))
for a,N,n,af,d,e,sd,k,se in ea: s.append(dict(table_ref=TB,site=a,unit='Mealy Mountain anorthosite suite',comp_paper='east component (predominantly reversed)',comp_compiled='AN',dec=d,inc=e,k=k,n=str(n).strip('*'),n_type='samples (cores)',N_total=N,tilt='n.a.',
    polarity_note=("N' site: observed magnetization is normal (steeply up to W); direction printed after inversion by authors" if "N'" in a else ''),notes=f'AF {af} Oe; angular SD {sd}; deltaR {se}; alpha95 not printed'+('; n marked * (special case, both components present, see text)' if '*' in str(n) else '')))
for a,N,n,af,d,e,sd,k,se in sh: s.append(dict(table_ref=TC,site=a,unit='Shabogamo Gabbro',comp_paper='stable magnetization',comp_compiled='AN',dec=d,inc=e,k=k,n=n,n_type='samples (cores)',N_total=N,tilt='n.a.',notes=f'AF {af} Oe; angular SD {sd}; deltaR {se}; alpha95 not printed'))
m=[dict(table_ref='Table 1, p. 22',name='Mealy Mountains northwest component, all sites',comp_paper='northwest (normal)',comp_compiled='AR',N=11,n_type='sites',dec=300,inc=-19,k=7,pole_lat=8.4,pole_lon=180.7,A95=12.0,K=12,notes='angular SD 31; deltaR 9.3; mean (north) pole 179.3W 8.4N k=12 alpha95=12.0; all cores 56: 298/-18 k7'),
   dict(table_ref='Table 1, p. 22',name='Mealy Mountains east component, all sites',comp_paper='east (reversed; 3 normal sites inverted)',comp_compiled='AN',N=14,n_type='sites',dec=87,inc=65,k=35,pole_lat=-37.9,pole_lon=178.3,A95=9.1,K=17,notes='angular SD 14; deltaR 3.6; mean (south) pole 181.7W 37.9S k=17 alpha95=9.1; all cores 66: 088/64 k35'),
   dict(table_ref='Table 2, p. 28',name='Shabogamo Gabbro, all sites',comp_paper='stable',comp_compiled='AN',N=13,n_type='sites',dec=97,inc=30,k=8,pole_lat=-10.1,pole_lon=188.8,A95=12,K=10,notes='angular SD 28; deltaR 7.7; mean (south) pole 171.2W 10.1S k=10 alpha95=12; all cores 70: 98/31 k8')]
w('Fahrig1974a',s,m)

# ---------------- Park1996a (means only) ----------------
T='Table 1, p. 749'
rows=[('A','I, S',9,331,-49,8.78,36,9,-2,326,8,11,30,'AR'),('B','I',11,120,54,10.82,54,6,13,348,6,9,35,'AN'),('B','S',13,121,54,12.78,54,6,13,348,6,8,35,'AN'),
 ('C','I',33,133,26,32.17,39,4,-11,348,2,4,13,'AN'),('C','S',36,133,26,35.13,40,4,-11,348,2,4,14,'AN'),('D','I',30,351,80,29.55,65,3,73,292,6,6,71,''),('D','S',35,352,80,34.51,69,3,73,293,5,6,71,''),('E','I, S',5,352,-76,4.87,30,14,29,305,24,26,64,'')]
m=[dict(table_ref=T,name=f'Component {c} (unit weight {u})',comp_paper=c,comp_compiled=cc,N=N,n_type={'I':'intrusions','S':'sites','I, S':'intrusions = sites'}[u],dec=d,inc=i,k=k,a95=a,lat=54.5,lon=-58.5,pole_lat=pl,pole_lon=po,dp=dp,dm=dm,notes=f'R {r}; paleolatitude {lam}N; mean site locality 54.5N 058.5W (Table 1 notes)'+('; D and E are recent overprints (paper)' if c in 'DE' else '')) for c,u,N,d,i,r,k,a,pl,po,dp,dm,lam,cc in rows]
w('Park1996a',[],m)

# ---------------- Park1972a ----------------
T='Table 1, p. 763'
rows=[('1','northwest trend',44.278,-76.297,"44 16.7'N, 76 17.8'W",400,100,43,5,59,10,-11,169,170,'AN'),('2','northwest trend',44.267,-76.453,"44 16.0'N, 76 27.2'W",300,101,46,6,96,7,-11,166,162,'AN'),
 ('3','northwest trend',44.278,-76.285,"44 16.7'N, 76 17.1'W",300,104,49,6,485,3,-11,162,143,'AN'),('5','northwest trend',44.287,-76.212,"44 17.2'N, 76 12.7'W",500,120,59,6,59,9,-10,147,157,'AN'),
 ('6','northwest trend',44.295,-76.218,"44 17.7'N, 76 13.1'W",300,106,49,5,129,7,-10,162,162,'AN'),('10','northwest trend',44.277,-75.230,"44 16.6'N, 75 13.8'W (as printed)",300,93,51,9,83,6,-20,168,'298 (avg of 2 specimens)','AN'),
 ('8','northeast trend',44.288,-75.877,"44 17.3'N, 75 52.6'W",200,111,70,8,9,20,-24,141,174,'AN'),('9','northeast trend',44.318,-76.690,"44 19.1'N, 76 41.4'W",300,139,45,6,7,28,10,140,'352 (avg of 2 specimens)','AN'),
 ('12','northeast trend',44.318,-75.877,"44 19.1'N, 75 52.6'W",300,283,30,7,30,11,20,197,152,''),('15','northeast trend',44.288,-76.372,"44 17.3'N, 76 22.3'W",300,131,60,8,28,11,-6,139,233,'AN'),
 ('4*','north trend',44.278,-76.438,"44 16.7'N, 76 26.3'W",500,315,-15,9,134,5,24,154,156,'')]
s=[dict(table_ref=T,site=a,unit=f'Frontenac Axis diabase dike, {b}',comp_paper='stable (cleaned)',comp_compiled=cc,dec=d,inc=e,k=k,a95=a95,n=N,n_type='samples (independent cores, 2 specimens each)',site_lat=la,site_lon=lo,vgp_lat=vl,vgp_lon=vo,tilt='n.a. (dikes)',
   notes=f'printed location {txt}; a.f. {af} Oe; m.d.f. of IRM {mdf} Oe; printed pole '+(f'{abs(vl)}{"S" if vl<0 else "N"}, {vo}E' if vo<=180 else f'{vl}N, {360-vo}W')+('; site 10 printed longitude 75 13.8 W probably a misprint for 76 13.8 W (Fig. 1 map places site 10 next to site 6; printed VGP fits 76.23W better) - kept as printed' if a=='10' else '')+('; westerly direction; K-Ar whole rock 407+-23 and 411+-23 m.y.; compiled label was AN' if a=='12' else '')+('; average of sites 4 and 14 (N=9 cores); Franklin-like pole (~675 Ma); not in compiled file' if a=='4*' else '')) for a,b,la,lo,txt,af,d,e,N,k,a95,vl,vo,mdf,cc in rows]
m=[dict(table_ref=T,name='Northwest trend dikes, average of directions',comp_compiled='AN',N=6,n_type='sites',dec=103,inc=50,k=112,a95=6,pole_lat=-12,pole_lon=163,notes='pole printed 12S, 163E'),
   dict(table_ref=T,name='Northwest trend dikes, average of poles',comp_compiled='AN',N=6,n_type='sites',pole_lat=-12,pole_lon=162,A95=7,K=85,notes='pole printed 12S, 162E; K 85; A95 07; K-Ar whole rock 751+-75, biotite 817+-70 m.y. (site 5) = minimum age ~800 m.y.')]
w('Park1972a',s,m)

# ---------------- Irving1972a ----------------
T='Table 1, p. 345'
B=(45.683,-76.617,"Bryson 45 41'N, 076 37'W (text p. 345)"); C=(45.583,-76.667,"Cheneaux Falls 45 35'N, 076 40'W"); G=(45.617,-75.850,"Gatineau Valley 45 37'N, 075 51'W")
rows=[('B04',B,301,-63,4,33,16,'AR'),('B05',B,269,-59,4,94,10,'AR'),('B11',B,278,-69,5,47,11,'AR'),('C13',C,109,64,4,14,26,'AN'),('G01',G,266,-70,7,761,2,'AR'),('G20',G,104,59,4,11,29,'AN'),
 ('G19',G,68,69,9,9,18,'AN'),('G18 (N)',G,41,62,4,57,12,'AN'),('G18 (R)',G,267,-74,5,633,3,'AR'),('G17',G,266,-61,5,57,10,'AR'),('G27',G,70,67,3,71,15,'AN'),('G26',G,276,-61,3,17,31,'AR')]
s=[dict(table_ref=T,site=a,unit={'B':'Bryson intrusion','C':'Cheneaux Falls intrusion','G':'Gatineau Valley bodies marginal to Wakefield syenite'}[a[0]],comp_paper='normal' if cc=='AN' else 'reversed',comp_compiled=cc,dec=d,inc=e,k=k,a95=a95,n=N,n_type='samples (cores, 2 specimens each)',site_lat=L[0],site_lon=L[1],tilt='n.a.',notes=f'locality coordinate: {L[2]}; dioritic or gabbroic, foliated; AF end points 150-300 Oe'+('; compiled label was AN' if a=='G18 (R)' else '')) for a,L,d,e,N,k,a95,cc in rows]
m=[dict(table_ref=T,name='B (Bryson) average',N=4,n_type='sites (incl. B04(M) of Murthy 1971)',dec=286,inc=-64,k=99,a95=9),dict(table_ref=T,name='G (Gatineau) average',N=8,n_type='sites',dec=260,inc=-67,k=65,a95=7),
   dict(table_ref=T,name='Ottawa intrusions total',N=13,n_type='sites',dec=271,inc=-66,k=61,a95=5,pole_lat=-32,pole_lon=155,A95=8,K=25,notes='pole printed 32S, 155E; N 13, K 25, A95 08; normal sites inverted for the mean')]
w('Irving1972a',s,m)

# ---------------- Irving1974b ----------------
T3='Table 3, p. 5486 (hcf magnetization M1)'; T5='Table 5, p. 5486 (mcf magnetization M2)'
hcf=[('31',4,1750,277,-76,3.99),('32',5,1750,289,-69,4.96),('33an',5,1750,279,-77,4.88),('34',4,1000,245,-84,3.93),('35',5,2000,311,-69,4.96),('36',5,1500,229,-84,4.91),('37',5,1250,280,-76,4.99),
 ('38',5,1750,293,-70,4.97),('39',5,1500,271,-66,4.91),('40',5,1500,234,-70,4.98),('41',5,1250,315,-83,4.78),('42',5,2000,280,-68,4.98),('43',5,1250,280,-64,4.92),('44',5,1500,309,-61,4.97),
 ('45',5,1250,308,-73,4.93),('53',5,1500,213,-51,4.90),('54',5,1250,233,-81,4.88),('55',5,1250,268,-69,4.96),('56',5,1500,132,-82,4.97),('57',5,1500,298,-70,4.71),('58',5,500,142,-76,4.79),
 ('59',5,1500,212,-75,4.95),('60',5,1750,265,-76,4.96),('74',4,1750,191,-67,3.89)]
utm={'31':'623917','32':'609967','33an':'528947','34':'528928','35':'554962','36':'523895','37':'535865','38':'539941','39':'424021','40':'406050','41':'574039','42':'623115','43':'650068','44':'720009','45':'751035','53':'638914','54':'623925','55':'616938','56':'552005','57':'525031','58':'511043','59':'472055','60':'439077','74':'395063'}
mcf=[('31',6,120,46,4.88),('32',9,93,40,7.91),('33an (1)',3,110,14,2.91),('33an (2)',3,192,12,2.90),('34',2,138,41,1.90),('35',9,128,10,7.67),('36',8,113,27,6.47),('37',8,114,30,7.57),
 ('39',8,61,56,7.04),('40',2,123,53,1.98),('41',9,21,51,7.26),('43',8,190,54,7.58),('54',8,106,45,7.75),('55',3,122,37,2.75),('56',6,112,28,5.67)]
s=[dict(table_ref=T3,site=a,unit='Morin anorthosite-leucogabbro (western lobe)',comp_paper='hcf (M1)',comp_compiled='AR',dec=d,inc=e,n=N,n_type='samples (cores, usually 2 specimens each)',r=r,tilt='present horizontal (no correction)',notes=f'UTM grid ref 18-{utm[a]}; af {af} Oe; k and alpha95 not printed (only R)'+('; anorthosite samples (site 33an)' if a=='33an' else '')+('; compiled file had 198/0 (transcription error)' if a=='57' else '')) for a,N,af,d,e,r in hcf]
s+=[dict(table_ref=T5,site=a,unit='Morin anorthosite-leucogabbro (western lobe)',comp_paper='mcf (M2)',comp_compiled=('AN' if a not in ('33an (2)','43') else ''),dec=d,inc=e,n=n,n_type='specimens',r=r,tilt='present horizontal (no correction)',notes='k and alpha95 not printed (only R)'+('; omitted by authors from Table 4 analysis' if a in ('33an (2)','43') else '')) for a,n,d,e,r in mcf]
m=[dict(table_ref='Table 4, p. 5486',name='M1 hcf magnetization, mean of sites',comp_paper='hcf',comp_compiled='AR',N=24,n_type='sites',dec=266.2,inc=-76.8,k=32,a95=5.3,pole_lat=-42.2,pole_lon=140.6,dp=9.2,dm=9.9,notes='Model A pole 42.2S 140.6E, error (dm, dp) 9.9, 9.2; Model B pole 42.4S 139.3E K 11 A95 9.1'),
   dict(table_ref='Table 4, p. 5486',name='M2 mcf magnetization, mean of sites',comp_paper='mcf',comp_compiled='AN',N=13,n_type='sites',dec=114,inc=38,k=17,a95=10,pole_lat=0,pole_lon=164,dp=9,dm=12,notes='Model A pole 0, 164E, error (dm, dp) 12, 9')]
w('Irving1974b',s,m)
print('done')
