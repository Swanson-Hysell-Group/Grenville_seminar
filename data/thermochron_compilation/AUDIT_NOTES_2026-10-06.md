# Compilation review history — 2026-10-06

This preserves the complete earlier gap/audit log. It includes compiled source discrepancies, missing published fields, unverified existing values and deliberate exclusions; these are not all uncompiled studies. Current outstanding compilation work is in TO_CHECK.md. No interpretive choices or legacy notebook inputs were changed in this reconciliation.

Rimsaite1982 GSC Paper81-23 and Rimsaite1985 GSC Paper85-1A are now recovered in the literature repository. Sample/mineral/age reconciliation remains incomplete; the data have not been silently corrected. Existing compiler-derived/unknown-sigma ages below remain quality warnings, not completed primary-source verification. Candidate raw-grain ranges and out-of-scope studies are deliberately excluded rather than missing preferred means.

## Earlier log (preserved)

# TO_CHECK: Grenville thermochron compilation (audit 2026-10)

These are items that could not be verified or that need a decision. Nothing listed here was changed or added; the exception is the explicitly flagged notes in Age_Note. See CHANGELOG_audit_2026-10.md for what was changed.

## A. Existing compiled studies whose source could not be verified

None of the values below was changed.

| Study file | Rows | What is missing | Where to find it |
|---|---|---|---|
| `Corfu1995a.csv` | 31 | Corfu & Easton 1995, CJES 32:959–976 (doi:10.1139/e95-081): U-Pb table and sigma statement. Rows 12 and 13 (C-93-16, C-93-15) share the identical age 1036 +9/−5, so one may be a duplicate. Sample C-93-7 has titanite ages of both 965 ± 13 and 1130 ± 4.5 (rows 4 and 19). | Institutional access (cdnsciencepub) |
| `Slagstad2004a.csv` | 2 | Slagstad et al. 2004, GSA Memoir 197:209–241 (doi:10.1130/0-8137-1197-5.209) | GSA e-books |
| `Slagstad2004b.csv` | 14 | Slagstad et al. 2004, CJES 41:1339–1365 (doi:10.1139/e04-068): SHRIMP table and sigma | cdnsciencepub |
| `Scharer1986a.csv` | 24 | Schärer, Krogh & Gower 1986, CMP 94:438–451 (doi:10.1007/BF00376337) | Springer |
| `Reynolds1989a.csv` | 13 | Reynolds 1989, CJES 26:1567–1573 (doi:10.1139/e89-133). The abstract mentions amphibole plateaus of about 1215 and 1150 Ma. Compiled row 11 (Mealy 8R, 1286 Ma "Biotite") is worth checking. | cdnsciencepub |
| `Park1983a.csv` | 2 | Park & Emslie 1983, CJES 20:1818–1833 (doi:10.1139/e83-173). Snippets suggest these are K-Ar ages, not step-heated 40Ar/39Ar, so Age_Method/Age_Technique may be wrong. | cdnsciencepub |
| `Streepey2002a.csv` | 6 | The ages and errors exist only in AGU data repository 2001JB001094. The paper states no sigma convention. Latitude/Longitude (45, −77) are placeholders. | AGU/Wiley supplementary data |
| `Ketchum1997a.csv` | 11 | The cited GAC-MAC 1997 abstract (Program with Abstracts v.22, A78) is not available. The local PDF is the 1998 Goldschmidt abstract (Min. Mag. 62A), which differs for rows 0, 1, 3, 5, 6, 7 and 9: 997 ± 4 vs 1001 ± 4; 1063 +35/−48 vs 1016 +32/−46; 1085 ± 3 vs 1088 ± 3; 1091 ± 11 and 1095 absent; ca. 1450 vs 1441 +86/−72. | GAC-MAC 1997 abstract volume |
| `Studies_not_in_use/Ketchum1994a.csv` (3 rows, also in study_summary) | 3 | Ketchum et al. 1994, Geology 22:215–218 | GSA |
| `Studies_not_in_use/Moecher1997a.csv` | 5 | Moecher et al. 1997, CJES 34:1185–1201. The abstract gives a titanite range of 1085–1035 Ma, but compiled row 1 is 1087 ± 4. | cdnsciencepub |
| `Studies_not_in_use/Miller1996a.csv` | 6 | Miller et al. 1996, GSA Bull. 108:127–140. Row 1 (Red River anorthosite, 996 +6/−5) is labelled "Igneous Crystallization", but the abstract says "metamorphosed at"; row 5 (1217, errors 0/0) needs checking. | GSA |
| `Studies_not_in_use/Grammatikopoulos1997a.csv`, `Miller1985a.csv` | 7, 6 | PhD theses (Queen's 1997; Toronto 1985) | University libraries |
| `Studies_not_in_use/Rimsalte1981a.csv` | 8 | Row 0 (RF-1 uraninite, 995 Ma) is not in Rimsaite 1981; it may come from Rimsaite 1982 (GSC Paper 81-23, 39.5 MB PDF on NRCan OSTR). Rows 4–7 come from Rimsaite 1985 (GSC Paper 85-1A, 42.6 MB PDF on NRCan OSTR). These large downloads were not made without the user's OK. | NRCan OSTR (open) |

## B. Values that do not appear in the source, or are compiler-derived

These were kept and flagged. Decide whether to keep, replace or drop them.

- **Reynolds1978a.csv** (all 6 rows). The paper reports no age with an uncertainty for these dikes. The compiled values (e.g. K2-3B 911.57 ± 26.97) reproduce exactly as unweighted mean ± 1 SD of step ages over sample-specific temperature ranges, so they are compiler-derived. The authors conclude only ~900–910 Ma (abstract) and ~910 Ma (mean total gas, p.1830). For K3-4B the paper gives an explicit plateau of "890 ± ~5 Ma over three heating steps" (p.1829), whereas the compiled value is 871.8.
- **Streepey2001a.csv** (17 rows). Table 3 (p.487) gives integer 207Pb/206Pb titanite ages with no uncertainties. The compiled decimal ages and ±2.2–11.9 errors are not in the paper, so their sigma level is unknown. Row 16 duplicates row 9 (SE98-17).
- **Streepey2000a.csv**:
  - Row 0 (A102 hornblende 990, Error_Minus 0) is not in Streepey et al. 2000. It appears only in Streepey et al. 2004 Table 1 ("from Streepey et al. 2000 and unpublished data") with no error.
  - Rows 5 and 6 (A136 937 ± 8, A142 919 ± 1): the paper calls these total-gas ages, but DR1 lists them as "plateau" and gives total-gas ages of 892 ± 4 and 900 ± 4 (1σ).
- **Streepey2004a.csv**:
  - Rows 16–29 (CCSZ hornblendes) come from Streepey 2004 Table 1 ("from Streepey et al. 2001"), which gives no errors, so the compiled Error_Minus = 0 is not a real uncertainty. Several values differ from the original Streepey 2001 Fig. 5 plateau ages (e.g. CR3-87 941 vs 948, DH98-1 998 vs 989, RWS-3C 999 vs 1009).
  - Rows 0–15 are total-gas ages from Table 2. The authors' preferred ages in Table 1 differ for A117 (967), A124 (1043), A145 (935) and A142 (953).
- **Smye2018a.csv** (2 rows). The paper reports no weighted-mean rutile age. The compiled 899.03 ± 9.44 and 927.98 ± 14.48 were computed from supplementary Tables S1/S2, which were not accessible, so the sigma is unknown.
- **Tuccillo1992a.csv** row 1 (SH88-7b monazite 1060 ± 2). Table 12 gives 1053 ± 2 and 1062 ± 2; the text gives only "~1060 Ma".
- **Timmermann1997a.csv** row 0 (HT-95-60-G1 titanite 1064 ± 2). The source gives T1 1065 ± 1, T2 1063 ± 1 and T3 1079 ± 1, so 1064 ± 2 is a compiler summary.
- **Childe1993a.csv**:
  - Row 9 (F-20 monazite 1070 ± 22) is the midpoint/half-range of 1049–1092 Ma. The paper's concordant F-20 monazite is 1072 ± 2 (2σ, p.1064).
  - Row 2 (F-8 Pb-Pb 998 ± 40) is Gariépy's unpublished isochron, with no sigma stated.
- **Friedman1995a.csv**: rows 1, 4, 7, 9, 10, 11 and 15 are midpoints of published ranges, or "ca." ages to which errors were added.
- **Martignole1998a.csv** row 9 (1600 +0/−0.2) is published only as "ca. 1.6 Ga".
- **Martignole1997a.csv**:
  - Rows 17 and 30 (GFZ 27 <1020, GFZ 1 Hbl <1080) are upper limits but were compiled as point ages ± 0.
  - Row 18 (MLT 6) is a 1010–1036 range compiled as 1028 ± 13.
  - Row 1 Sample_No should read "MLT 33".
- **Cosca1991a.csv**: 11 rows have no published uncertainty, so their errors were invented by the original compilers: rows 5, 6, 7, 8, 24, 36, 49 (K-feldspar total-gas ages) and rows 16, 30, 43, 47 (hornblende). **Cosca1992a.csv** row 18 (FA86-5, 1098 ± 6) is in the same situation.
- **Haggart1993a.csv**:
  - Rows 11–13 (K-feldspar) are not the paper's interpreted MDD model ages (CI-20 900, GF-35 960, CI-14 990 Ma).
  - Row 10 (GF-18 Kfs 935) has no age in the paper.
- **Reynolds1995a.csv** row 13 (E-GFTZ 94 Kfs 943 ± 8). The Fig. 4 label (938) appears to be a typo, because the spectrum and the text statistics (p.217) imply ~893 Ma.
- **Palmer1979a.csv**. The ±11 on the 911 Ma K-Ar whole-rock isochron has no stated sigma.
- **Lopez-Martinez1983a.csv** rows are integrated ages. The authors' plateau ages (1146 ± 12, 1056 ± 8, 447 ± 8, 446 ± 2, all 1σ) are now noted in Age_Note; consider switching to them.
- **Morisset2009a.csv**:
  - Row 21 (2123-B): Table 4 gives ±5.7 (2σ), but the abstract, text p.107 and Fig. 5i give 1057.4 ± 8.4.
  - Row 24 (2109-A): Table 4 gives 926 ± 32, but the text p.110 and Fig. 7f give 962 ± 32, so Table 4 probably transposed digits.
  - The zircon ages are igneous crystallization ages but are labelled "Metamorphic", except the 2033-D rims.
- **Schneider2013a.csv** row 6 (NC9-11 biotite 973 ± 4). Table 1 gives 973 ± 4, but the text p.52 and Fig. 4 give 972.9 ± 4.8 (2σ).
- **Culshaw1991a.csv**, **Haggart1993a.csv**: the sigma level of the plateau ages is not stated (the figures show 1σ between-step bars). The errors were left as published.
- **Cosca1991a.csv**: no sigma level is stated anywhere in the paper. The original compilers doubled the published errors, i.e. they assumed 1σ.

## C. Lac-Saint-Jean primary sources not obtained

The ages from these papers are held only as `secondary_cited_ages_UNVERIFIED.csv` in the LacStJean folders. They have not been added to the main compilation.

- **Higgins & van Breemen 1992**, CJES 29:1412–1423 (doi:10.1139/e92-113). This is the main LSJ anorthosite age, 1157 ± 3 Ma from zircon and baddeleyite. It also gives 1142 ± 3 Ma for the SW part, the Bégin and Lac Chabot megadykes, Jonquière, and Taché 1076 ± 3. The UQAC Constellation copy (eprint 4590) is access-restricted and the publisher returned 403.
- **Higgins & van Breemen 1996**, Precambrian Research 79:327–346 (doi:10.1016/0301-9268(95)00102-6). This gives Labrecque 1146 ± 3, Chicoutimi 1082 ± 3, La Baie 1067 ± 3 and St-Ambroise 1020 +4/−3 Ma, plus metamorphic ages. The UQAC copy (eprint 4611) is restricted.
- **Higgins, Ider & van Breemen 2002**, CJES 39:1093–1105 (doi:10.1139/e02-033). This covers the central LSJ: Du Bras granite 1148 ± 2, a ~1140 ± 10 pluton, and titanite from a wollastonite deposit (reported as 1163 ± 18 Ma in a search snippet). It is the only titanite age near LSJ. The UQAC copy (eprint 4589) is restricted.
- **van Breemen & Higgins 1993**, CJES 30:1453–1457 (doi:10.1139/e93-125). Havre-Saint-Pierre SW lobe, 1062 ± 4 Ma (abstract).
- **Owens et al. 1994**, Lithos 31:189–206. Labrieville massif, 1010–1008 Ma.
- **MRN Québec reports:** Gobeil et al. 2002 (De La Blache 1327 ± 16, Hulot 1434 +64/−28), Hébert et al. 1998 (Poulin-de-Courval 1068 ± 3) and Hébert et al. 2009 (Vanel 1080 ± 2). These are probably on SIGEOM / Géologie Québec (e-sigeom.mines.gouv.qc.ca).
- **No 40Ar/39Ar or titanite/rutile cooling ages from the LSJ massif itself were found.** The nearest are Morisset et al. 2009 (Saint-Urbain) and the Stevens et al. 1982 K-Ar ages. A dedicated literature or MRN search for Ar/Ar in the Saguenay region is still needed.

## D. Candidate studies identified but not accessible, or not extracted

### Adirondacks
- **AMCG intrusion ages:**
  - Chiarenzelli & McLelland 1991, J. Geol. 99:571–590. High priority: AMCG granitoid TIMS (Oswegatchie, Stark, Piseco, Rooster Hill, Diana, Hawkeye).
  - McLelland & Chiarenzelli 1989 (Dresden olivine metagabbro 1144 ± 7).
  - Grant et al. 1986 (Diana 1155 ± 4).
  - Silver 1969.
  - Basu & Premo 2001 (Diana).
- **Ottawan / Lyon Mountain:**
  - McLelland et al. 2001, Precambrian Res. 109:39–72.
  - Selleck et al. 2005, Geology 33:781.
  - Valley et al. 2009, Geology.
  - Wong et al. 2012, GSAB (zircon and monazite).
- **Lowlands:**
  - Wasteneys et al. 1999, CJES 36:967 (SHRIMP).
  - McLelland et al. 1992 and 1993 (Hyde School 1183 ± 7, Edwardsville 1164 ± 4).
  - Johnson et al. 2004, GSA Mem. 197 (Carthage-Colton / Dana Hill; possibly titanite).
- **Other:**
  - Valentino et al. 2019 (Piseco Lake shear zone).
  - Bonamici et al. 2014 (titanite).
  - Storm & Spear 2005.
  - Grey literature: McLelland et al. 2002, Orrell & McLelland 1996, Chappell et al. 2006, Aleinikoff & Walsh 2019.
- **Zhang et al. 2026 (Tectonics):**
  - The rutile (890–850 Ma) and apatite (882–851 Ma) results are single-grain date ranges only, so they were not extracted. The grain data are in the SI or on Zenodo.
  - The text and Fig. 3 disagree for grains MA1-z7 (1013.17 ± 0.65 vs 1012.98 ± 0.82) and MA5-z6 (1016.75 vs 1016.70).

### Ontario
- Corfu & Easton 1997, CJES 34:1239. High value: Sharbot Lake–Frontenac titanite, monazite and rutile.
- Gesner 1997, BSc thesis, Dalhousie (hdl 10222/79365). 40Ar/39Ar hornblende and K-feldspar from Muskoka, McClintock, Kawagama and CMBbz. The OCR text is internally inconsistent, so check against the 16 MB PDF before extracting.
- Pfister, Kontak & Marsh 2023, Precambrian Res. 395:107128. This is in the local library but not extracted, because it has several generations per sample and includes xenotime.
- Nadeau & van Breemen 1998, CJES 35:1423.
- Nadeau 1990 (PhD thesis).
- van Breemen et al. 1986 (GAC SP 31).
- Davidson & van Breemen 1988 (CMP 100:291).
- Heaman & LeCheminant 1993.
- Wodicka, Ketchum & Jamieson 2000 (Can. Mineral. 38:471; titanite and monazite).
- McEachern & van Breemen 1993.
- Baksi 1982 (Tudor gabbro hornblende).
- Cosca 1989 (PhD thesis).
- Markley et al. 2018 (EPMA monazite; out of scope).
- `Davidson1998a` in the library is a corrupt PDF.

### Québec / Labrador
- **Manicouagan / Baie-Comeau:**
  - Kavanagh-Lepage et al. 2022 and 2023 (Geosci. Frontiers 14:101496, open access; ScienceDirect blocked).
  - Jannin et al. 2018, CJES 55:406.
  - Jordan, Indares & Dunning 2006, CJES 43:1309.
  - Dunning & Indares 2010, PR 180:204.
  - Indares & Dunning 2004.
  - Indares et al. 1998, 2000.
  - Lasalle et al. 2014.
  - Labat et al. 2020.
  - Turlin et al. 2017.
- **Mauricie and western Québec:**
  - Corrigan & van Breemen 1997, CJES 34:299 (zircon and monazite, 12 samples).
  - Indares & Dunning 1997.
  - Augland et al. 2015.
  - Soucy La Roche et al. 2015 (Mékinac-Taureau).
- **Labrador U-Pb:**
  - Connelly & Heaman 1993 (includes rutile at 994 Ma).
  - Connelly et al. 1995.
  - Wasteneys et al. 1997 (titanite 939 ± 5).
  - Corrigan et al. 2000.
  - Tucker & Gower 1994.
  - Schärer & Gower 1988.
  - Gower et al. 1991.
  - James et al. 2001.
  - Krogh et al. 1996 and 2002.
  - Gobeil et al. 1999.
  - Clark & Machado 1995.
- **Labrador Ar/Ar:**
  - Dallmeyer 1982a,b.
  - van Nostrand 1988 (MSc thesis).
  - Connelly 1991 (PhD thesis).
  - Reynolds et al. 1988.

### Out of scope
These were noted but not added, following the existing compilation's scope:
- Llano uplift.
- Blue Ridge and other Appalachian inliers.
- Long Range Inlier (Heaman et al. 2004 sample CG97-301).
- New Jersey Highlands (Volkert & Rivers 2019; Gorring 2024).
- Carrizo Mountains, west Texas (Grimes 2004).
- Sept-Îles (Higgins & van Breemen 1998; ~564 Ma).

## Additional structural review findings

All compilation CSVs parse with consistent row widths and physical numeric ranges. Streepey2001a and the aggregate retain the known exact duplicate SE98-17 row; retained pending primary-source selection rather than silently changing the historical database. Some per-study results are not represented by identical scientific keys in the 1126-row study_summary. Full-row comparison also detects regional duplicate summaries and intentional text/method normalization; raw mismatch counts are not counts of missing ages. See task report and machine-readable reconciliation for candidate integration review. Do not regenerate the aggregate by blindly concatenating regional *_ages files and Studies_not_in_use.
