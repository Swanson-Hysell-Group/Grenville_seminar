# Audit of the Grenville paleomagnetic compilation (October 2026)

**Scope:** the 32 compiled csv files in `data/pmag_compilation/`, used by `code/Grenville_pmag.ipynb`, `code/Grenville_Loop.ipynb` and `code/Pmag_unblocking_temperatures.ipynb`.

**Method:**
- Every compiled row was compared with the result tables of its source publication.
- Tables were read visually from the PDFs in `~/Github/literature`, re-rendered at 300 dpi where digits were unclear, and never taken from OCR text alone.
- The transcriptions are archived in `_audit_2026-10/transcriptions/<Study>/`:
  - `sites_transcribed.csv` holds site- or sample-level rows exactly as printed, with table and page references.
  - `means_transcribed.csv` holds the published means and poles.
- A script diffed every compiled row against these transcriptions (`_audit_2026-10/scripts/compare.py`).
- Printed VGPs were recomputed from dec/inc/lat/lon to detect antipodes, swapped E/W longitudes and transcription errors.
- The scripts in `_audit_2026-10/scripts/` are archival. Their paths point to the auditor's scratch space.

**Policy for the ORIGINAL csv files** (kept in place so the notebooks keep working):
- A value was changed only when it disagrees with the publication. This covers dec, inc, k, a95, n, coordinates, VGP and published labels.
- Every change asserted the old value first, and the formatting (BOM, CRLF) was preserved byte-for-byte.
- Site names and component labels on which a notebook *selects rows* were NOT changed, so the analysis logic is unchanged. They are listed under "Flagged, not changed".
- Statistics that the compiler derived (a95 or k computed rather than printed) were left in place but flagged. The one exception is Stupavsky1982a k, which came from an invalid formula and was blanked.
- No rows were added or deleted, so the row indices used by `.drop()` in the notebook are unchanged.

## 1. Corrections made to the original compiled csv files (270 cell edits in 22 files)

The machine log with every cell is `_audit_2026-10/scripts/fixes_log.json`.

| File | Row(s) (0-based data row; site) | Column | Old -> New | Evidence |
|---|---|---|---|---|
| Brown2012a.csv | 23 (ADH34) | k | 46 -> 49 | Brown & McEnroe 2012 Table 1 p.65, site ADH34 k=49 |
| Brown2012a.csv | 25 (ADH8) | vgp_lat | -3.4 -> -33.4 | Brown & McEnroe 2012 Table 1 p.65, site ADH8 VGP lat -33.4 |
| Buchan1973a.csv | 3 (D5) | dir_k | 27 -> 28 | Buchan & Dunlop 1973 Table 1 p.29, site D5 k=28 |
| Buchan1976a.csv | 4 (B9) | dir_inc | -75.1 -> -75 | Buchan & Dunlop 1976 Table 1 p.2957, site B9 I=-75.0 |
| Buchan1983a.csv | 0 (1): 7->4; 5 (8): 7->5; 6 (10): 8->3; 7 (11): 7->3; 8 (25): 6->5; 10 (32): 7->5; 11 (33): 8->3; 12 (34): 7->6; 13 (35): 8->7; 14 (37): 7->5; 17 (40): 8->7; 21 (9): 6->5; 22 (12): 7->6; 23 (3): 7->5; 25 (14): 6->4; 28 (26): 8->3; 29 (29): 6->3; 32 (36): 8->7 | dir_n_samples | see row list | Buchan et al. 1983 Tables 1-2 pp.253-254: column n (stable cores used; sums to "All cores") instead of N (cores collected) |
| Dubois1962a.csv | 13 (W2) | dir_inc | -54.5 -> -54 | Du Bois 1962 GSC Bull. 71 Table XXI p.67, W2 I=-54.0 |
| Dubois1962a.csv | 0 (M12): 45.78->45; 1 (M13): 45.78->45; 2 (M23): 45.78->45; 3 (M24): 45.78->45; 4 (M29): 45.78->45; 5 (M30): 45.78->45; 6 (M40): 45.78->45; 7 (M41): 45.78->45; 8 (M42): 45.78->45; 9 (M43): 45.78->45; 10 (Ban1): 45.08->45; 11 (Ban2): 45.08->45; 12 (W1): 45.68->45; 13 (W2): 45.68->45; 14 (W3): 45.68->45; 15 (W4): 45.68->45 | lat | see row list | Du Bois 1962 Table XXI title: directions "corrected to 45N and 78.5W" -> VGPs must be computed at the reference point, not the locality |
| Dubois1962a.csv | 0 (M12): -78.92->-78.5; 1 (M13): -78.92->-78.5; 2 (M23): -78.92->-78.5; 3 (M24): -78.92->-78.5; 4 (M29): -78.92->-78.5; 5 (M30): -78.92->-78.5; 6 (M40): -78.92->-78.5; 7 (M41): -78.92->-78.5; 8 (M42): -78.92->-78.5; 9 (M43): -78.92->-78.5; 10 (Ban1): -78.25->-78.5; 11 (Ban2): -78.25->-78.5; 12 (W1): -75.88->-78.5; 13 (W2): -75.88->-78.5; 14 (W3): -75.88->-78.5; 15 (W4): -75.88->-78.5 | lon | see row list | idem |
| Dunlop1985a.csv | 9 (13) | dir_dec | 297 -> 297.5 | Dunlop & Stirling 1985 Table 1 p.530, site 13 A D=297.5 |
| Dunlop1985a.csv | 19 (2) | dir_alpha95 | 8 -> 8.5 | Dunlop & Stirling 1985 Table 3 p.533, site 2 B a95=8.5 |
| Fahrig1972a.csv | 0 (1): 6->5; 1 (2): 6->5; 2 (3): 7->6; 8 (13): 6->4; 9 (29): 7->4; 11 (32): 7->5; 13 (58): 5->3; 15 (60): 5->4; 17 (62): 5->4 | dir_n_samples | see row list | Fahrig & Larochelle 1972 Table 1 p.1290: n (samples used after rejecting theta>20 cores) instead of N (collected) |
| Fahrig1974a.csv | 0 (1): 7->6; 1 (5): 7->3; 2 (6): 6->5; 3 (10): 7->5; 4 (16): 7->3; 7 (19): 6->5; 9 (26): 7->6; 10 (33): 6->5; 11 (1): 7->2; 14 (8): 5->2; 15 (10): 7->2; 18 (13): 6->3; 19 (23): 8->6; 20 (25N): 6->4; 21 (26): 7->3; 22 (27): 6->4; 24 (35): 6->4; 26 (41): 7->5; 29 (44): 6->5; 34 (49): 7->3; 35 (50): 7->5; 36 (51): 7->4 | dir_n_samples | see row list | Fahrig et al. 1974 Tables 1-2 pp.22, 28: n (samples used) instead of N (cores collected) |
| Fahrig1974a.csv | 10 (33) | dir_comp_name | AN -> AR | Fahrig et al. 1974 Table 1 p.22 lists site 33 (302/+37) in the Mealy Mountains NORTHWEST component (= compiled AR), not the east component |
| Halls2015a_compilation.csv | 17 (Lac St. Jean Anorthosite reversed direction) | vgp_lon | 33 -> 327 | Buchan et al. 1983 Table 2: north pole 213W 19S -> antipode 19N 33W = 327E (W longitude had been copied as E); Halls 2015 DR3 gives 19.3N 326.1E |
| Halls2015a_compilation.csv | 18 (Lac St. Jean Anorthosite normal direction) | vgp_lon | 13 -> 347 | Buchan et al. 1983 Table 1: south pole 193W 8S -> antipode 8N 13W = 347E (W longitude had been copied as E) |
| Halls2015a_compilation.csv | 15 (Morin Anorthosite reversed direction) | dp | 9.9 -> 9.2 | Irving et al. 1974 Table 4 p.5486: Model A pole error (dm, dp) = 9.9, 9.2 (dp/dm were swapped) |
| Halls2015a_compilation.csv | 15 (Morin Anorthosite reversed direction) | dm | 9.2 -> 9.9 | idem |
| Halls2015a_compilation.csv | 16 (Morin Anorthosite normal direction) | dm | 9 -> 12 | idem |
| Halls2015a_compilation.csv | 16 (Morin Anorthosite normal direction) | dir_k | 44 -> 17 | Irving et al. 1974 Table 4 p.5486: M2 precision k=17 |
| Halls2015a_compilation.csv | 16 (Morin Anorthosite normal direction) | dp | 12 -> 9 | Irving et al. 1974 Table 4: M2 error (dm, dp) = 12, 9 (dp/dm were swapped) |
| Halls2015a_compilation.csv | 28 (Bustard Islands Gneiss) | B | 8 -> 6 | Halls 2015 Geology GSA DR2015352 Table DR1 p.3, row BI (the compiled row had copied the adjacent SL = Seal Lake row) |
| Halls2015a_compilation.csv | 28 (Bustard Islands Gneiss) | dir_dec | 123.4 -> 121.8 | Halls 2015 Geology GSA DR2015352 Table DR1 p.3, row BI (the compiled row had copied the adjacent SL = Seal Lake row) |
| Halls2015a_compilation.csv | 28 (Bustard Islands Gneiss) | dir_inc | 32.5 -> 30.3 | Halls 2015 Geology GSA DR2015352 Table DR1 p.3, row BI (the compiled row had copied the adjacent SL = Seal Lake row) |
| Halls2015a_compilation.csv | 28 (Bustard Islands Gneiss) | dir_k | 79 -> 141 | Halls 2015 Geology GSA DR2015352 Table DR1 p.3, row BI (the compiled row had copied the adjacent SL = Seal Lake row) |
| Halls2015a_compilation.csv | 28 (Bustard Islands Gneiss) | dir_alpha95 | 6.3 -> 5.1 | Halls 2015 Geology GSA DR2015352 Table DR1 p.3, row BI (the compiled row had copied the adjacent SL = Seal Lake row) |
| Halls2015a_compilation.csv | 28 (Bustard Islands Gneiss) | vgp_lat | -9 -> -9.2 | Halls 2015 Geology GSA DR2015352 Table DR1 p.3, row BI (the compiled row had copied the adjacent SL = Seal Lake row) |
| Halls2015a_compilation.csv | 28 (Bustard Islands Gneiss) | vgp_lon | 334.6 -> 336.7 | Halls 2015 Geology GSA DR2015352 Table DR1 p.3, row BI (the compiled row had copied the adjacent SL = Seal Lake row) |
| Halls2015a_compilation.csv | 28 (Bustard Islands Gneiss) | dp | 4.2 -> 3 | Halls 2015 Geology GSA DR2015352 Table DR1 p.3, row BI (the compiled row had copied the adjacent SL = Seal Lake row) |
| Halls2015a_compilation.csv | 28 (Bustard Islands Gneiss) | dm | 7.3 -> 5.6 | Halls 2015 Geology GSA DR2015352 Table DR1 p.3, row BI (the compiled row had copied the adjacent SL = Seal Lake row) |
| Halls2015a_compilation.csv | 28 (Bustard Islands Gneiss) | pole_alpha95 | 6.3 -> 5.1 | Halls 2015 Geology GSA DR2015352 Table DR1 p.3, row BI (the compiled row had copied the adjacent SL = Seal Lake row) |
| Hargraves1974a.csv | 2 (90) | dir_comp_name | AR -> (blank) | Hargraves & Roy 1974 p.857: site 90 direction anomalous (shocked, 12-14 km from centre), not a reversed Grenville A |
| Hargraves1974a.csv | 3 (91) | dir_comp_name | AN -> (blank) | Hargraves & Roy 1974 Table 1/p.858: site 91 = impactite (K-Ar ~350 m.y.), not Grenville A |
| Hargraves1974a.csv | 13 (101) | dir_comp_name | AR -> AN | Hargraves & Roy 1974 p.857: site 101 (112/-10) = primary (SE) vector tectonically rotated, like sites 96, 97 |
| Hyodo1993a.csv | all () | dir_comp_name | (column absent) -> AR | Hyodo & Dunlop 1993 Table 4 p.8010 / Fig. 12: Grenvillian host NRM (NW-up, plotted as reversed); column was missing |
| Irving1972a.csv | 8 (G18(R)) | dir_comp_name | AN -> AR | Irving et al. 1972 Table 1 p.345: G18 (R) 267/-74 is reversed (NW-up) |
| Irving1974b.csv | 19 (57) | dir_dec | 198 -> 298 | Irving et al. 1974 Table 3 p.5486: site 57 = 298, -70 (4.71) |
| Irving1974b.csv | 19 (57) | dir_inc | 0 -> -70 | idem |
| Murthy1976a.csv | 1 (2) | dir_comp_name | AR -> (blank) | Murthy & Rao 1976 p.78: Steel Mountain site 2 (143/-14.5) "entirely anomalous", excluded; not a reversed A direction |
| Palmer1973a.csv | 0 (WP_block) | dir_k | 58 -> 56.8 | Palmer & Carmichael 1973 Table 1 p.1179: block samples K=56.8 |
| Palmer1973a.csv | 5 (TG2) | dir_dec | 333.7 -> 337.7 | Palmer & Carmichael 1973 Table 2 p.1182: Tudor site 2 D=337.7 (printed VGP 24.0N 125.8E recomputes from 337.7) |
| Palmer1973a.csv | 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 | lat | 46.33 -> 44.75 (all listed rows) | Tudor gabbro: 46.33,-80.40 is the St. Charles anorthosite locality (p.1180-1181); Tudor coordinate 44 45 N 77 41 W from Halls 2015 GSA DR2 (TG); Palmer & Carmichael Fig. 3 grid 44 43-45 N, 77 35-40 W |
| Palmer1973a.csv | 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 | lon | -80.40 -> -77.683 (all listed rows) | idem |
| Palmer1979a.csv | 20 (SP12) | vgp_lon | 161 -> -161 | Palmer et al. 1979 Table 2 p.463: SP12 VGP printed 161W 36S |
| Park1972a.csv | 8 (12) | vgp_lon | 163 -> 197 | Park & Irving 1972 Table 1 p.763: dike 12 pole printed 20N, 163W (= 197E) |
| Park1972a.csv | 8 (12) | dir_comp_name | AN -> (blank) | dike 12 has a westerly direction (283/+30) with K-Ar 407-411 m.y.; not Grenville A normal |
| Park1996a.csv | 0, 1, 2, 3, 4, 5, 6, 7 | lon | -59 -> -58.5 (all listed rows) | Park & Gower 1996 Table 1 notes: mean site locality 54.5N, 058.5W |
| Park1996a.csv | 7 (8) | dir_comp_name | AR -> (blank) | Park & Gower 1996 p.749: component E is a recent overprint (reverse of PEF), not Grenville A |
| Seguin1984a.csv | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 | lat | 46.58 -> 45.583 (all listed rows) | Seguin & Brun 1984 print no coordinates (Fig. 2 graticule misprinted); Cheneaux Falls 45 35 N 76 40 W (Irving et al. 1972 p.345); Table II poles recompute at ~45.6N 76.7W |
| Seguin1984a.csv | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 | lon | -77.75 -> -76.667 (all listed rows) | idem |
| Seguin1984a.csv | 11, 12, 13, 14, 15, 16 | dir_comp_name | AR -> W- (all listed rows) | Seguin & Brun 1984 Table I: these rows are component W- (WSW, intermediate up, ~1075 Ma), not the reverse of SE (= compiled AN) |
| Stupavsky1982a.csv | 0 (1): 136->(blank); 1 (2): 162->(blank); 2 (3): 100->(blank); 3 (4): 2178->(blank) | dir_k | see row list | Stupavsky & Symons 1982 (CJES 19:819-828) Table 2 p.824 prints no k; compiled k = (140/a95)^2 (invalid) |
| Symons1978a.csv | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | lat | 44.95 -> 44.94 (all listed rows) | Symons 1978 Table 1 footnote p.958: mean site location 44.94N, 77.79W |
| Symons1978a.csv | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | lon | -77.75 -> -77.79 (all listed rows) | idem |
| Warnock2000a.csv | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 | lat | 44.95 -> 44.8875 (all listed rows) | Warnock et al. 2000 p.19,436: locality 44 53'15"N, 78 23'15"W |
| Warnock2000a.csv | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 | lon | -78.44 -> -78.3875 (all listed rows) | idem |
Also, `Hyodo1993a.csv` gained a `dir_comp_name` column (`AR` for all rows). Source: Hyodo & Dunlop (1993), Table 4 and Fig. 12, where these are the Grenvillian host NRM directions, NW-up.

### Notes on the most consequential fixes
- **Halls2015a_compilation.csv:**
  - The table comes from **Halls (2015, *Geology* 43:1051-1054, doi:10.1130/G37188.1), GSA Data Repository 2015352, Tables DR1-DR4**, not from Halls et al. (2015, *Precambrian Research*).
  - Several rows were added by the compiler from the original papers: BD_A, BD_B, CG_C, HI_B/C, MA_N, LA_N and CG_A/B.
  - The LA_R and LA_N pole longitudes had W longitudes written as E (33 should be 327; 13 should be 347).
  - The BI row was a copy of the adjacent Seal Lake row.
  - MA_R and MA_N had dp and dm swapped, and MA_N had the wrong k.
  - The notebook uses only the MMG and RV25 rows of this file, and those are correct.
- **Palmer1973a.csv:** all ten Tudor gabbro rows carried the St. Charles anorthosite coordinates (46.33, -80.40). They now use 44.75, -77.683, Halls (2015) TG; the printed Tudor VGPs reproduce at about this location. This changes the Tudor VGPs the notebook recomputes. TG2 dec was also a typo (333.7 instead of 337.7).
- **Dubois1962a.csv:** Du Bois's Table XXI directions are corrected to a common point at 45N, 78.5W. Recomputing VGPs at the locality coordinates was therefore wrong, so lat/lon is now 45, -78.5. The locality coordinates are in `Dubois1962a/specimens.txt`.
- **Coordinates replaced with the values printed in the papers:**
  - Warnock2000a: 44.8875, -78.3875.
  - Symons1978a: 44.94, -77.79.
  - Park1996a: 54.5, -58.5.
  - Seguin1984a: 45.583, -76.667. The old 46.58, -77.75 came from a misprinted map graticule; the new value is the Cheneaux Falls locality printed by Irving et al. (1972) for the same intrusion, and Seguin's Table II poles recompute at about 45.6N, 76.7W.
- **dir_n_samples held N (collected) instead of n (used)** in three files:
  - Buchan1983a: 18 rows.
  - Fahrig1972a: 9 rows.
  - Fahrig1974a: 22 rows.
- **Irving1974b site 57:** the row read 198/0 and should be 298/-70. The notebook already drops this row (index 19), probably because of the error, so it can now be reconsidered.

### Effect on notebook results
The table compares `code_output/pole_means.csv` before and after the fixes. Both versions were executed on scratch copies of the repository.

| Pole | PLat old -> new | PLon old -> new | A95 old -> new | Cause |
|---|---|---|---|---|
| Whitestone | -18.3 -> -18.3 | 148.7 -> 148.7 | 6.9 -> 6.9 | none |
| Haliburton | -35.1 -> -35.1 | 141.6 -> 141.7 | 5.2 -> 5.2 | Warnock2000a coordinates |
| Adirondack | -23.7 -> -24.6 | 142.9 -> 142.9 | 6.7 -> 6.6 | Brown2012a ADH8 vgp_lat -3.4 -> -33.4 (the notebook uses the printed VGPs for Brown2012a) |
| Allard + Urbain | -10.6 -> -10.6 | 154.0 -> 154.0 | 11.2 -> 11.2 | none |

The VGPs that the notebook recomputes for other units change where coordinates or dec/inc were corrected: Tudor gabbro (Palmer1973a), Dubois1962a, Symons1978a, Park1996a and Seguin1984a. These only feed plots, not `pole_means.csv`.

## 2. Flagged, NOT changed (decision left to the user)
These are real discrepancies with the source. Changing them would alter which rows the notebook selects, or they are interpretive. They are repeated in `TO_CHECK.md`.

1. **Buchan1976a.csv:**
   - Rows 26-35, labelled `AN`, are the paper's component **C** (Hb2; Table 3, p. 2960), not normal-polarity A.
   - Row 14 (G4, 293.2/-67.4), labelled `B`, is the **A** component (Table 1). It should be `AR`.
   - Cell 51 of the notebook (`contains('A')` / `== 'B'`) therefore puts the C samples into the Haliburton "A" mean and G4 into B.
   - Rows 15-35 are single-sample results (Tables 2-3), not site means. Repeated site names are different samples; the sample numbers are in `Buchan1976a/samples.txt`.
   - Buchan1973a re-reports the same collection, so concatenating 1973 and 1976 double-counts data.
2. **Palmer1979a.csv:**
   - The U-series site labels are shifted in rows 2-7 and 11-14 (for example, row 3 "U5" is really U4). The values themselves are correct.
   - Rows 2, 5, 10, 21, 26 and 27 are unmarked or superseded treatments that the authors did not use.
   - The notebook filter `~site.contains('U12|U13')` depends on the current labels.
   - Correct labels and `result_quality` are in `Palmer1979a/sites.txt` and `samples.txt`.
3. **Palmer1973a.csv row 2:** site `Wp2` is excluded by the notebook's `startswith('WP')`. The paper calls it Wilberforce site 2.
4. **Hargraves1967a.csv:**
   - Rows 0 and 1 (Lac Tio, Lac Allard-McRae) carry a95, k and n from one rock-type row, not from the locality mean. The paper prints no statistics for the locality means.
   - Notebook cell 112 filters on `dir_alpha95 <= 15`, so blanking these values would remove both rows from the Allard + Urbain pole.
5. **Compiler-derived statistics** (not printed in the papers):
   - Ueno1975a dir_k = (N-1)/(N-R). Its dir_alpha95 uses 140/sqrt(kN), which underestimates the exact a95 by 20-40%.
   - Irving1974b dir_k and a95 are derived from R.
   - Fahrig1972a and Fahrig1974a a95 = 140/sqrt(k*N) with N collected, not n. The papers print only the angular standard deviation and delta-R.
   - Ueno1976a a95 = 140/sqrt(kN).
   - These are omitted from the MagIC files.
6. **Count columns that are not samples:**
   - Robertson1979a dir_n_samples = N4 (specimen treatments).
   - Brett2008a N is undefined (probably specimens).
   - Warnock2000a N is undefined (probably demagnetization steps; each row is a single-specimen PCA direction).
   - Alvarez1998a n = sites (rows are regional means).
   - McWilliams1975a n = 23 sites.
   - Stupavsky1982a rows are four whole-collection estimates of component A (Table 2), not sites.
7. **Labels the authors interpret differently:**
   - Murthy1976a Steel Mountain rows 2-8 (`AN`) are interpreted as a lower-Paleozoic (~451 Ma) overprint.
   - Brett2008a site 26 is "AR?", sites 5 and 6 are "AN?", and sites 38 and 54 are "Tu?".
   - Dunlop1985a C sites 13 and 15 are excluded from the grand mean.
   - Symons1978a sites 2, 5, 10, 12, 13 and 15 are rejected (only 10 sites enter the mean).
8. **Coordinates that are only study-area approximations** (not printed in the papers), kept as they were:
   - Fahrig1972a, Fahrig1974a, Ueno1976a, Hargraves1974a and Robertson1979a.
   - Stupavsky1982a: 46.25, -80.75 is the NW corner tick of the Fig. 1 map.
   - Buchan1978a, Dunlop1985a and Dunlop1985b.
   - Where Halls (2015) DR2/DR4 lists the locality, the MagIC files use and cite that value.
9. **Errors inside the papers themselves** (kept as printed):
   - Brown2012a: the VGPs of ADH10 and ADH12 are inconsistent with their directions (ADH12 repeats the ADH34 direction).
   - Park1972a: site 10 longitude is probably 76°13.8'W, not 75°13.8'W.
   - Buchan1978a: the T2 B row duplicates sample 20.
   - Murthy1976a: the site-1 VGP hemisphere is wrong.
   - Buchan1976a: several k/a95 pairs are inconsistent.

## 3. New MagIC 3 folders (one per source study)
Each folder `data/pmag_compilation/<Study>/` contains MagIC 3 tables: the first line is `tab\t<table>` and the second line holds the column names.
- `sites.txt` holds site-level results.
- `samples.txt` and `specimens.txt` hold sample- or specimen-level results where the paper publishes only those.
- `locations.txt` holds the published unit means and poles.

**Conventions:**
- `citations` = DOI.
- `description` = the paper, table and page, the paper's own component and polarity term, the coordinate source, and notes.
- `dir_comp_name` uses the compilation labels (`AN` = SE-down Grenville A, `AR` = NW-up, plus `B`, `C` and so on). Mislabelled rows are corrected in the MagIC version; for example, Buchan1976a C rows are labelled `C`.
- `vgp_lat`/`vgp_lon` = the VGP of the listed direction. Printed antipodes are converted, and the printed value is kept in `description`. No VGP was computed where none was printed.
- `pole_lat`/`pole_lon` in `locations.txt` are as printed, in the paper's hemisphere, with longitude converted to E.
- `result_quality = b` marks results the authors rejected or did not use.
- No a95 or k was computed.
- Coordinates come from, in order of preference: the paper; Halls (2015) GSA DR Table DR2/DR4 (cited); the compiler's approximate value (flagged in `description`).

**Validation:** every folder loads with `pmagpy.contribution_builder.Contribution`. Row counts:

| Folder | Tables (rows) |
|---|---|
| Alvarez1998a | 'sites': 35, 'locations': 12 |
| Brett2008a | 'sites': 33, 'locations': 8 |
| Brown2012a | 'sites': 36, 'locations': 6 |
| Buchan1973a | 'samples': 6, 'sites': 11, 'locations': 4 |
| Buchan1976a | 'samples': 21, 'sites': 15, 'locations': 15 |
| Buchan1978a | 'samples': 18, 'sites': 21, 'locations': 6 |
| Buchan1983a | 'sites': 33, 'locations': 6 |
| Dubois1962a | 'specimens': 16, 'locations': 1 |
| Dunlop1985a | 'sites': 43, 'locations': 7 |
| Dunlop1985b | 'sites': 8, 'locations': 3 |
| Fahrig1972a | 'sites': 21, 'locations': 1 |
| Fahrig1974a | 'sites': 38, 'locations': 3 |
| Halls2015b | 'locations': 43 |
| Hargraves1967a | 'sites': 19, 'locations': 15 |
| Hargraves1974a | 'sites': 14, 'locations': 1 |
| Hyodo1993a | 'samples': 30, 'locations': 7 |
| Irving1972a | 'sites': 12, 'locations': 3 |
| Irving1974b | 'sites': 39, 'locations': 2 |
| McWilliams1975a | 'locations': 1 |
| Murthy1976a | 'sites': 14, 'locations': 3 |
| Palmer1973a | 'sites': 23, 'locations': 3 |
| Palmer1979a | 'samples': 16, 'sites': 13, 'locations': 4 |
| Park1972a | 'sites': 11, 'locations': 2 |
| Park1983a | 'sites': 53 |
| Park1996a | 'locations': 8 |
| Robertson1979a | 'sites': 12, 'locations': 2 |
| Seguin1984a | 'sites': 42, 'locations': 17 |
| Stupavsky1982a | 'locations': 37 |
| Symons1978a | 'sites': 16, 'locations': 6 |
| Ueno1975a | 'sites': 17, 'locations': 9 |
| Ueno1976a | 'sites': 5, 'locations': 3 |
| Warnock2000a | 'specimens': 23, 'locations': 6 |

Folder notes:
- `Halls2015b/` holds the Halls (2015, *Geology*) data-repository tables (DR1-DR4, 43 rows). The original csv keeps its old name, `Halls2015a_compilation.csv`.
- `Park1983a/` is **UNVERIFIED**. It is a straight conversion of the compiled csv, because no copy of Park & Emslie (1983, doi:10.1139/e83-173) was accessible. See `TO_CHECK.md`.
- `McWilliams1975a/`, `Park1996a/` and `Stupavsky1982a/` contain only `locations.txt`, because the papers publish means, not site data.

## 4. Lac-Saint-Jean deliverables
The MagIC folders `Buchan1983a/`, `Hargraves1974a/` and `Robertson1979a/` were copied, each with a README covering sources, polarity conventions and caveats, to:
- `/Users/yimingzhang/Github/LacStJean/data/literature_compilation/pmag/`
- `/Users/yimingzhang/Github/Grenville_orogen/data/literature_compilation/LacStJean/pmag/`

The `Buchan1983a/` copies hold 33 sites and 6 locations, the `Hargraves1974a/` copies 14 sites and 1 location, and the `Robertson1979a/` copies 12 sites and 2 locations. All six copies load with PmagPy.

No other published paleomagnetic study of the LSJ anorthosite or the Saguenay region was found. The search covered the local library, Zotero, the web and MagIC.

## 5. Notebook execution
Each notebook was executed with `jupyter nbconvert --execute --allow-errors`, with PYTHONPATH set to PmagPy, on scratch copies of the repository both before and after the changes. The original notebooks and their outputs were not touched.

| Notebook | Before | After |
|---|---|---|
| `Pmag_unblocking_temperatures.ipynb` | No errors | No errors |
| `Grenville_pmag.ipynb` | Errors in the same 37 cells | Same 37 cells |
| `Grenville_Loop.ipynb` | Errors in the same 5 cells | Same 5 cells |

- In `Grenville_pmag.ipynb`, all 37 errors are `AttributeError: can't set attribute 'threshold'`, raised by cartopy 0.24 `Orthographic` in the plotting cells. Every data-loading and pole-mean cell runs.
- In `Grenville_Loop.ipynb`, the 5 errors are matplotlib colorbar `ValueError`s.
- Both sets of errors come from the installed cartopy/matplotlib versions and were present before the audit; the data changes cause none of them.
- Run strictly (without `--allow-errors`), `Grenville_pmag.ipynb` stops at its first cartopy plotting cell (cell 7) before and after the audit alike. `Grenville_Loop.ipynb` then fails because `code_output/pole_means.csv` was never written; that file is not in the repository and is produced by `Grenville_pmag.ipynb`.

## 6. Deferred items
See `TO_CHECK.md`.
