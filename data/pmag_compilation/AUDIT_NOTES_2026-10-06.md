# Compilation review history — 2026-10-06

This preserves the complete earlier gap/audit log. It includes compiled source discrepancies, missing published fields, unverified existing values and deliberate exclusions; these are not all uncompiled studies. Current outstanding compilation work is in TO_CHECK.md. No interpretive choices or legacy notebook inputs were changed in this reconciliation.

The unit means/poles previously under Possible additions already exist in canonical MagIC locations.txt. Unpublished Lac-Saint-Jean data are not recoverable published table omissions. Legacy CSV label/filter decisions and primary-source quality warnings below remain unresolved but are separated from the uncompiled-data queue.

## Earlier log (preserved)

# Items needing attention (pmag compilation audit, October 2026)

## A. Could not be verified
1. **Park1983a.csv** (Park & Emslie 1983, *CJES* 20:1818-1833, doi:10.1139/e83-173). Not one of the 53 rows has been verified.
   - No PDF of the paper is in `~/Github/literature`, Zotero has only a link to it, and the publisher's copy is paywalled.
   - The `Park1983a/` MagIC folder is a straight conversion, flagged UNVERIFIED in every row.
   - Partial consistency check against the GPMDB-derived MagIC contribution 12307 ("Mealy dykes"):
     - Rows 43-52 average 275.0/-47.7 (k 13.5, a95 13.6). This matches "Comp. A" at 275/-47.7 (k 14, a95 13.6, N=10).
     - Rows 22-42 average 97.0/52.1 (k 8.7). GPMDB gives 95.2/52, k 46, N=20, so k disagrees.
     - Rows 0-21 average 325/-80 (k 7.5). GPMDB "Comp. B" is 304.9/-75.6 (N=18).
   - The coordinates 53.7, -59 are unverified.
   - Action: obtain the paper and check Tables 1-3.
2. **Coordinates that are not printed in the papers**, kept as study-area approximations and flagged in the MagIC `description`:
   - Fahrig1972a (54.5, -59).
   - Fahrig1974a Shabogamo (53.5, -65); the Mealy sites now use Halls (2015) MM, 53.1, -60.7.
   - Ueno1976a (49.5-50, -74).
   - Hargraves1974a and Robertson1979a (47.5, -70.33).
   - Buchan1978a (45, -78).
   - Stupavsky1982a: 46.25, -80.75 is a map corner tick; the sites lie at about 46.0-46.2N, 80.5-80.7W.
   - Brown2012a, Brett2008a, Alvarez1998a, Buchan1973a, Buchan1976a, Buchan1983a, Dunlop1985a, Dunlop1985b and Irving1974b use Halls (2015) DR2 locality values.
   - Ueno1975a and Irving1974b print UTM grid references that were not converted:
     - Ueno's references have zone "18T", probably a misprint for 17T.
     - Irving's references lack the 100-km square IDs.
     - Converting these would give per-site coordinates.
3. **Park1972a site 10.** The longitude is printed as 75°13.8'W. The map and the printed VGP indicate 76°13.8'W. It is kept as printed.

## B. Decisions for the user (not changed, because notebook row selection depends on them)
1. **Buchan1976a.csv.**
   - Rows 26-35 are component C (Table 3), not AN.
   - Row 14 (G4, 293.2/-67.4) is component A (`AR`), not B.
   - Notebook cell 51 therefore mixes C into the Haliburton "A" mean.
   - Buchan1973a and Buchan1976a overlap; concatenating them double-counts data.
2. **Palmer1979a.csv.**
   - The U-sample labels are shifted (rows 2-7, 11-14). See the CHANGELOG and `Palmer1979a/` for the correct labels.
   - Rows 2, 5, 10, 21, 26 and 27 are unmarked or superseded treatments, not used by the authors.
   - The notebook filter `'U12|U13'` was written against the shifted labels.
3. **Palmer1973a.csv row 2.** `Wp2` is Wilberforce site 2 but is excluded by `startswith('WP')`.
4. **Hargraves1967a.csv rows 0-1.**
   - Their a95, k and n belong to single rock-type rows (Table IV); the locality means have no published statistics.
   - Cell 112 filters on `dir_alpha95 <= 15`.
5. **Compiler-derived statistics** (see CHANGELOG section 2.5): Ueno1975a k and a95, Irving1974b k and a95, Fahrig1972a and Fahrig1974a a95, Ueno1976a a95. Decide whether to keep, recompute exactly from R, or blank them.
6. **Interpretive labels.**
   - Murthy1976a Steel Mountain `AN` rows are a Paleozoic (~451 Ma) overprint according to the authors.
   - Brett2008a uses '?' labels.
   - Seguin1984a `AN` (SE component) is a ~950 Ma overprint that fails the baked-contact test.
   - Stupavsky1982a rows are four alternative estimates of one component, not four sites.
   - Hyodo1993a has 30 samples from one site (GD8), with anisotropy-corrected directions.

## C. Errors inside the source papers (kept as printed)
- **Brown & McEnroe 2012, Table 1:**
  - ADH12 prints the same direction as ADH34 but a different k, a95 and VGP; its printed VGP does not fit its direction.
  - ADH10's printed VGP does not fit its direction (off by 33°).
- **Buchan 1978:**
  - The T2 B site row duplicates sample 20 of site T5.
  - The abstract gives the A1 pole latitude as 38.0S; Table 1 gives 28.0S.
- **Buchan & Dunlop 1976:** several k/a95 pairs are internally inconsistent: B2 sample 149, samples 24 and 113, and site B1.
- **Murthy & Rao 1976, p. 82:** the site-1 VGP hemisphere ("N") is wrong; the VGP is in the southern hemisphere.
- **Du Bois 1962:** the printed mean, 286.5/-44.5, differs by about 3° in declination from the Fisher mean of the 16 tabulated directions.
- **Halls 2015 DR1:** the BI k/a95 values (141/5.1) equal those of site 37 (the dyke) in Halls et al. 2015 (PR), not site 37H2. The DR4 italics for TH and LA look swapped.
- **Seguin & Brun 1984:**
  - The Fig. 2 graticule is misprinted.
  - The marble W- direction differs between Table I (292/-17) and Table II (272/-43).
  - Table I has an unexplained line "3(6) 70 27".

## D. Possible additions (not done)
- **Unit means and poles that are absent from the original csv files** are now in the MagIC `locations.txt` files, for example:
  - Brett2008a terrane means (Table 2).
  - Brown2012a unit means (Table 2).
  - Dunlop1985a/b and Palmer1979a means.
  - Symons1978a mean pole.
  - Ueno1975a WW/WY/WZ (Table 3).
- **Lac-Saint-Jean:** no paleomagnetic study besides Buchan et al. (1983) was found. Fahrig et al. (1974, p. 28) mention unpublished LSJ data.

## Additional structural review findings

Park1983a has only sites.txt and no locations.txt. In 24 other Grenville folders, locations rows use result-specific names while sites rows use unit names, producing MagIC join mismatches; this is a naming/integration issue, not absent numerical mean results. No referenced sites listed in locations.sites are missing. Preserve result_name while reconciling location keys in a later integration correction. All parsed CSV/MagIC rows have consistent widths and physical numeric ranges.
