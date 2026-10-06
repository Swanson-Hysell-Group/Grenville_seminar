# Audit and extension of the Grenville thermochronology compilation (October 2026)

Done 2026-10-06 by Claude Code at the request of Y. Zhang. Nothing was committed.

- **Cell-level record:** every edited cell (file, 0-based row, Sample_No, column, old → new,
  evidence) is in `CHANGELOG_audit_2026-10_cells.csv`. Original values are kept there.
- **Original files:** git HEAD also holds the originals.
- **No rows were deleted.**
- **Open items:** see `TO_CHECK.md`.

## 0. Uncertainty convention adopted

The compilation has no sigma column, so it must assume a single convention. The evidence points to **2σ / 95% confidence**:

- **U-Pb studies (Mezger, Corrigan, Ketchum, Timmermann, Martignole 1993/1998, Childe, Friedman):** report 2σ or 95% and were entered as published.
- **Ar/Ar studies from the CGKB/OGS-derived block:** the original compilers had already converted 1σ sources to 2σ (×2). This covers Cosca 1991 and 1992, Busch 1996a/b and Cureton 1997, and is verified for every row in the audit.
- **GSC K-Ar reports** (95%) and Martignole & Reynolds 1997 (2σ) were entered as published.

Several studies added to the compilation later were entered at their published **1σ** (or, for Morisset 2009, at 2σ/2). They are listed in Section 1. Their errors were converted to 2σ by multiplying the published 1σ by exactly 2. Every converted row carries the tag `[audit 2026-10: errors converted published 1σ to 2σ (x2)]` in `Age_Note`.

Studies whose published sigma level is unstated were **not** changed; they are listed in `TO_CHECK.md`, section B. New rows with an unstated sigma carry the tag `[audit 2026-10: sigma level ... not stated ...]` in `Age_Note`.

**Monitor-age recalculation (found, not changed).** In the CGKB/OGS block, the Ar/Ar ages of the following studies are **5–8 Ma older than published**:
- Cosca 1991, 1992, 1995; Busch 1996a/b; Cureton 1997.
- Culshaw 1991, Reynolds 1995, Haggart 1993, Martignole & Reynolds 1997.

The offset reproduces exactly a recalculation from the original MMhb-1 age (519–520.4 Ma) to 523.1 Ma. It appears to be a deliberate, undocumented recalculation by the original database compilers, so the ages were left untouched.

The recalculation was not applied everywhere:
- Culshaw 1991 rows 5 and 14 and Reynolds 1995 row 12 are unshifted.
- The later-added Ar/Ar studies (Berger, Warnock, Dahl, Heizler, Streepey, Onstott, Schneider, Morisset, Dallmeyer) are not recalculated either.

Treat inter-study Ar/Ar age differences of less than ~1% with care.

## 1. Sigma fixes (errors ×2, or ×2 to restore published 2σ)

Format: `row(0-based):Sample ±old→±new`. Each change was applied to the per-study file and mirrored to the identical rows in:
- `study_summary.csv`
- the notebook-derived files `Bancroft_ages.csv`, `Adirondack_lowlands_ages.csv`, `Adirondack_highlands_ages.csv` and `Allard_Urbain_ages.csv`

`Whitestone_ages.csv` was unaffected.

Studies **checked and found already consistent with 2σ** (no change):
- Mezger 1989, 1991a/b, 1992, 1993; Tuccillo 1992.
- Hanes 1988; Dallmeyer & Sutter 1980; Cosca 1991, 1992; Busch 1996a/b; Cureton 1997.
- Corrigan 1994; Ketchum 1998; Timmermann 1997, 2002; Smye 2018.
- Childe 1993; Friedman 1995; Martignole 1993, 1997, 1998; Schneider 2013.
- Studies_not_in_use: Stevens 1982, Martignole 1994.

### Details

- **Berger1981a.csv** (12 rows). Evidence: Berger & York 1981a Table 1 footnote d p.799: "All errors in this table are 1σ". Changes as row:Sample old→new (Error_Plus/Error_Minus):
  0:G14-4 ±8→±16; 1:G21-113 ±8→±16; 2:G16-25 ±7→±14; 3:B12-415 ±9→±18; 4:B2-149 ±7.6→±15.2; 5:B5-139 ±8→±16; 6:D1-190 ±7→±14; 7:B12-415 ±7.2→±14.4; 8:B2-149 ±7.1→±14.2; 9:G4-22 ±6.1→±12.2; 10:G16-25 ±6.9→±13.8; 11:B2-149 ±6.5→±13.
- **Berger1981b.csv** (4 rows). Evidence: Berger & York 1981b Table 1 p.268: "Errors in this table are 1σ". Changes as row:Sample old→new (Error_Plus/Error_Minus):
  0:TH15-86 ±10→±20; 1:TH2-34 ±8.1→±16.2; 2:TH15-86 ±7.8→±15.6; 3:TH15-85 ±7.7→±15.4.
- **Cosca1995a.csv** (5 rows). Evidence: Cosca et al. 1995 GSA Data Repository 9518 Table A footnote: "Errors on individual ages are one standard deviation". Changes as row:Sample old→new (Error_Plus/Error_Minus):
  0:Ban 34 ±3→±6; 1:Ban 36 ±3→±6; 2:Ban 42 ±3→±6; 3:Ban 39 ±2→±4; 4:Ban 44 ±2→±4.
- **Dahl2004a.csv** (37 rows). Evidence: Dahl et al. 2004 p.306 and Table 3 note: "Age uncertainties are at the 1σ level". Changes as row:Sample old→new (Error_Plus/Error_Minus):
  0:AL97-13b ±4→±8; 1:AL97-15a ±3→±6; 2:AL97-18a ±10→±20; 3:AL97-22b ±3→±6; 4:AL97-27c ±10→±20; 5:AL97-27d ±3→±6; 6:AL97-35 ±4→±8; 7:AL98-45 ±10→±20; 8:AL98-46 ±3→±6; 9:AL98-47b ±3→±6; 10:AL98-50 ±3→±6; 11:AL97-2a ±3→±6; 12:AL97-5b ±3→±6; 13:AL97-8 ±3→±6; 14:AL97-12a ±3→±6; 15:AL97-13a ±3→±6; 16:AL97-14a ±3→±6; 17:AL97-15a ±3→±6; 18:AL97-17c ±2→±4; 19:AL97-18a ±2→±4; 20:AL97-22b ±3→±6; 21:AL97-27c ±3→±6; 22:AL97-27d ±3→±6; 23:AL97-31 ±3→±6; 24:AL97-33 ±2→±4; 25:AL97-35 ±3→±6; 26:AL97-37 ±3→±6; 27:AL97-40a ±2→±4; 28:AL98-42 ±3→±6; 29:AL98-43 ±3→±6; 30:AL98-45 ±2→±4; 31:AL98-47b ±3→±6; 32:AL98-49 ±2→±4; 33:AL98-50 ±3→±6; 34:AL98-51 ±3→±6; 35:AL98-55 ±3→±6; 36:AL98-57 ±3→±6.
- **Heizler1998a.csv** (36 rows). Evidence: Heizler & Harrison 1998 pp.29,797-29,798: "Uncertainties for the ages are quoted at the one-sigma level" (excl. 1% J error). Changes as row:Sample old→new (Error_Plus/Error_Minus):
  0:1 ±1→±2; 1:1 ±1→±2; 2:2 ±1→±2; 3:2 ±1→±2; 4:3 ±1→±2; 5:3 ±1→±2; 6:5 ±1→±2; 7:5 ±1→±2; 8:6 ±1→±2; 9:6 ±1→±2; 10:7 ±14→±28; 11:8 ±1→±2; 12:9 ±21→±42; 13:11 ±2→±4; 14:12 ±1→±2; 15:12 ±1→±2; 16:13 ±2→±4; 17:13 ±2→±4; 18:15 ±10→±20; 19:15 ±2→±4; 20:16 ±10→±20; 21:16 ±2→±4; 22:17 ±3→±6; 23:17 ±8→±16; 24:17 ±2→±4; 25:18 ±2→±4; 26:19 ±2→±4; 27:21 ±2→±4; 28:21 ±1→±2; 29:21 ±2→±4; 30:22 ±2→±4; 31:23 ±1→±2; 32:24 ±1→±2; 33:25 ±1→±2; 34:26 ±1→±2; 35:44 ±2→±4.
- **Lopez-Martinez1983a.csv** (4 rows). Evidence: Lopez-Martinez & York 1983 Table 2 p.956: "Errors in this table are 1σ SE". Changes as row:Sample old→new (Error_Plus/Error_Minus):
  0:HBDE CG 121 ±13→±26; 1:HBDE CG 64 ±8→±16; 2:PLAG CG 121 ±4→±8; 3:PLAG CG 121 ±1→±2.
- **Morisset2009a.csv** (27 rows). Evidence: Morisset et al. 2009 p.100 "all errors are quoted at the 2σ level"; Table 4 (p.112) "Age (Ma) ±2σ"; compiled errors were exactly half of Table 4 values. Changes as row:Sample old→new (Error_Plus/Error_Minus):
  0:2033-D ±3.6→±7.2; 1:2006-C2 ±1.3→±2.6; 2:2042-A ±1.55→±3.1; 3:2043-A ±1.2→±2.4; 4:2020 ±0.75→±1.5; 5:2023 ±1.4→±2.8; 6:2006-G1 ±0.55→±1.1; 7:2015-B4 ±5.5→±11; 8:2030-B2 ±1.5→±3; 9:2033-D ±19.5→±39; 10:2006-B4 ±2.4→±4.8; 11:2006-C4 ±2.7→±5.4; 12:2015-A4 ±2.35→±4.7; 13:2033-A2 ±2.4→±4.8; 14:2033-D ±2.3→±4.6; 15:2036-B1B ±2.25→±4.5; 16:2015-A4 ±2.7→±5.4; 17:2033-D ±4.15→±8.3; 18:2042-A ±2.7→±5.4; 19:2102 ±3.25→±6.5; 20:2114-B ±1.5→±3; 21:2123-B ±2.85→±5.7; 22:2132 ±0.95→±1.9; 23:2104-D ±12.5→±25; 24:2109-A ±16→±32; 25:2103-B3 ±4.15→±8.3; 26:2103-B3 ±6→±12.
- **Onstott1987a.csv** (6 rows). Evidence: Onstott & Peacock 1987 Table 1 footnote p.2892: "All errors are ± 1 S.D.". Changes as row:Sample old→new (Error_Plus/Error_Minus):
  0:3-1-H-1 ±1.8→±3.6; 1:3-1-H-2 ±1.9→±3.8; 2:3-2-H ±1.6→±3.2; 3:3-3-H-1 ±1.7→±3.4; 4:3-3-H-2 ±1.6→±3.2; 5:3-1-B ±1.8→±3.6.
- **Streepey2000a.csv** (11 rows). Evidence: Streepey et al. 2000 Fig. 3 caption p.1527 "plateau ages and 1 σ errors"; GSA DR 2000100 Table DR1 column "1σ error (Ma)". Changes as row:Sample old→new (Error_Plus/Error_Minus):
  1:A112 ±4→±8; 3:A128 ±4→±8; 4:A129 ±3→±6; 5:A136 ±8→±16; 6:A142 ±1→±2; 7:CN596-56 ±2→±4; 8:HM95 ±2→±4; 9:LB596-31b ±2→±4; 10:LB93 ±2→±4; 11:PP596-60 ±1→±2; 12:SE596-49 ±1→±2.
- **Streepey2004a.csv** (16 rows). Evidence: Streepey et al. 2004 Table 2 (pp.400-405) column "1σ error (Ma)". Changes as row:Sample old→new (Error_Plus/Error_Minus):
  0:A112 ±1.39→±2.78; 1:A114 ±1.78→±3.56; 2:A117 ±1.62→±3.24; 3:A124 ±1.34→±2.68; 4:A125 ±1.93→±3.86; 5:A128 ±1.29→±2.58; 6:A129 ±2.1→±4.2; 7:A133 ±1.63→±3.26; 8:A134 ±1.34→±2.68; 9:A135 ±1.24→±2.48; 10:A136 ±1.37→±2.74; 11:A137 ±2.31→±4.62; 12:A138 ±1.5→±3; 13:A140 ±1.27→±2.54; 14:A142 ±1.44→±2.88; 15:A145 ±1.12→±2.24.
- **Warnock2000a.csv** (5 rows). Evidence: Warnock et al. 2000 p.19444: "All uncertainties are reported as 1σ"; Table 9 header "1σ, Ma". Changes as row:Sample old→new (Error_Plus/Error_Minus):
  0:94GG01a ±8→±16; 1:94GG01b ±8→±16; 2:Tamarack ±8→±16; 3:94GG01a ±7→±14; 4:94GG01b ±7→±14.


## 2. Transcription fixes

Error values here are at the published level. For studies listed in Section 1 they were then doubled: for example, Dahl AL98-45 went ±3 → ±10 (1σ) → ±20 (2σ).

- **Berger1981a.csv** row 0 (G14-4): Age: `995.8` → `995`. Evidence: Table 1 p.798 prints "995.±8"; text p.797 "995 ± 8 (1σ) Ma"
- **Berger1981a.csv** row 4 (B2-149): Age: `965` → `965.8`. Evidence: Table 1 p.798 total-gas 965.8 ± 7.6
- **Berger1981a.csv** row 9 (G4-22): Age: `905` → `905.6`. Evidence: Table 1 p.799 total-gas 905.6 ± 6.1
- **Dahl2004a.csv** row 7 (AL98-45): Error_Plus: `3` → `10`, Error_Minus: `3` → `10`. Evidence: Dahl et al. 2004 Table 3 p.307: AL98-45 hornblende 1055 ± 10 (1σ)
- **Dahl2004a.csv** row 18 (AL97-17c): Error_Plus: `3` → `2`, Error_Minus: `3` → `2`. Evidence: Dahl et al. 2004 Table 3 p.307: AL97-17c biotite 987 ± 2 (1σ)
- **Friedman1995a.csv** row 17 (1b MLt): Error_Minus: `0` → `146`. Evidence: Friedman & Martignole 1995 p.2110: sample 1b upper intercept 1667 +164/-146 Ma
- **Ketchum1998a.csv** row 21 (GC91-270): Error_Minus: `4` → `2`. Evidence: Ketchum et al. 1998 text p.35 and Fig. 5b: GC91-270 1042 +4/-2 Ma
- **Martignole1998a.csv** row 0 (1 GF zone peg): Error_Minus: `151.3` → `151`. Evidence: Martignole & Friedman 1998 p.153 / Fig. 2: 827 ± 151 Ma
- **Mezger1991a.csv** row 2 (87B): Age: `1092` → `1093`. Evidence: Mezger et al. 1991a Table 1 p.418: 87B sphene rim 207Pb/206Pb 1093 ± 4
- **Mezger1993a.csv** row 16 (90-48): Error_Plus: `2` → `3`, Error_Minus: `2` → `3`. Evidence: Mezger et al. 1993 Table 1 p.17: 90-48 calc-silicate 1050 ± 3


## 3. Metadata fixes

- **Morisset2009a.csv** (all 27 rows):
  - Province changed `ON` → `QC`.
  - Geological_Info changed from `Habre-Saint-Pierre anorthosite` to `Saint-Urbain anorthosite` (St-Urbain localities) or `Havre-Saint-Pierre anorthosite (Lac Allard lobe)` (Lac Allard localities).
  - The same Geological_Info change was made in `Allard_Urbain_ages.csv` (24 rows).
- **Mezger1991b.csv** row 18 (REN-28): Age_Material `Titanite` → `Monazite`. Source: Table 1 p.696 "*Monazite"; text says "a monazite from a metapelite was dated at 1041 ± 2 Ma". Mirrored to study_summary and Bancroft_ages where present.
- **Palmer1979a.csv** row 0: Age_Method `Ar/Ar` → `K/Ar`, Age_Technique `Ar Furnace-Step` → `Ar Furnace-Fusion`. Age_Note now explains that 911 ± 11 is a conventional K-Ar whole-rock isochron, not the age of sample SP-25 (Palmer et al. 1979, p.459, 461).
- **Lopez-Martinez1983a.csv** rows 0–3: Age_Note `Plateau age` replaced. The compiled numbers are the integrated (total-gas) ages of Table 2; the authors' plateau ages (text p.957) are now recorded in the note. The values themselves were not changed.
- **Busch1996a.csv** rows 0 (LVT130 muscovite) and 3 (RVL118 hornblende): References changed to Busch et al. 1996b (Tectonics). These separates are in Busch et al. 1996b Table 1; Busch & van der Pluijm 1996a reports biotites only.
- **Dahl2004a.csv** row 31 (AL98-47b): it carried AL98-46's coordinates and now has its own (Table 1: 44°35.124'N 75°10.803'W).
- **Heizler1998a.csv**:
  - Rows 2–5: latitude 43.2 → 43.4 (Table 1: 43°24').
  - Row 12: latitude 44.32 → 43.32 (43°19').

## 4. New studies added

All new values were read from the PDF table or page cited in brackets at the end of each row's `References` field. Every new per-study file is registered in `study_summary.csv`, which grew from 687 to 1126 rows.

Errors are converted to 2σ where the paper gives 1σ: 5 rows (Chiarenzelli 2018 ×4, McLelland 2004 BMH-01-11 ×1), each tagged in Age_Note. Where coordinates were given as UTM, they were converted with pyproj on the stated datum.

**Totals: 50 new studies, 439 new interpreted ages.**

**Lac-Saint-Jean / Saguenay**

| File | n | Material | Notes |
|---|---|---|---|
| Hebert2004a | 8 | Zircon | |
| Hebert2005a | 3 | Zircon | |
| Hervet1994a | 6 | Zircon | GSC CR 1994-F, found inside the volume PDF filed as `Dudàs1994…pdf` |
| Miloski2023a | 2 | Zircon | |
| Emslie1990a | 9 | Zircon | Morin, LSJ, Rivière-Pentecôte, Labrieville, Lac Allard, Atikonak, Mealy |

**Adirondacks**

| File | n | Material |
|---|---|---|
| McLelland1988a | 22 | Zircon, baddeleyite, monazite |
| McLelland1990a | 12 | Zircon, baddeleyite (Marcy) |
| McLelland2004a | 17 | Zircon (SHRIMP, anorthosite) |
| Hamilton2004a | 10 | Zircon |
| Aleinikoff2021a | 25 | Zircon |
| Chiarenzelli2010a | 1 | Zircon |
| Chiarenzelli2011a | 5 | Zircon, titanite |
| Chiarenzelli2017a | 10 | Zircon |
| Chiarenzelli2018a | 6 | Zircon |
| Valley2011a | 8 | Zircon |
| Heumann2006a | 9 | Zircon |
| Peck2010a | 1 | Zircon |
| Peck2013a | 3 | Zircon |
| Peck2018a | 6 | Zircon |
| Peck2025a | 12 | Zircon |
| Peck2026a | 5 | Zircon |
| Regan2019a | 4 | Zircon |
| Lupulescu2011a | 14 | Zircon |
| McLelland2011a | 2 | Zircon |
| Metzger2021a | 4 | Zircon |
| Alcock2004a | 5 | Zircon |
| Shinevar2021a | 3 | Zircon |
| Baird2011a | 4 | Zircon |
| Baird2020a | 3 | Zircon |
| Krestianinov2021a | 2 | Zircon |
| Buchanan2015a | 4 | Zircon (MSc thesis) |
| Zhang2026a | 4 | Zircon CA-ID-TIMS, titanite |

**Ontario**

| File | n | Material |
|---|---|---|
| Easton2011a | 13 | Zircon, titanite, monazite |
| Corfu2000a | 17 | Zircon, titanite, monazite, rutile, apatite |
| Krogh1994a | 25 | Zircon, titanite, monazite, rutile (GFTZ, ON/QC/NL) |
| Bussy1995a | 11 | Zircon, titanite |
| Culshaw2016a | 10 | Zircon |
| Marsh2014a | 6 | Zircon |
| Marsh2017a | 3 | Titanite, zircon |
| VanBreemen1986a | 6 | Zircon |
| Dudas1994a | 4 | Baddeleyite, monazite |
| Davidson1994a | 1 | Zircon |

**Québec / Labrador**

| File | n | Material |
|---|---|---|
| Doig1991a | 3 | Zircon (Morin AMCG) |
| Peck2022a | 18 | Zircon, titanite (Morin terrane) |
| Dallmeyer1983a | 35 | Hornblende, biotite 40Ar/39Ar (SW Labrador) |
| Dallmeyer1987a | 32 | Hornblende, biotite, muscovite 40Ar/39Ar (central Labrador) |
| Heaman2004a | 19 | Zircon, baddeleyite, titanite (Pinware) |
| Jannin2018a | 2 | Zircon |
| Kavanagh-Lepage2024a | 4 | Titanite, zircon |
| Turlin2018a | 1 | Apatite U-Pb cooling |

**Things to know about the new studies**

- **Duplicate samples:** Adirondack samples AC-85-6, -7, -8, -10, 9-23-85-7 and CGAB appear in McLelland 1988, 1990 and 2004. The 2004 SHRIMP ages are the authors' later preferred ages. Baird 2020 has one pooled row that re-uses Hamilton 2004 AM86-1.
- **Overlap:** the Krogh 1994 location-1 titanite intercepts include Haggart et al. 1993 analyses.
- **Excess argon:** Dallmeyer 1983/1987 rows judged by the author to carry excess Ar keep `Plateau Age` or `Total Gas Age` as Age_Interpretation, with a note. Only the samples the author treats as cooling ages are labelled `Cooling`.
- **Validation:** all new rows were spot-checked or text-matched against the PDFs, and the McLelland 1988 and 1990, Hervet 1994 and Emslie 1990 tables were checked visually. Remaining internal source inconsistencies are noted in the Age_Note of each affected row.
- **Scope decision:** the Llano uplift, Blue Ridge and other Appalachian inliers, the New Jersey Highlands, the Long Range inlier and west Texas (Grimes 2004) were not added. The existing compilation contains only the Grenville Province and the Adirondacks.
- **Notebook:** the new per-study files are not loaded by `code/Grenville_thermochron.ipynb`. The notebook's `closure_temp_bounds_dict` has no `Apatite` or `Baddeleyite` key, so add one before plotting those minerals.

## 5. Lac-Saint-Jean subset

Copies were written to `LacStJean/data/literature_compilation/geochron/` and `Grenville_orogen/data/literature_compilation/LacStJean/geochron/`, each with a README.

- 7 per-study files totalling 45 rows, plus a `study_summary.csv` of those rows:
  - Hebert2004a, Hebert2005a, Hervet1994a, Miloski2023a
  - Emslie1990a (2 LSJ-area samples)
  - Stevens1982a (5 Saguenay-area K-Ar rows)
  - Morisset2009a (Saint-Urbain rows)
- `secondary_cited_ages_UNVERIFIED.csv` (15 rows) holds ages known only as cited in Hébert et al. 2005 Table 1, Hébert & van Breemen 2004 Table 1, Miloski et al. 2023, or the Higgins & van Breemen 1992/1996 abstracts. Every row has `Age_Qualifier = Unverified`, and these rows are not in the main compilation.

## 6. Notebook check

`code/Grenville_thermochron.ipynb` was executed headlessly with `--allow-errors` on a scratch copy of the code and data, so the notebook's own `to_csv` cells did not overwrite repo files.

The run raises the same 5 errors with the patched data as with the original git-HEAD data, so **no new breakage** was introduced. All 5 are pre-existing:
- Cell 10: `NameError: Mezger1993a_ages` is used before it is defined (it is defined in cell 27). This cascades to `Bancroft_ages` in cells 11–12.
- Cells 24 and 43: `KeyError: 'Whole Rock'`, because `closure_temp_bounds_dict` has no `Whole Rock` entry. This affects Palmer 1979 and Reynolds 1978.

The derived CSVs regenerated by the notebook match the patched derived files.

## 7. Deferred items

See `TO_CHECK.md`. In summary:
- **Unverifiable compiled studies (no access):** Corfu & Easton 1995, Slagstad 2004a/b, Schärer 1986, Reynolds 1989, Park & Emslie 1983, Streepey 2002 (data repository), and the Ketchum 1997 abstract.
- **Compiler-derived values with no source counterpart:** Reynolds 1978, Streepey 2001, the Streepey 2004 zero errors, Smye 2018, and others.
- **Primary papers still needed:** Higgins & van Breemen 1992/1996 and Higgins et al. 2002 for Lac-Saint-Jean, plus about 40 further candidate studies.
