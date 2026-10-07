# Regional summary membership reconciliation — 2026-10-07

The regional study_summary.csv was stale relative to its individual study files. Lac St-Jean primary zircon sources (Emslie, Hébert 2004/2005, Hervet, Miloski) and Stevens Saguenay K-Ar rows were present. Morisset2009a.csv (27 Saint-Urbain/Lac Allard rows) was absent. Saint-Urbain is regional context, not the Lac St-Jean massif.

Audit of top-level primary study CSVs found 102 records not matching summary sample/date/material/method identity. Added 94 existing compiled records across 11 source files, including Morisset. Original 1126 rows and their values/notes remain unchanged; summary now has 1220 rows. This repairs membership; it does not certify every old scientific interpretation or establish previously unverified source values.

Eight representations remain withheld: six Reynolds1978 rounded/compiler-derived values overlap existing results; two Smye2018 weighted means are already present under differently labelled methods and numeric interpretation fields. They require scientific/source reconciliation, not duplicate rows. Full record/action manifest: REGIONAL_SUMMARY_MEMBERSHIP_AUDIT_2026-10-07.csv. Existing source-file quality warnings remain in AUDIT_NOTES_2026-10-06.md; no individual study table was edited.

The Grenville_orogen snapshot is refreshed with exact Study_File identities for the appended records. Its notebook still de-duplicates the local Saint-Urbain subset and Stevens provenance alias. Unverified secondary-cited Lac St-Jean ages remain excluded. Studies_not_in_use was not automatically activated; its deliberate exclusions and source-verification warnings require separate review.
