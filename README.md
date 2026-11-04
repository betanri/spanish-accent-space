# Spanish accents mirror geography, drift like neutral genes: code and data

Code and processed data for the paper by Ricardo Betancur-R., "Spanish accents mirror geography, drift like neutral
genes".

The study measures pronunciation in public recordings of 2,126 vetted speakers from 137 places in Spain and the
Americas. Of 82 measured features, 66 pass the checks and are used. The accent space is then tested against the map,
against a model of drift, and against borders and time. This repository holds the per-speaker measurements and every
script from those measurements to the numbers, figures and tables of the paper.

No audio is included. The recordings belong to their owners. `data/Data_S1_speakers.csv` gives a public link to the
recording and to the biography used for each speaker.

## Contents

| Folder or file | What it holds |
|---|---|
| `run_all.sh` | Runs every analysis, in order, from the measurements to the figures and tables. |
| `scripts/` | All scripts. `run_all.sh` runs the analysis scripts; the measurement scripts need the audio (see below). |
| `features/` | Measurements: one row per speaker, one column per feature. |
| `data/meta/` | Speakers (place, sex, batch, years), places (coordinates, history, elevation, contact, census data), checks. |
| `data/external/` | Surname counts for the Spanish provinces (INE). |
| `data/raw/geo/` | Country outlines (Natural Earth). |
| `data/Data_S1_speakers.csv` | Data S1: every speaker, with name, place, links and the eligibility notes. |
| `results/` | The outputs of `run_all.sh`, as used in the paper. Data S2 is `results/feature_catalog/Data_S2_features.xlsx`. |
| `environment.yml` | Software and versions. |

## Run the analyses

```bash
micromamba env create -f environment.yml
micromamba activate accents
bash run_all.sh
```

The run takes about 30 minutes on a laptop. It writes `results/` and `logs/`. A new run gives the same numbers as the
files in `results/`: every random step has a fixed seed. In some figures the place labels can land in slightly
different positions, because the step that moves labels apart is not seeded; the data in the figures do not change.

## Where each figure and table comes from

| Item | Script | Output |
|---|---|---|
| Fig. 1 to 5 | `80_ms_figures.py` | `results/figures/fig1.png` to `fig5.png` |
| Fig. S1 | `37_space3.py` | `results/c44/space_continents.png` |
| Fig. S2 | `07_geography.py` (Spain) | `results/c44_geo_spain/geography_map.png` |
| Fig. S3 | `42_drift_test.py` | `results/c44_drift/drift_fig4.png` |
| Fig. S4 | `46_alignment_figure.py` | `results/c44/alignment_all.png` |
| Fig. S5 | `47_islands.py` | `results/c44_islands/islands.png` |
| Table S1 | `86_feature_catalog.py` | `results/feature_catalog/table_s1.csv` |
| Table S2 | `09b_matched_test.py`, `08_compare_read_vs_free.py` | `results/read_vs_free_c44/` |
| Table S3 | `76_zone_control.py` | `results/c44_zone_control/summary.json` |
| Table S4 | `74_test_register.py` | `results/c44_tests/register.csv` |
| Table S5 | (data) | `data/meta/place_history.csv` |
| Table S6 | `26_centroids.py` (ranks) | `results/c44_centres/centre_distances.csv` |
| Table S7 | `77_robust_upbringing.py` | `results/c44_robust_upbringing/summary.json` |
| Table S8 | `78_robust_subsets.py` | `results/c44_robust_subsets/summary.json` |
| Table S9 | `75_contact.py`, `69_environment.py` | `results/c44_contact/indigenous_places.csv`, `results/c44_environment/indigenous_1800_places.csv` |
| Data S1 | `84_data_s1.py` | `data/Data_S1_speakers.csv` |
| Data S2 | `86_feature_catalog.py` | `results/feature_catalog/Data_S2_features.xlsx` |

The numbers in the text come from these scripts:

| Result | Script | Output |
|---|---|---|
| Checks of the measurement tools (expert tags, dialect zones) | `06_validate_osu.py`, `24_validate_zones.py`, `tools_for_c44.py` | `results/validation/`, `logs/tools_c44.log` |
| Read speech against public recordings, and the pilot | `09b_matched_test.py`, `08_compare_read_vs_free.py`, `09c_pilot.py`, `09d_pr.py` | `results/read_vs_free_c44/`, `results/openslr_shared/` |
| Accent space and classifier | `05_space.py` | `results/c44/` |
| Map fit (Procrustes, Mantel), with more axes | `07_geography.py`, `68_geo_kaxes.py` | `results/c44_geo_*/`, `results/c44_geo_kaxes/` |
| Drift test, and map tests by sound set | `42_drift_test.py`, `67_geo_by_soundset.py` | `results/c44_drift/`, `results/c44_geo_soundsets/` |
| Center of the accent space | `26_centroids.py`, `39_balanced_centre.py`, `43_mahalanobis_centre.py` | `results/c44_centres/` |
| Borders, time, diversity by country | `47_islands.py` | `results/c44_islands/` |
| Founder effect | `66_founder_drift.py` | `results/c44_founder/` |
| Elevation, indigenous share around 1800 | `69_environment.py` | `results/c44_environment/` |
| Travel cost over land and sea | `70_resistance.py` | `results/c44_resistance/` |
| Ancestry | `72_ancestry.py` | `results/c44_ancestry/` |
| Surnames (Spain) | `71_surnames_spain.py` | `results/c44_surnames/` |
| Change across generations | `73_apparent_time.py` | `results/c44_apparent_time/` |
| Contact with other languages | `75_contact.py` | `results/c44_contact/` |
| Perception (media speech survey) | `32_perception.py` | `results/perception/` |
| Robustness: upbringing, groups, strict leave-one-out, all 77 features | `77_robust_upbringing.py`, `78_robust_subsets.py`, `78_loo_strict.py`, `run_all.sh` | `results/c44_robust_*/`, `results/c44_geo_loo_strict/`, `results/c44_alltools*/` |
| All 58 tests with corrections | `74_test_register.py` | `results/c44_tests/register.csv` |

## From audio to measurements

These steps need the recordings, which are not included. They are listed so that each step can be checked or redone.
The links in Data S1 lead to the recordings. The scripts ran in batches; the order in each batch was:

| Step | Script |
|---|---|
| 1. Download the audio (YouTube; podcasts, radio and archives) | `13_get_tierC.py`, `13b_get_planb.py` |
| 2. Speech detection, the speaker's voice, transcription (Whisper) | `02_prepare_spontaneous.py` |
| 3. Pronunciation dictionary with regional variants, then forced alignment (Montreal Forced Aligner, `spanish_mfa` model) | `03_build_dict.py`, then `mfa align` |
| 4. Make sure the kept voice is the guest, not the interviewer | `18_guest_voice.py`, `19_text_guest.py` |
| 5. Drop chunks with music, another language or noise | `16_filter_chunks.py` |
| 6. Measure each speaker (Praat) | `04_features.py`, with `spanish.py` |
| 7. Recording quality | `19_audio_quality.py` |
| 8. Recording year | `20_video_dates.py` |
| 9. Biographies and the residence rule | `30_fetch_bios.py`, `31_bio_filter.py` |
| 10. Exclusions (bad audio, recordings before 2005, failed checks) | `21_exclusions.py` |

The batches were then merged into `features/c44_speakers.csv` and `data/meta/c44_speakers.csv` (frozen on 19 October
2026). The read speech comes from `00_get_openslr.sh` and `02_prepare_openslr.py`; the expert-tagged clips come from
`01_get_osu.py`.

## Notes

- **Place and speaker codes.** A place code is the country, then the region (`Colombia_Cali`, `Spain_Galicia`). A
  speaker code is the place code, then the YouTube video ID or the person's Wikidata number (`Q...`).
- **Places in the tests.** A place enters the tests if it has at least 15 speakers (84 places, 1,826 speakers).
  Places with 5 to 14 speakers are shown in the figures but not tested.
- **Features left out.** The two pause measures are left out of free speech, because speech is cut into chunks at
  pauses. Eleven features failed the dialect-zone check (`logs/tools_c44.log`) and are left out of the main analyses;
  the robustness run puts them back. A feature missing for more than 35% of speakers is not used.
- **Two early features.** `rr_tap` and `vowel_nasal` were made only by an early version of `04_features.py`, so they
  exist for the early batches only (about 60% missing). Both are discarded by the 35% rule. The current
  `04_features.py` does not make them.
- **The dialect-zone check.** `24_validate_zones.py` used the 877 speakers of the batches recorded before batch 8
  (`data/meta/zone_check_places.csv`). It was not redone after batch 8, so the set of kept features stayed fixed.
- **Small tables in place of the vetting tables.** The sampling and vetting tables (candidate lists, biography text,
  download logs) are not included. Four small tables hold what the analyses take from them:
  `data/meta/speaker_years.csv` (birth year and recording year), `data/meta/speaker_quality.csv` (recording quality),
  `data/meta/speaker_profession.csv` (profession, from the first lines of each biography) and
  `data/meta/boost8_planb_sources.csv` (the podcast, radio or archive source used for 807 speakers). For 140 of these
  807 speakers the code ends in a YouTube video ID, because they were first listed with a video that could not be
  downloaded. `84_data_s1.py` and the measurement scripts need the full tables, so they do not run from this repository.
- **Names.** Data S1 names each speaker. All are public figures, and all recordings are public.

## License

Code: MIT (`LICENSE`). Data: CC BY 4.0 (`LICENSE-DATA`), except the files listed there under their own terms.

## Citation

Cite the paper and this archive (`CITATION.cff`).
