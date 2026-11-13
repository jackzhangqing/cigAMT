# cigAMT data inventory and Dataset Viewer

The release contains two signal branches and an independently supplied noise library. The Viewer exposes searchable **row indexes** into the stored HDF5 banks and a small, separately identified waveform gallery. Indexes and gallery images do not add new observations or enlarge the benchmark.

## Viewer configurations

| Configuration | Train | Validation | Test | Total | Contents |
|---|---:|---:|---:|---:|---|
| `sferic_24khz` | 22,520 | 3,130 | 2,488 | 28,138 | 16,878 original labeled windows plus 11,260 frozen training augmentation entries |
| `reference_150hz` | 359,299 | 66,902 | 73,844 | 500,045 | Four-component reference windows |
| `noise_library` | 694,426 | 10,195 | 13,738 | 718,359 | Usable one-component noise templates |
| `sample_previews` | - | - | - | 32 examples | 12 sferic windows, 10 reference windows, and 10 noise examples with embedded PNG images |

The gallery split is called `examples`; each row also records the actual benchmark `split`. Examples are selected deterministically to illustrate resource types, not to estimate their prevalence or quality. Full-index configurations contain metadata, not embedded waveform arrays or per-row images.

## 24 kHz sferic branch

| Remote HDF5 path | Stored `x` shape | Original windows | Frozen augmented entries |
|---|---|---:|---:|
| `sferic_24khz/train/windows.h5` | 22,520 x 4 x 240 | 11,260 | 11,260 |
| `sferic_24khz/val/windows.h5` | 3,130 x 4 x 240 | 3,130 | 0 |
| `sferic_24khz/test/labeled/windows.h5` | 2,488 x 4 x 240 | 2,488 | 0 |

Component order is `Ex,Ey,Hx,Hy`. The stored input `x` is frozen standardized input. Label 1 denotes an annotated sferic center within the window; label 0 denotes a non-sferic center. `transform=0` identifies an original window; `transform=1` identifies its frozen training augmentation. `parent_row` addresses the parent in the same split and is -1 for originals. `augmentation_snr_db` is null when not supplied. The annotation mask `mx` in the bank is not a missing-data mask. All four labeled test records are from Nanjing; this partition does not establish cross-region performance. Full source recordings are not added by the Viewer.

## 150 Hz reference branch

Every normalized reference bank uses `data` with shape `N x 4 x 2048`; component order is `Ex,Ey,Hx,Hy`. Values retain the source's stored coordinates and are not declared calibrated SI units. The index includes the optional raw-bank path separately.

| Source | Family | Train | Validation | Test | Total | Normalized remote bank |
|---|---|---:|---:|---:|---:|---|
| `real` | `real` | 4,205 | 1,021 | 2,274 | 7,500 | `amt_150hz/clean/real/clean_normalized.h5` |
| `syn_amt_A` | `syn_discrete` | 95,931 | 16,929 | 12,540 | 125,400 | `amt_150hz/clean/syn_amt_A/clean_normalized.h5` |
| `syn_amt_B` | `syn_discrete` | 95,931 | 16,929 | 12,540 | 125,400 | `amt_150hz/clean/syn_amt_B/clean_normalized.h5` |
| `syn_mt` | `syn_mt` | 4,038 | 673 | 1,346 | 6,057 | `amt_150hz/clean/syn_mt/clean_normalized.h5` |
| `ddpm` | `generated` | 10,000 | 0 | 0 | 10,000 | `amt_150hz/clean/ddpm/clean_normalized.h5` |
| `ddim` | `generated` | 10,000 | 0 | 0 | 10,000 | `amt_150hz/clean/ddim/clean_normalized.h5` |
| `syn_continuous_A` | `syn_continuous` | 67,925 | 15,675 | 20,900 | 104,500 | `amt_150hz/clean/syn_continuous_A/clean_normalized.h5` |
| `syn_continuous_B` | `syn_continuous` | 67,925 | 15,675 | 20,900 | 104,500 | `amt_150hz/clean/syn_continuous_B/clean_normalized.h5` |
| `syn_3d_A` | `syn_3d` | 1,672 | 0 | 1,672 | 3,344 | `amt_150hz/clean/syn_3d_A/clean_normalized.h5` |
| `syn_3d_B` | `syn_3d` | 1,672 | 0 | 1,672 | 3,344 | `amt_150hz/clean/syn_3d_B/clean_normalized.h5` |

The reference total comprises 7,500 curated field windows, 472,545 physics-based windows, and 20,000 diffusion-generated windows. Diffusion-generated entries are restricted to benchmark training. An index identifies the benchmark assignment; it does not establish that all generator fitting sources were isolated.

## Independent noise library

The remote bank is `amt_150hz/noise/noise_bank.h5`, dataset key `samples`, stored shape `718359 x 2048`. The full index records `support_length`, `crop_start`, and `storage_length`; these fields should be retained when interpreting active segments. The 150 Hz noise rate is an index convention, not an inferred measured sampling rate for a generator.

| Origin | Usable templates |
|---|---:|
| Field-extracted | 8,362 |
| Procedural synthesis | 59,997 |
| Diffusion-generated (training only) | 650,000 |
| **Total** | **718,359** |

| Noise class | Count |
|---|---:|
| Pulse | 71,374 |
| Quasi-sinusoidal | 73,319 |
| Square wave | 72,110 |
| Step wave | 72,760 |
| Triangular wave | 73,796 |
| Power frequency | 71,000 |
| White noise | 71,000 |
| Colored noise | 71,000 |
| Harmonic | 71,000 |
| Composite | 71,000 |

Active support lengths are 100, 150, 200, 300, 400, 450, 500, 600, 750, 1000, 1500, and 2048 samples. Class, origin, and length are different marginal descriptions of the same library and must not be added together.

## Reading an index or a preview

```python
from datasets import load_dataset

# Small gallery with an Image column, appropriate for an initial preview.
gallery = load_dataset("ZhangQing123/cigAMT", "sample_previews", split="examples")
gallery[0]["preview_image"].show()

# Full searchable metadata; this does not download all signal banks.
index = load_dataset("ZhangQing123/cigAMT", "reference_150hz", split="test")
row = index[0]
print(row["data_file"], row["array_key"], row["array_row"])
```

Each `data_file` is relative to the dataset repository. For the associated waveform, download that file and address `h5[row["array_key"]][row["array_row"]]` with h5py. Reference and sferic samples are four-component arrays; a noise row is one component. Preview PNGs are embedded directly in Parquet; their metadata contains no local filesystem paths.

## Index files

Each full index uses `viewer/<configuration>/{train,validation,test}.parquet`. Gallery images and their row provenance use `viewer/sample_previews/examples.parquet`. `viewer/viewer_manifest.json` records row counts, file sizes, SHA-256 checksums, and validation results. The upload includes only these portable files and this inventory; local build caches and source audit files are excluded.

The Viewer is a browsable inventory of the released core banks. Task configurations, partition manifests, and loading/replay code remain authoritative for controlled task construction. Large materialized offline task exports, trained weights, and experiment-code releases are separate resources.

License: CC BY-SA 4.0, as confirmed by the dataset authors.
