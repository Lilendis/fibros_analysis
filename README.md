# NanoParticleAnalyzer

An interactive pipeline for segmentation and colocalization analysis of nanoparticles and marker-positive cells in fluorescence microscopy images.

NanoParticleAnalyzer processes raw microscopy data end-to-end: from LIF conversion to quantitative statistics of nanoparticle distribution, cellular localization, and colocalization with markers of interest. It is designed for researchers in biology and bioimage analysis who need a reproducible, transparent, and tunable pipeline without writing code.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Input Data](#input-data)
- [Usage](#usage)
  - [Step 1. Convert LIF to TIFF](#step-1-convert-lif-to-tiff)
  - [Step 2. Launch the interactive application](#step-2-launch-the-interactive-application)
  - [Step 3. Analyze a single sample](#step-3-analyze-a-single-sample)
  - [Step 4. Run batch analysis](#step-4-run-batch-analysis)
- [Processing Pipeline](#processing-pipeline)
- [Parameters](#parameters)
- [Output Data](#output-data)
- [Applications](#applications)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [License](#license)
- [Authors](#authors)

---

## Features

- **End-to-end pipeline** — from raw LIF files to publication-ready visualizations and statistics.
- **Nanoparticle segmentation** — adaptive thresholding with watershed-based splitting of merged clusters, including iterative refinement for dense regions.
- **Macrophage (marker-positive cell) segmentation** — based on combined nuclei and marker channels with a tunable expansion factor.
- **Vesicle classification** — four classes: regular, marker-colocalized, in cell, in macrophage.
- **Colocalization analysis** — quantitative metrics for nanoparticle localization relative to autofluorescence and nuclei, and for marker colocalization.
- **Interactive visualization** — composite images, colored overlays with scale bars, live statistics.
- **Batch processing** — one-click analysis of multiple samples with parameters from saved configurations or defaults.
- **Reproducibility** — all parameters are stored in `analysis_params.json` alongside results.

---

## Installation

NanoParticleAnalyzer runs in a Python virtual environment. Clone the repository and install dependencies:

```bash
git clone https://github.com/Lilendis/fibros_analysis.git
cd fibros_analysis
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
pip install -r requirements.txt

Project Structure
text
fibros_analysis/
├── README.md
├── requirements.txt
└── scripts/
    ├── app.py                          # Streamlit application
    ├── vesicle_processor.py            # Core processing and analysis logic
    ├── batch_process.py                # Batch processing script
    ├── convert_lif_to_tiff.py          # LIF → TIFF converter
    └── multi_lif_analyzer_legacy.py    # Helper used by the converter
Input Data
NanoParticleAnalyzer expects pre-converted TIFF files organized in the following structure:

text
preprocessed/
└── <protein_name>/
    └── <group_name>/
        └── <sample_name>/
            ├── channels.tif            # 4-channel TIFF (uint16): nuclei, autofluorescence, nanoparticles, marker
            └── metadata.json           # Sample metadata
Channel order (as stored in channels.tif)
Index	Channel	Description
0	Nuclei	Nuclear stain
1	Autofluorescence	Autofluorescence background signal
2	Nanoparticles	Channel of interest (vesicles / nanoparticles)
3	Marker	Marker of interest (e.g., macrophage marker)
metadata.json
The following fields are required:

Field	Type	Description
sample_name	string	Display name of the sample
sample_slug	string	Filesystem-safe name used for output folders
group	string	Group name (e.g., timepoint or condition)
channels	dict	Per-channel statistics (min, max, mean)
original_index	int	Position of the image in the original LIF file
source_lif	string	Path to the source LIF file
Usage
Step 1. Convert LIF to TIFF
Place your .lif file anywhere on disk and run:

bash
cd scripts
python convert_lif_to_tiff.py "/path/to/file.lif" --output "/path/to/preprocessed"
The converter:

Reads the LIF file using readlif.

Extracts all channels for each image.

Saves a 4-channel uint16 TIFF per sample.

Writes a metadata.json with channel statistics.

Control samples (IgG) are recognized by the igg_ prefix in the sample name and handled appropriately during background subtraction.

Step 2. Launch the interactive application
From the scripts/ directory:

bash
cd scripts
streamlit run app.py
The application opens at http://localhost:8501.

Step 3. Analyze a single sample
In the sidebar, select Protein, Group, and Sample.

Click 📥 Download sample.

Adjust parameters in the expanders (Segmentation, Clustering, Background subtraction, Channel brightness, Colocalization, Macrophages).

View results in the main panel — composites, segmentation overlays, and live statistics.

Click 💾 Save to store results, or 📥 Export CSV to export statistics.

Step 4. Run batch analysis
Click 🤖 Run batch analysis in the sidebar to process all samples of the currently selected Protein automatically.

Parameters are read from saved_results/<protein>/<mouse_number>/<sample>/analysis_params.json if present.

If a sample has no saved parameters, defaults are used.

Samples already processed (i.e., statistics.csv exists in the output) are skipped.

Results are saved to saved_results/auto_analysis/<protein>/<group>/<sample>/.

Processing Pipeline
The pipeline consists of five stages:

Background subtraction

Nanoparticles: pixel-wise subtraction of the autofluorescence channel scaled by a tunable factor.

Marker: pixel-wise subtraction of the IgG background, where the background level is defined as a percentile of the marker channel. IgG samples are recognized by the igg_ prefix.

Nanoparticle segmentation

Binary mask based on intensity threshold and (optionally) difference from autofluorescence.

Morphological cleaning (small object removal).

Watershed-based splitting of merged clusters using a combined distance/intensity map.

Iterative refinement for very large clusters.

Macrophage (marker-positive cell) segmentation

Seed mask: intersection of nuclei (above percentile threshold) and marker (above threshold).

Expansion of each nucleus by a tunable factor to cover the entire cell.

Watershed separation of merged macrophages.

Vesicle classification

Class 0: regular vesicle (outside cells).

Class 1: colocalized with marker (inside a cell).

Class 2: inside a cell, no marker.

Class 3: inside a macrophage.

Statistics and visualization

Composite images (nuclei + autofluorescence + vesicles + marker).

Colored overlay with class-specific colors.

Quantitative metrics for colocalization and macrophage association.

Parameters
All parameters are tunable via the sidebar. Defaults were validated on fibrotic lung tissue images.

Parameter	Default	Range	Description
min_intensity	5	0–255	Minimum vesicle intensity for segmentation
autofluorescence_diff_threshold	0	0–100	Minimum difference between vesicle and autofluorescence channels
min_vesicle_size	2	1–100	Minimum vesicle size (pixels)
max_vesicle_size	200	10–1000	Maximum vesicle size (pixels)
max_iterations	15	1–100	Maximum iterations for cluster splitting
cluster_threshold_factor	1.5	0.01–3.0	Threshold factor for marker detection in watershed
igg_percent	90	50–100	Percentile for IgG background subtraction (95 in batch mode)
autofluorescence_factor	0.5	0.0–1.0	Factor for autofluorescence subtraction from nanoparticles
vesicles_brightness	1.0	0.01–10.0	Display brightness for corrected vesicles
protein_brightness	1.7	0.01–10.0	Display brightness for corrected marker
nuclei_brightness	1.0	0.01–10.0	Display brightness for nuclei
nuclei_contrast	1.0	0.5–5.0	Gamma correction for nuclei display
protein_threshold	50	0–65500	Minimum marker intensity for colocalization
autofluorescence_percentile	50	0–100	Percentile for autofluorescence threshold (in-cell criterion)
nuclei_percentile	50	0–100	Percentile for nuclei threshold (in-cell criterion)
expansion_factor	1.58	1.0–2.5	Factor by which the nucleus radius is expanded to cover a macrophage
analyze_macrophages	True	True/False	Enable macrophage segmentation and colocalization analysis
Output Data
For each processed sample, the following files are saved:

Manual mode (saved_results/<protein>/<sample>/)
File	Description
analysis_params.json	Parameters used for the analysis
statistics.json	Full statistics dictionary
sample_metadata.json	Copy of the sample metadata
corrected_channels.tif	Corrected vesicles and marker channels (uint16)
segmentation_mask.tif	Labeled nanoparticle mask
composite_original.png	Composite before background subtraction
composite_corrected.png	Composite after background subtraction
colored_overlay.png	Composite with colored vesicle contours and scale bar
vesicles_with_contours.png	Nanoparticle channel with contours
Batch mode (saved_results/auto_analysis/<protein>/<group>/<sample>/)
File	Description
statistics.csv	Statistics for the sample
composite_original.png	Composite after background subtraction, no overlay
colored_overlay.png	Composite with macrophages and colored vesicles
Statistics fields
Field	Description
total_vesicles	Total number of nanoparticles detected
vesicles_in_cells	Number of nanoparticles inside cells
percent_in_cells	Percentage of nanoparticles inside cells
protein_colocalized	Number of nanoparticles colocalized with marker
protein_colocalization_percent	Percentage of colocalized nanoparticles
colocalization_to_cell_ratio	Ratio of colocalized to in-cell nanoparticles (%)
total_macrophages	Total number of macrophages detected
macrophages_with_vesicles	Number of macrophages containing at least one nanoparticle
macrophage_colocalization_percent	Percentage of macrophages with nanoparticles
vesicles_in_macrophages	Total number of nanoparticles inside macrophages
mean_vesicles_per_macrophage	Average number of nanoparticles per macrophage
Applications
NanoParticleAnalyzer is a general-purpose pipeline for nanoparticle analysis and has been used for:

Colocalization analysis of extracellular vesicles with macrophages in fibrotic lung tissue.

Analysis of nanoparticles in lung tissue under fibrotic conditions.

Any fluorescence microscopy dataset with a nanoparticle channel, a cellular marker, and reference channels (nuclei, autofluorescence).

The pipeline is not limited to a specific cell type or marker — the "macrophage" segmentation step in fact detects any marker-positive cell with a nucleus, and can be repurposed accordingly.

Dependencies
All dependencies are listed in requirements.txt. Core libraries:

streamlit — interactive application

numpy, scipy, scikit-image — numerical processing and segmentation

opencv-python — image operations

tifffile, imageio, Pillow — image I/O

pandas — statistics export

matplotlib — visualization utilities

readlif — LIF file reading

