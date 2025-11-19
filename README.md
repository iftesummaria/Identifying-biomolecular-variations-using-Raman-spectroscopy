# Identifying-biomolecular-variations-using-Raman-spectroscopy
Identifying biomolecular variations among Caucasian and African American prostate cancers through Raman spectroscopy and MCR-ALS Analysis

Raman Spectral Unmixing Pipeline (ICA-PLS + MCR-ALS)
This repository contains the data-processing pipeline used in our study on prostate cancer Raman spectroscopy. It includes ICA-PLS paraffin removal, wavelet denoising, MCR-ALS spectral unmixing, and optional Random Forest classification.

Workflow Overview
  ICA-PLS Wax Removal
    Removes paraffin contributions from FFPE Raman spectra.
  Wavelet Denoising
    Reduces noise while preserving key Raman features.
  MCR-ALS Spectral Unmixing
    Extracts pure biochemical components (lipid, protein, collagen, nucleic acid) and their concentration profiles.
  Random Forest Classification
    Distinguishes control vs. cancer samples and compares African American vs. Caucasian tissues.

Usage
  Run wax_removal_icapls.ipynb to correct raw Raman spectra.
  Run spectral_unmixing_mcrals.ipynb to perform MCR-ALS and visualize component scores.
  Run " " to distinguish among groups using Random Forest classification.
