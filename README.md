# 🔬 Raman Spectral Unmixing Pipeline  
### *ICA-PLS Wax Removal • Wavelet Denoising • MCR-ALS Decomposition*

This repository contains the code used to preprocess and analyze Raman spectra from prostate tissue samples.  
The workflow includes **ICA-PLS paraffin removal**, **wavelet denoising**, **MCR-ALS spectral unmixing**, and optional **machine-learning classification**.

---

## 🚀 Workflow Summary

### **1. ICA-PLS Wax Removal**  
- Removes paraffin contributions from FFPE Raman spectra  
- Combines Independent Component Analysis (ICA) with PLS regression  

### **2. Wavelet Denoising**  
- Reduces high-frequency noise  
- Preserves important Raman peak structures  

### **3. MCR-ALS Spectral Unmixing**  
- Extracts pure biochemical components  
- Components include: *lipid*, *protein*, *collagen*, *nucleic acid*  
- Produces normalized concentration profiles (MCR scores)

### **4. Random Forest Classification**  
- Distinguishes **normal vs cancer** tissues  
- Compares **African American vs Caucasian American** samples  
- Includes SMOTE oversampling for class balancing  

---

## 📁 Repository Contents

```text
📄 wax_removal_icapls.ipynb        → ICA-PLS + wavelet denoising
📄 spectral_unmixing_mcrals.ipynb  → MCR-ALS unmixing + score analysis
