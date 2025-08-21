# supporting-material-insect-biochem-2025

**Supporting data and code for QSAR modeling of _Ceratitis capitata_ attractants using machine learning techniques**

This repository contains the supplementary material for the article:

> **Converging XGBoost Machine Learning and Molecular Docking Strategies to Identify Attractants for _Ceratitis capitata_: Molecular Characterization and Database Curation of Natural Ligands for _in vitro_/_in vivo_ Tests**  
> _Archives of Insect Biochemistry and Physiology_, 2025 (submitted)

---

## ✍️ Authors

- **Edilson B. Alencar-Filho¹***   (corresponding author)
- Ramiro P. Guimarães¹  
- Vanessa Costa Santos  
- Ana B. P. Bispo²  
- Beatriz A. G. Paranhos³  
- Nathaly C. Aquino⁴  
- Ruth R. Nascimento⁴  
- Rosalvo F. Oliveira Neto⁵  

<sup>¹</sup>Graduate Program in Health and Biological Sciences, Federal University of Vale do São Francisco, Petrolina, PE, Brazil  
<sup>²</sup>Department of Pharmaceutical Sciences, Federal University of Vale do São Francisco, Petrolina, PE, Brazil  
<sup>³</sup>Laboratory of Biological Control, Embrapa Semiárido, Petrolina, PE, Brazil  
<sup>⁴</sup>Chemical Ecology Laboratory, Institute of Chemistry and Biotechnology, Federal University of Alagoas, Maceió, AL, Brazil  
<sup>⁵</sup>Department of Computer Engineering, Federal University of Vale do São Francisco, Juazeiro, BA, Brazil  

**\*** Corresponding author: edilson.alencar@univasf.edu.br

## 🧠 What `main.py` does

The `main.py` script implements a full pipeline for molecular descriptor selection and machine learning modeling, designed to assist in the identification of attractant compounds for _Ceratitis capitata_.

It includes the following steps, all executable in **Google Colab**:

- 📁 **CSV Upload and Preprocessing**  
  Handles manual upload of descriptor datasets and parses both comma and semicolon-separated files.

- 🐝 **Artificial Bee Colony (ABC) + Random Forest Feature Selection**  
  Uses a population-based metaheuristic to select informative variables based on AUC performance with cross-validation.

- 🧠 **Post-filtering using a Best-First-like strategy (BFS)**  
  Refines the subset by evaluating individual and combined variable contributions to model performance.

- 🌲 **XGBoost Classifier Training**  
  Trains a model with 5-fold stratified cross-validation, generates AUC values and ROC curves, and visualizes descriptor importance.

- 🧪 **Application of Trained Model to New Compounds (e.g., NuBBE)**  
  Applies the trained model to new samples from an Excel file, validates descriptor availability, and outputs predictions.

- 💾 **Export & Download of Results**  
  Saves and downloads selected subsets, trained model (`.pkl`), and prediction results as `.csv`.

This script supports the supplementary analyses described in the manuscript:

> *Converging XGBoost Machine Learning and Molecular Docking Strategies to Identify Attractants for Ceratitis capitata: Molecular Characterization and Database Curation of Natural Ligands for in vitro/in vivo tests*  
> Archives of Insect Biochemistry and Physiology (2025, submitted)
