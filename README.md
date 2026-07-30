# Gradient Pattern Analysis (GPA):
## Two New Applications for Pattern Characterization in Physics of Extreme Events and Extragalactic Astronomy

This repository contains the source code, datasets, experiments, documentation, and research materials developed during the Master's Dissertation:

> **Gradient Pattern Analysis (GPA): Two New Applications for Pattern Characterization in Physics of Extreme Events and Extragalactic Astronomy**

The research investigates the application of **Gradient Pattern Analysis (GPA)** to characterize complex spatial and spatio-temporal patterns in two scientific domains:

- Physics of Extreme Events
- Extragalactic Astronomy

The repository includes the complete Python implementation of the GPA framework, experimental notebooks, datasets, and all supporting material produced throughout the research.

📄 **Master's Dissertation Proposal**

The complete dissertation proposal is available here:

[Gradient Pattern Analysis (GPA): Two New Applications for Pattern Characterization in Physics of Extreme Events and Extragalactic Astronomy](https://github.com/Desduh/msc_gradient_pattern_analysis_extreme_events_extragalactic_astronomy/blob/main/documents/falandes_master_proposal_gradient_pattern_analysis_gpa_two_new_applications.pdf)

---

# Author

**Carlos Eduardo Falandes**

M.Sc. Student in Applied Computing (PPGCAP)

National Institute for Space Research (INPE)

📧 carlos.falandes@inpe.br

📧 eduardofalandess@gmail.com

---

# Advisor

**Prof. Dr. Reinaldo Roberto Rosa**

National Institute for Space Research (INPE)

📧 reinaldo.rosa@inpe.br

---

# Institution

**Graduate Program in Applied Computing (PPGCAP)**

National Institute for Space Research (INPE)

São José dos Campos – SP, Brazil

---

# Research Overview

Gradient Pattern Analysis (GPA) is a methodology for quantifying the morphology of complex systems through the analysis of gradient fields and their associated gradient moments. This dissertation investigates the potential of GPA as a quantitative tool for describing asymmetric patterns, nonlinear structures, and structural reorganizations across different scientific domains.

The research is organized around two complementary applications.

The first investigates **physics of extreme events**, where a **Spatio-Temporal Multifractal Cascade Model (STM-model)** is proposed as a multidimensional extension of the classical multifractal *p-model*. The model is implemented in both **(1D + 1)t** and **(2D + 1)t** configurations to generate synthetic multifractal fields exhibiting different dynamical regimes associated with endogenous and exogenous extreme events. The temporal evolution of the gradient moments is analyzed before, during, and after these events to evaluate their sensitivity to structural changes within multifractal systems.

The second application focuses on **extragalactic astronomy**, investigating the morphological characterization of peculiar and gravitationally interacting galaxies using observations primarily from the **Legacy Survey of Space and Time (LSST)**, complemented by data from the **Sloan Digital Sky Survey (SDSS)**. After image preprocessing and homogenization, the gradient moments are computed and compared with established non-parametric morphological descriptors, including entropy, the Gini coefficient, and the CASGM parameters. Particular emphasis is placed on the **second gradient moment (G2)** as a descriptor of gravitational interactions, structural asymmetries, and galaxy merger stages.

Beyond these applications, this repository also contains the development of a modern Python implementation of GPA, supporting experiments, reproducible notebooks, and datasets. The ultimate goal is to consolidate GPA as a robust quantitative descriptor of complex patterns and to investigate its potential as a feature extraction technique for artificial intelligence and machine learning models applied to classification and pattern recognition problems.

---

# Repository Structure

```text
.
├── documents/                 # Dissertation, papers, reports, presentations, and documentation
├── references/                # Main bibliography and reference papers (PDF)
└── source/
    ├── data/                  # Datasets used and generated during the research
    ├── gpa/                   # Python implementation of Gradient Pattern Analysis
    └── pocs/                  # Proof-of-concept experiments and prototypes
```

---

# Citation

If you use this repository in your research, please cite the corresponding dissertation.

```bibtex
@mastersthesis{falandes2026,
  author = {Carlos Eduardo Falandes},
  title = {Gradient Pattern Analysis (GPA): Two New Applications for Pattern Characterization in Physics of Extreme Events and Extragalactic Astronomy},
  school = {National Institute for Space Research (INPE)},
  year = {2026}
}
```

---

# License

This project is intended for academic and research purposes.
