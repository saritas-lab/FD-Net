# FD-Net
Official repository for "FD-Net: An Unsupervised Deep Forward-Distortion Model for Susceptibility Artifact Correction in EPI".
- [arXiv](https://arxiv.org/abs/2303.10436)
- [MRM](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.29851)

```A. Zaid Alkilani, T. Çukur, and E.U. Saritas, “FD-Net: A Deep Forward-Distortion Model for Unsupervised Susceptibility Artifact Correction in EPI.” Accepted to Magnetic Resonance in Medicine (15 August 2023).```

# Demo
- The files under ```/TOPUP_files``` demonstrate how TOPUP could be performed.
- The files under ```/slicing``` demonstrate how the dataset could be sliced.
- The files under ```/network``` demonstrate how the proposed FD-Net can be constructed, trained, and evaluated.

# Dataset
Data is accessible from the Human Connectome Project's [database](https://db.humanconnectome.org/) (WU-Minn HCP Data - 1200 Subjects; see the paper for more details). Detailed information on the installation and usage of TOPUP can be found at [the official website](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup).

# Pretrained Networks
Weights for the pre-trained FD-Net can be found under ```/network/weights```.

# Training
Training code is provided in ```/network/fdnet.py```. See the relevant code section(s) (i.e., ```#%% TRAIN (DWI)``` and ```#%% LOAD DATA + TRAIN (FMRI)```).

# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.

```
@article{ZaidAlkilani2023,
	author = {Zaid Alkilani, Abdallah and Çukur, Tolga and Saritas, Emine Ulku},
	title = {FD-Net: An unsupervised deep forward-distortion model for susceptibility artifact correction in EPI},
	journal = {Magnetic Resonance in Medicine},
	volume = {n/a},
	number = {n/a},
	pages = {},
	doi = {https://doi.org/10.1002/mrm.29851},
	url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.29851},
	eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/mrm.29851}
}

@misc{ZaidAlkilani2023arxiv,
	title = "FD-Net: An Unsupervised Deep Forward-Distortion Model for Susceptibility Artifact Correction in EPI", 
	author = "Zaid Alkilani, Abdallah AND {\c C}ukur, Tolga AND Saritas, Emine Ulku",
	year = "2023",
  	eprint = "2303.10436",
  	archivePrefix = "arXiv",
  	primaryClass = "eess.IV"
}

@InProceedings{ZaidAlkilani2022,
  	author = "Zaid Alkilani, Abdallah AND {\c C}ukur, Tolga AND Saritas, Emine Ulku",
	title = "A Deep Forward-Distortion Model for Unsupervised Correction of Susceptibility Artifacts in EPI",
	booktitle = "Proceedings of the 30th Annual Meeting of ISMRM",
	year = "2022",
	pages = "0959",
 	month = "May",
	address = "London, United Kingdom"
}

```
(c) ICON Lab 2023

# Prerequisites
- python=3.9.6
- contextlib2==21.6.0
- numpy==1.23.3
- scipy==1.9.1
- scikit-image==0.18.1
- nibabel==4.0.2
- dipy==1.5.0
- tensorflow==2.10.0

# Acknowledgements
A preliminary version of this work was presented in the Annual Meeting of ISMRM in London, 2022. This work was supported by the Scientific and Technological Council of Turkey (TÜBİTAK) via Grant 117E116. Data were provided by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.

For questions/comments please send an email to: `alkilani[at]ee.bilkent.edu.tr`
