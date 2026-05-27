# Contrastive Diffusion Alignment: Learning Structured Latents for Controllable Generation
![Method Overview](./docs/method_overview.png)
## Installation
Clone the repository:
```bash 
git clone https://github.com/Grosenick-Lab-Cornell/ConDA.git
cd ConDA
```
Create the conda environment:
```bash 
conda env create -f embedding_env.yml
conda activate embedding_env
```
To recreate the environment file from an existing conda environment, use:
```bash 
conda env export -n embedding_env --no-builds | grep -v "^prefix:" > embedding_env.yml
```
## Datasets information
- We release the Flow Past a Cylinder benchmark dataset, available on Hugging Face at `ruchi-sandilya/Flow-Past-Cylinder`. It can be downloaded using:
```bash
hf download ruchi-sandilya/Flow-Past-Cylinder \
  --repo-type dataset \
  --local-dir ./data/flow_past_cylinder
```
We also provide details of the FEM simulations used to generate the Flow Past a Cylinder benchmark image and graph datasets in the following repository: https://github.com/ruchi-sandilya/Flow_past_a_cylinder_benchmark

- The two-photon calcium imaging data are available upon request from the authors of the original study (Spellman et al., 2021). 
- The DISFA dataset is available upon request through the official dataset website https://www.mohammadmahoor.com/pages/databases/disfa/. 
- The monkey reaching dataset, MC\_Maze, is publicly available at https://dandiarchive.org/dandiset/000128 (download using `dandi download DANDI:000128/0.220113.0400` in `data/monkey`)
- However, for E-Field simulation, we cannot provide the real patient neuroimaging data because these data are part of an ongoing clinical trial and are HIPAA-protected.

## Experiments
For training the cLDM model use the script file run_train.sh. Use diffusion_latents_generator.py to generate the diffusion latents from pretrained diffusion model. The experiments for comparing the linear and nonlinear interpolations, n-step extrapolations and baseline comparisons are done in the .ipynb file and respective folders. 

## Citation
```
@inproceedings{sandilya2026conda,
	author = {Ruchi Sandilya and Sumaira Perez and Charles Lynch and Lindsay Victoria and Benjamin Zebley and Derrick Matthew Buchanan and Mahendra T. Bhati and Nolan Williams and Timothy J. Spellman and Faith M. Gunning and Conor Liston and Logan Grosenick},  
	title = {Contrastive Diffusion Alignment: Learning Structured Latents for Controllable Generation},  
	journal = {Forty-Third International Conference on Machine Learning},
	year = {2026}  
}
```
## Contact
For questions, please contact: Ruchi Sandilya, Grosenick Lab, Weill Cornell Medicine, Email: sar4018@med.cornell.edu

