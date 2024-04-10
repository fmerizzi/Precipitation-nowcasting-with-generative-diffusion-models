# Precipitation-nowcasting-with-generative-diffusion-models
Code relative to the publication [Precipitation nowcasting with generative diffusion models](https://arxiv.org/abs/2308.06733).

The code is implemented in Tf/Keras, Dataset is avaliable [here](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview), setup.py defines the hyperparameters for training and evaluation.

### Installation 
Users can easily reproduce the environment by running following commands
```
git clone https://github.com/fmerizzi/Precipitation-nowcasting-with-generative-diffusion-models
cd Precipitation-nowcasting-with-generative-diffusion-models
conda env create -f precipitation_nowcasting.yml
conda activate precipitation_nowcasting
```
