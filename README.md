# Subtype-MGTP
We propose a cancer subtyping framework called Subtype-MGTP,
which integrates Multiple-Genomics-To-Proteomics translation.
Firstly, the translation module employs partially real protein
data as guidance to translate multi-omics data into predicted
protein data. Subsequently, the deep subspace clustering module
incorporates contrastive learning to cluster the predicted protein
data, leading to the final subtyping results. 

## Quick start
Subtype-MGTP is based on the Python program language. The network's implementation was based on the open-source library Pytorch 1.8.1.
We used the NVIDIA RTX 2080 Ti for the model training. It is recommended to use the conda command to configure the environment:
```
# create an environment for running
conda env create -f environment.yaml

# activate environment
conda activate environment

# The trained model model.pth file is large(>25MB) and inconvenient to upload.
# You can use the following command to train the translation module and the trained model will be saved as model.pth.
python A_train.py 

# Then you can run the following command to use the traind model and obtain the predicted protein data of all samples.
python A_main.py

# Finally, you can use the following command to cluster the predicted protein data. 
python B_main.py -t BRCA 
```
## Data
The genomics datasets used in this study are available at: https://github.com/haiyang1986/Subtype-GAN.
And the protein expression datasets used in this study was uploaded.



