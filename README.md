
# GaitDynamics: A Foundation Model for Analyzing Gait Dynamics 
By Tian Tan, Tom Van Wouwe, Keenon F. Werling, C. Karen Liu, Scott L. Delp, Jennifer L. Hicks, and Akshay S. Chaudhari

## Exclusive Summary
GaitDynamics is a generative foundation model for general-purpose gait dynamics prediction.
We illustrate in three diverse tasks with different inputs, outputs, and clinical impacts: i) estimating 
external forces from kinematics, ii) predicting the influence of gait modifications on knee loading without human 
experiments, and iii) predicting comprehensive kinematics and kinetic changes that occur with increasing running 
speeds.

## Environment
Our code is developed under the following environment. Versions different from ours may still work.

Python 3.9.16; Pytorch 1.13.1; Cuda 11.6; Cudnn 8.3.2; numpy 1.23.5;

## Trained model
GaitDynamics has [a diffusion model](/example_usage/GaitDynamicsDiffusion.pt) and 
[a force refinement model](/example_usage/GaitDynamicsRefinement.pt).
Downstream task 1 uses both models, while downstream task 2 and 3 use only the diffusion model.

## Force estimation with GaitDynamics
[A Google Colab notebook](https://colab.research.google.com/drive/1n6kH3gnwLdQ2DH5krigbkiO06NjDtyxI?usp=sharing)
is provided for the downstream tasks 1 – force estimation using flexible combinations of kinematic inputs.
Upload an OpenSim model file (.osim) and kinematic data files (.mot) following the instructions in the notebook.
Example files can be found in the [example_usage](/example_usage) folder.

## Dataset
[AddBiomechanics Dataset](https://addbiomechanics.org/download_data.html)

## Publication
This repository includes the code and models for a [preprint](https://assets-eu.researchsquare.com/files/rs-6206222/v1_covered_f6a08d22-5432-4743-b062-b8b8d886d664.pdf?c=1742524004)
and an [abstract](./figures/readme_fig/Tan_ASB2024.pdf).
