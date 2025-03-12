
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
By executing the code, the GaitDynamics models, and example .mot file with joint angles of an OpenSim skeletal model will be imported from GitHub.
Users can upload their own .mot files to the Colab notebook to obtain force predictions.
To use a reduced kinematic input combinations, simply delete the corresponding columns in the .mot file.

## Dataset
[AddBiomechanics Dataset](https://addbiomechanics.org/download_data.html)

## Publication
This repository includes the code and models for a [preprint]() and an [abstract](./figures/readme_fig/Tan_ASB2024.pdf).
