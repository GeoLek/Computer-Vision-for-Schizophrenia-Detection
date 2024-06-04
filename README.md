# Computer-Vision-for-Schizophrenia-Detection
An endeavor to create a Computer Vision project to detect Schizophrenia patients among Healthy Controls from fMRI data using 2D, 3D & 4D CNNs and a hybrid 3D CN + RNN (LSTM) model.

# Dataset
We will make use of the 4D fMRI data from the [UCLA](https://openfmri.org/dataset/ds000030/) dataset, under the revision 1.0.5. The data were preprocessed with the fMRIPrep pipeline.

# Run on 3D preprocessed fMRI data

After averaging the 4th dimensional of time, we preserved only the preprocessed fMRI data in 3 dimensions. Our simple 3D CNN achieved:
# Accuracy = 96.7 %

# Performance Metrics
![Screenshot 2024-05-21 23:04:09](https://github.com/GeoLek/Computer-Vision-for-Schizophrenia-Detection/assets/89878177/48470912-5b86-4cea-9b03-b9bf0ee4258d)

# Confusion Matrix
![confusion_matrix](https://github.com/GeoLek/Computer-Vision-for-Schizophrenia-Detection/assets/89878177/c0844a78-e3d7-4d18-a644-c551886de6c0)

# LICENSE
This project is licensed under the Apache License - see the [LICENSE](https://github.com/GeoLek/Computer-Vision-for-Schizophrenia-Detection/blob/main/LICENSE) file for details.
