[performance_metrics.txt](https://github.com/user-attachments/files/15928839/performance_metrics.txt)# Computer-Vision-for-Schizophrenia-Detection
An endeavor to create a Computer Vision project to detect Schizophrenia patients among Healthy Controls from fMRI data using 2D, 3D & 4D CNNs and a hybrid 3D CN + RNN (LSTM) model.

# Dataset
We will make use of the 4D fMRI data from the [UCLA](https://openfmri.org/dataset/ds000030/) dataset, under the revision 1.0.5. The data were preprocessed with the fMRIPrep pipeline.

# Run on 2D preprocessed fMRI data

After averaging the 4th dimensional of time, we preserved only the preprocessed fMRI data in 3 dimensions. Then we converted the 3D data to 2D images. Our 2D CNN achieved:
# Accuracy = 93 %

# Performance Metrics
![classification report](https://github.com/GeoLek/Computer-Vision-for-Schizophrenia-Detection/assets/89878177/b61d5c3a-acd1-452c-879d-34b642888a04)

# Confusion Matrix
![confusion_matrix](https://github.com/GeoLek/Computer-Vision-for-Schizophrenia-Detection/assets/89878177/ad2a294d-423e-4957-9e82-238216150680)

# LICENSE
This project is licensed under the Apache License - see the [LICENSE](https://github.com/GeoLek/Computer-Vision-for-Schizophrenia-Detection/blob/main/LICENSE) file for details.
