# Quantifying Alzheimer's Disease Progression Through Automated Measurement of Hippocampal Volume

## Overview

Alzheimer's disease (AD) is a progressive neurodegenerative disorder that results in impaired neuronal (brain cell) function and eventually, cell death. AD is the most common cause of dementia. Clinically, it is characterized by memory loss, inability to learn new material, loss of language function, and other manifestations.

For patients exhibiting early symptoms, quantifying disease progression over time can help direct therapy and disease management.

A radiological study via MRI exam is currently one of the most advanced methods to quantify the disease. In particular, the measurement of hippocampal volume has proven useful to diagnose and track progression in several brain disorders, most notably in AD. Studies have shown reduced volume of the hippocampus in patients with AD.

The hippocampus is a critical structure of the human brain (and the brain of other vertebrates) that plays important roles in the consolidation of information from short-term memory to long-term memory. In other words, the hippocampus is thought to be responsible for memory and learning.

![Hippocampus](./readme.img/Hippocampus_small.gif)

Humans have two hippocampi, one in each hemishpere of the brain. They are located in the medial temporal lobe of the brain. Fun fact - the word "hippocampus" is roughly translated from Greek as "horselike" because of the similarity to a seahorse, a peculiarity observed by one of the first anatomists to illustrate the structure.

<img src="./readme.img/Hippocampus_and_seahorse_cropped.jpg" width=200/>

According to [studies](https://www.sciencedirect.com/science/article/pii/S2213158219302542), the volume of the hippocampus varies in a population, depending on various parameters, within certain boundaries, and it is possible to identify a "normal" range when taking into account age, sex and brain hemisphere.

<img src="./readme.img/nomogram_fem_right.svg" width=300>

There is one problem with measuring the volume of the hippocampus using MRI scans, though - namely, the process tends to be quite tedious since every slice of the 3D volume needs to be analyzed, and the shape of the structure needs to be traced. The fact that the hippocampus has a non-uniform shape only makes it more challenging. Do you think you could spot the hippocampi in this axial slice?

<img src="./readme.img/mri.jpg" width=200>

I built a piece of AI software that could help clinicians perform this task faster and more consistently. I focused on the technical aspects of building a segmentation model and integrating it into the clinician's workflow, leaving the dataset curation and model validation questions largely outside the scope of this project.

## Project Goal

I built an end-to-end AI system which features a machine learning algorithm that integrates into a clinical-grade viewer and automatically measures hippocampal volumes of new patients, as their studies are committed to the clinical imaging archive.

I used the dataset that contains the segmentations of the right hippocampus and the U-Net architecture to build the segmentation model.

After that, I integrated the model into a working clinical PACS such that it runs on every incoming study and produces a report with volume measurements.

## The Dataset

I used the "Hippocampus" dataset from the [Medical Decathlon competition](http://medicaldecathlon.com/). This dataset is stored as a collection of NIFTI files, with one file per volume, and one file per corresponding segmentation mask. The original images here are T2 MRI scans of the full brain. As noted, in this dataset I used cropped volumes where only the region around the hippocampus has been cut out. This makes the size of the dataset quite a bit smaller, the machine learning problem a bit simpler and allows me to have reasonable training times.

## Local Environment

Python 3.7+ environment with the following libraries for the first two sections of the project:

* nibabel
* matplotlib
* numpy
* pydicom
* PIL
* json
* torch (preferably with CUDA)
* tensorboard

In the 3rd section of the project, I worked with three software products for emulating the clinical network:

* [Orthanc server](https://www.orthanc-server.com/download.php) for PACS emulation
* [OHIF zero-footprint web viewer](https://docs.ohif.org/development/getting-started.html) for viewing images. Note that if you deploy OHIF from its github repository, at the moment of writing the repo includes a yarn script (`orthanc:up`) where it downloads and runs the Orthanc server from a Docker container. If that works for you, you won't need to install Orthanc separately
* If you are using Orthanc (or other DICOMWeb server), you will need to configure OHIF to read data from your server. OHIF has instructions for this: https://docs.ohif.org/configuring/data-source.html
* You need to configure Orthanc for auto-routing of studies to automatically direct them to your AI algorithm. For this you will need to take the script that you can find at `section3/src/deploy_scripts/route_dicoms.lua` and install it to Orthanc as explained on this page: https://book.orthanc-server.com/users/lua.html
* [DCMTK tools](https://dcmtk.org/) for testing and emulating a modality. Note that if you are running a Linux distribution, you might be able to install dcmtk directly from the package manager (e.g. `apt-get install dcmtk` in Ubuntu)

## Project Steps

### Section 1: Curating a dataset of Brain MRIs

<img src="./readme.img/Slicer.png" width=400em>

The data is located in `/data/TrainingSet` directory [here](https://github.com/iDataist/Hippocampal-Volume-Quantification-in-Alzheimer-s-Progression/tree/master/data/TrainingSet). I curated the dataset using [Final Project EDA.ipynb](https://github.com/iDataist/Hippocampal-Volume-Quantification-in-Alzheimer-s-Progression/blob/master/section1/Final%20Project%20EDA.ipynb).

### Section 2: Training a segmentation CNN

<img src="./readme.img/loss.png" width=400em>

I used [PyTorch](https://pytorch.org/) to train the model and [Tensorboard](https://www.tensorflow.org/tensorboard/) to visualize the results.

Run the script [run_ml_pipeline.py](https://github.com/iDataist/Hippocampal-Volume-Quantification-in-Alzheimer-s-Progression/blob/master/section2/src/run_ml_pipeline.py) to kick off the training pipeline. The code has hooks to log progress to Tensorboard. In order to see the Tensorboard output you need to launch Tensorboard executable from the same directory where `run_ml_pipeline.py` is located using the following command:

> `tensorboard --logdir runs --bind_all`

After that, Tensorboard will write logs into directory called `runs` and you will be able to view progress by opening the browser and navigating to default port 6006 of the machine where you are running it.

### Section 3: Integrating into a clinical network

<img src="./readme.img/ohif.png" width=400em>

I created an AI product that can be integrated into a clinical network and provide the auto-computed information on the hippocampal volume to the clinicians. The local environment replicates the following clinical network setup:

<img src="./readme.img/network_setup.png" width=400em>

Specifically, I have the following software in this setup:

* MRI scanner is represented by a script [section3/src/deploy_scripts/send_volume.sh](https://github.com/iDataist/Hippocampal-Volume-Quantification-in-Alzheimer-s-Progression/blob/master/section3/src/deploy_scripts/send_volume.sh). When you run this script it will simulate what happens after a radiological exam is complete, and send a volume to the clinical PACS. Note that scanners typically send entire studies to archives.
* PACS server is represented by [Orthanc](http://orthanc-server.com/) deployment that is listening to DICOM DIMSE requests on port 4242. Orthanc also has a DicomWeb interface that is exposed at port 8042, prefix /dicom-web. The PACS server is also running an auto-routing module that sends a copy of everything it receives to an AI server.
* Viewer system is represented by [OHIF](http://ohif.org/). It is connecting to the Orthanc server using DicomWeb and is serving a web application on port 3000.
* AI server is represented by a couple of scripts. [section3/src/deploy_scripts/start_listener.sh](https://github.com/iDataist/Hippocampal-Volume-Quantification-in-Alzheimer-s-Progression/blob/master/section3/src/deploy_scripts/start_listener.sh) brings up a DCMTK's `storescp` and configures it to just copy everything it receives into a directory that you will need to specify by editing this script, organizing studies as one folder per study.

[inference_dcm.py](https://github.com/iDataist/Hippocampal-Volume-Quantification-in-Alzheimer-s-Progression/blob/master/section3/src/inference_dcm.py) will analyze the directory of the AI server that contains the routed studies, find the right series to run the AI algorithm on, will generate report, and push it back to PACS.

In real system you would architect things a bit differently. Probably, AI server would be a separate piece of software that would monitor the output of the listener, and would manage multiple AI modules, deciding which one to run, automatically. In this case, for the sake of simplicity, all code sits in one Python script that was run manually after simulating an exam via the [deploy_scripts/send_volume.sh](https://github.com/iDataist/Hippocampal-Volume-Quantification-in-Alzheimer-s-Progression/blob/master/section3/src/deploy_scripts/send_volume.sh) script - [inference_dcm.py](https://github.com/iDataist/Hippocampal-Volume-Quantification-in-Alzheimer-s-Progression/blob/master/section3/src/inference_dcm.py). It combines the functions of processing of the listener output and executing the model.


The [deploy_scripts/send_volume.sh](https://github.com/iDataist/Hippocampal-Volume-Quantification-in-Alzheimer-s-Progression/blob/master/section3/src/deploy_scripts/send_volume.sh) script needs to be run from directory `section3/src` (because it relies on relative paths). An MRI scan will be sent to the PACS and to the AI module which will compute the volume, prepare the report and push it back to the PACS so that it could be inspected in our clinical viewer. At this point, go to *[YOUR IP ADDRESS]*:3000 which brings up the OHIF viewer. 
