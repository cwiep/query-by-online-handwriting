# Word Spotting with Online Handwritten Queries

Research code for master thesis of Christian Wieprecht (July 2015) and resulting DAS2016 paper

Christian Wieprecht, Leonard Rothacker, Gernot A. Fink, _"Word Spotting in Historical Document Collections with Online-Handwritten Queries"_, In Proc. IAPR Int. Workshop on Document Analysis Systems, Santorini, Greece, 2016

## Developed and tested with
* python2.7
* python-sklearn 0.14.1-2
* python-numpy 1.8.2
* vlfeat and python-bindings for calculating sift descriptors and using fast vl_ikmeans clustering

## Folders
* dataprep/ contains scripts for transforming raw word image and online trajectory data into feature vectors
* demo/ contains contains a very simple demo application that depends on precalculated files
* experiments/ contains the scripts with the experiments run for the DAS2016 paper
* tools/ contains helper modules for evaluation, transcription, machine learning and math operations
* wordspotting/ contains modules for feature calculation of word images, text words and online trajectories

## Formats

**Word image** definitions are given by an image of the document page (for calculating visual words) and a 
corresponding text file containing the bounds and transcription for each word image with line format
"xstart ystart xend yend transcription".

**Online handwritten trajectories** are defined in one text file per word with line format "x y penup", 
where penup is either 0 or 1 depending on the state of the pen (1 = pen is not touching surface). 
The transcription of the represented word is read from the filename. The format used for the presented experiments
is "id_transcription.txt" (e.g. "0346_company.txt"). See also [Online-Handwritten George Washington Dataset].

## How to run the experiments

0. In each step remember to check the data paths in the scripts!
1. Calculate visual words, keypoints and labels for the George Washington dataset with dataprep/extract_gw_codebook.py
2. Calculate online-handwriting codebook and labels for UNIPEN dataset with dataprep/extract_unipen_codebook.py
3. Calculate online-handwriting codebook labels for [Online-Handwritten George Washington Dataset] using codebook from step 2 with dataprep/calculate_gwo_labels.py
4. Precalculate feature vector matrices with dataprep/precalc_param_comb.py
5. Run any experiment in experiments/

Note that calculating sift descriptors and building a visual codebook requires the vlfeat library. 

## Folder structure of data ROOT_FOLDER

Here is the structure of the data folder that was used during the experiments. All current paths in all scripts
currently are aimed at supporting this structure.

```
/

--/gw_ls12           George Washington dataset
-----/GT             word image annotations
-----/page_data      output of scripts in dataprep
-----/pages          page image png files

--/gw_online         George Washington Online dataset
-----/keypointss     keypoint files for trajectories of each page
--------/2700270  
--------/2710271
         ...
-----/labels         label files for trajectories of each page
--------/256         codebook size
-----------/2700270 
-----------/2710271 
            ...
-----/raw            git clone of https://github.com/cwiep/gw-online-dataset

--/unipen
-----/clusters
--------/256         codebooks with size 256
-----/keypointss     keypoint files for trajectories of each writer
--------/writer1  
--------/writer2
         ...
-----/labels         label files for trajectories of each writer
--------/256         codebook size
-----------/writer1   
-----------/writer2 
            ...
-----/raw            writer folders from Unipen subset
```

## License

Apache License, Version 2.0

[//]: #

   [Online-Handwritten George Washington Dataset]: <https://github.com/cwiep/gw-online-dataset>
