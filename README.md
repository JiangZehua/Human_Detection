# CS6643 Project 2: Human Detection
## Install
```bash
python -m pip install -r requirements.txt
```

## Run
to run the program, run the following command, before you run it, please make sure you change the path of the image in the code, and comment/uncomment out the part you (don't) want to run.
```bash
python Human_Detection.py
```

## Visualization
to run the program to get the visualization, run the following command, before you run it, please make sure you change the path of the image in the code, and comment/uncomment out the part you (don't) want to run.
```bash
python visualization.py
```

## Results
<!-- put 3 images in a row with footnote-->

For all three test images, the results after applying 7x7 Gaussian filter, normalized magnitude images, normalized magnitude image after non-maximasuppression, the histograms of the normalized magnitude image after non-maxima suppression are shown below, respectively.
<!-- Put the original image in a row with text on the right and the result of Gaussian filter in another row. -->
Original images: Barbara, Goldhill, Peppers.

<img src="inputs/Barbara.bmp" width="200" height="200" alt="Barbara.bmp"/><img src="inputs/Goldhill.bmp" width="200" height="200" alt="Goldhill.bmp"/><img src="inputs/Peppers.bmp" width="200" height="200" alt="Peppers.bmp"/>

\
Results of Gaussian filter: Barbara, Goldhill, Peppers.

<img src="results/Barbara/Gaussian_smoothing.bmp" width="200" height="200" alt="Gaussian_smoothing.bmp"/><img src="results/Goldhill/Gaussian_smoothing.bmp" width="200" height="200" alt="Gaussian_smoothing.bmp"/><img src="results/Peppers/Gaussian_smoothing.bmp" width="200" height="200" alt="Gaussian_smoothing.bmp"/>

\
Normalized magnitude images: Barbara, Goldhill, Peppers.

<img src="results/Barbara/Normalized_magnitude_of_the_gradient.bmp" width="200" height="200" alt="Barbara.bmp"/><img src="results/Goldhill/Normalized_magnitude_of_the_gradient.bmp" width="200" height="200" alt="Goldhill.bmp"/><img src="results/Peppers/Normalized_magnitude_of_the_gradient.bmp" width="200" height="200" alt="Peppers.bmp"/>

\
Normalized magnitude image after non-maxima suppression: Barbara, Goldhill, Peppers.

<img src="results/Barbara/Non-Maximum_Suppression.bmp" width="200" height="200" alt="Barbara.bmp"/><img src="results/Goldhill/Non-Maximum_Suppression.bmp" width="200" height="200" alt="Goldhill.bmp"/><img src="results/Peppers/Non-Maximum_Suppression.bmp" width="200" height="200" alt="Peppers.bmp"/>

\
Histograms of the normalized magnitude image after non-maxima suppression: Barbara, Goldhill, Peppers.

<img src="results/Barbara/histogram_of_the_magnitude.png" width="200" height="200" alt="Barbara.bmp"/><img src="results/Goldhill/histogram_of_the_magnitude.png" width="200" height="200" alt="Goldhill.bmp"/><img src="results/Peppers/histogram_of_the_magnitude.png" width="200" height="200" alt="Peppers.bmp"/>



<!-- <img src="results/Barbara/binary_edge_map_0.25_threshold.bmp" width="200" height="200" /> <img src="results/Barbara/binary_edge_map_0.5_threshold.bmp" width="200" height="200" /> <img src="results/Barbara/binary_edge_map_0.75_threshold.bmp" width="200" height="200" /> -->

For the image `Barbara.bmp`, the edge maps with thresholds 0.25, 0.5, 0.75 are shown below, respectively.
<img src="results/Barbara/binary_edge_map_0.25_threshold.bmp" width="200" height="200" alt="binary_edge_map_0.25_threshold.bmp"/><img src="results/Barbara/binary_edge_map_0.5_threshold.bmp" width="200" height="200" alt="binary_edge_map_0.5_threshold.bmp"/><img src="results/Barbara/binary_edge_map_0.75_threshold.bmp" width="200" height="200" alt="binary_edge_map_0.75_threshold.bmp"/>
\
For the image `Goldhill.bmp`, the edge maps with thresholds 0.25, 0.5, 0.75 are shown below, respectively.
<img src="results/Goldhill/binary_edge_map_0.25_threshold.bmp" width="200" height="200" alt="binary_edge_map_0.25_threshold.bmp"/><img src="results/Goldhill/binary_edge_map_0.5_threshold.bmp" width="200" height="200" alt="binary_edge_map_0.5_threshold.bmp"/><img src="results/Goldhill/binary_edge_map_0.75_threshold.bmp" width="200" height="200" alt="binary_edge_map_0.75_threshold.bmp"/>
\
For the image `Peppers.bmp`, the edge maps with thresholds 0.25, 0.5, 0.75 are shown below, respectively.
<img src="results/Peppers/binary_edge_map_0.25_threshold.bmp" width="200" height="200" alt="binary_edge_map_0.25_threshold.bmp"/><img src="results/Peppers/binary_edge_map_0.5_threshold.bmp" width="200" height="200" alt="binary_edge_map_0.5_threshold.bmp"/><img src="results/Peppers/binary_edge_map_0.75_threshold.bmp" width="200" height="200" alt="binary_edge_map_0.75_threshold.bmp"/>


All corresponding results can be found in the folder `results`.
\
All codes are available in the repository https://github.com/JiangZehua/Canny_Edge_Detector

work done by Zehua Jiang and Sanyu Feng
