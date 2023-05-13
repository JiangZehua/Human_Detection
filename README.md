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

For all ten test images, the results after converting to grayscale, sobel operation, normalized HOG features, the gradient X, Gradient Y, Gradient magnitude, Gradient angle pictures are shown below, respectively.
<!-- Put the original image in a row with text on the right. -->

image: T1
<img src="outputs/Gradient Magnitude and angle of test images/T1.png" width="1000" height="200" />

\
image: T2
<img src="outputs/Gradient Magnitude and angle of test images/T2.png" width="1000" height="200" />

\
image: T3
<img src="outputs/Gradient Magnitude and angle of test images/T3.png" width="1000" height="200" />

\
image: T4
<img src="outputs/Gradient Magnitude and angle of test images/T4.png" width="1000" height="200" />

\
image: T5
<img src="outputs/Gradient Magnitude and angle of test images/T5.png" width="1000" height="200" />

\
image: T6
<img src="outputs/Gradient Magnitude and angle of test images/T6.png" width="1000" height="200" />

\
image: T7
<img src="outputs/Gradient Magnitude and angle of test images/T7.png" width="1000" height="200" />

\
image: T8
<img src="outputs/Gradient Magnitude and angle of test images/T8.png" width="1000" height="200" />

\
image: T9
<img src="outputs/Gradient Magnitude and angle of test images/T9.png" width="1000" height="200" />

\
image: T10
<img src="outputs/Gradient Magnitude and angle of test images/T10.png" width="1000" height="200" />

results for hellinger distance:
|  image   | 1st NN  |  2nd NN   | 3rd NN  |  3NN   | 
|  ----  | ----  |  ----  | ----  |  ----  | 
|T9.bmp	   | 'negative'| 'negative'| '-266.1388938664155, negative'|  negative|
|T8.bmp	   | '-352.2105569740596, negative'| '-349.2375502925308, positive'| '-334.58324231266977, positive'|	positive|
|T6.bmp	   | '-350.05272549617183, negative'| 'positive'| 'negative'|	negative|
|T10.bmp	 | 'negative'| 'positive'| 'negative'|	negative|
|T7.bmp	   | 'positive'| 'positive'| 'positive'| 	positive|
|T5.bmp	   | 'negative'| 'positive'| 'negative'| 	negative|
|T4.bmp	   | 'positive'| 'negative'| 'positive'|	positive|
|T1.bmp	   | 'negative'| 'positive'| 'positive'|	positive|
|T3.bmp	   | 'negative'| 'positive'| 'negative'|	negative|
|T2.bmp	   | 'negative'| 'positive'| 'positive'|	positive|

results for histogram_intersection distance:
|  image   | 1st NN  |  2nd NN   | 3rd NN  |  3NN   | 
|  ----  | ----  |  ----  | ----  |  ----  | 
|T9.bmp||	(-279.5175108973864, 'negative')|	|(-267.4690450067166, 'negative')|	(-266.1388938664155, 'negative')|	negative|
|T8.bmp	|(-352.2105569740596, 'negative')	|(-349.2375502925308, 'positive')	|(-334.58324231266977, 'positive')|	positive|
|T6.bmp	|(-350.05272549617183, 'negative')|	(-331.5974002492977, 'positive')|	(-325.19212061313, 'negative')|	negative|
|T10.bmp|	(-312.615625061327, 'negative')	|(-309.2273146190563, 'negative')	|(-301.7963253512087, 'positive')|	negative|
|T7.bmp	|(nan, 'positive')	|(nan, 'positive')	|(nan, 'positive')	|positive|
|T5.bmp	|(-356.7364950963952, 'negative')	|(-336.40645860827067, 'positive')|	(-335.7479312572468, 'negative')|	negative|
|T4.bmp	|(-339.98939783761955, 'positive')|	(-304.0639957778126, 'positive')|	(-290.9149007597948, 'positive')|	positive|
|T1.bmp	|(-298.4635541624302, 'negative')	|(-286.93469994389096, 'negative')|	(-284.0335986799098, 'positive')|	negative|
|T3.bmp	|(-333.16215706951823, 'negative')|	(-322.1184184820429, 'negative')|	(-319.97036283877503, 'positive')|	negative|
|T2.bmp	|(-359.62760551527356, 'positive')|	(-355.4920130195786, 'negative')|	(-335.50495321440695, 'positive')	|positive|

All corresponding results can be found in the folder `results`.
\
All codes are available in the repository https://github.com/JiangZehua/Human_Detection

work done by Zehua Jiang and Sanyu Feng
