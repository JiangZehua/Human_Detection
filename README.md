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
|T9.bmp|	(-412.02776823816373, 'negative')|	(-403.5743512163623, 'positive')|	(-391.33410057560525, 'negative')	|negative|
|T8.bmp|	(-492.36778388657046, 'negative')|	(-489.47683837922636, 'positive')|	(-467.8597154552168, 'positive')|	positive|
|T6.bmp	|(-498.0999280835657, 'positive')	|(-494.25040945401435, 'negative')	|(-466.5081533981148, 'positive')	|positive|
|T10.bmp|	(-452.02285563729515, 'negative')|	(-444.8698106246808, 'positive')|	(-435.10156485279913, 'negative')|	negative|
|T7.bmp	|(-377.6320122288134, 'negative')	|(-372.2645706859943, 'negative')	|(-368.96750131272387, 'positive')	|negative|
|T5.bmp	|(-493.0168005098008, 'negative')	|(-489.68741911592826, 'positive')|	(-466.9690275610776, 'positive')	|positive|
|T4.bmp	|(-479.5248788556692, 'positive')	|(-465.17185129343983, 'negative')|	(-442.0889380304301, 'positive')	|positive|
|T1.bmp	|(-447.01503017086213, 'positive')|	(-439.4784060773296, 'negative')	|(-417.2004655915621, 'negative')	|negative|
|T3.bmp	|(-473.14704845628336, 'positive')|	(-472.8489229488249, 'negative')	|(-450.316661552941, 'positive')	|positive|
|T2.bmp	|(-499.6178160879462, 'positive')|	(-497.15058668846814, 'negative')	|(-485.69467134362475, 'positive')|	positive|

results for histogram_intersection distance:
|  image   | 1st NN  |  2nd NN   | 3rd NN  |  3NN   | 
|  ----  | ----  |  ----  | ----  |  ----  | 
|T9.bmp	|(-279.5175108973864, 'negative')	|(-267.4690450067166, 'negative')	|(-266.1388938664155, 'negative')	|negative|
|T8.bmp	|(-352.2105569740596, 'negative')	|(-349.2375502925308, 'positive')	|(-334.58324231266977, 'positive')|	positive|
|T6.bmp	|(-360.5603295802222, 'positive')	|(-350.05272549617183, 'negative')	|(-331.5974002492977, 'positive')|	positive|
|T10.bmp|	(-312.615625061327, 'negative')	|(-309.2273146190563, 'negative')	|(-302.3972968817565, 'positive')	|negative|
|T7.bmp	|(-268.931912682815, 'positive')	|(-264.5766923601239, 'negative')	|(-261.54277512132586, 'negative')|	negative|
|T5.bmp	|(-356.7364950963952, 'negative')	|(-350.6425077927667, 'positive')	|(-336.40645860827067, 'positive')|	positive|
|T4.bmp	|(-339.98939783761955, 'positive')|	(-318.60341375031237, 'negative')|	(-304.0639957778126, 'positive')|	positive|
|T1.bmp	|(-308.542192230583, 'positive')	|(-298.4635541624302, 'negative')|	(-286.93469994389096, 'negative')	|negative|
|T3.bmp	|(-333.16215706951823, 'negative')|	(-331.06161567124013, 'positive')	|(-322.1184184820429, 'negative')	|negative|
|T2.bmp|	(-361.70123451476667, 'positive')|	(-359.62760551527356, 'positive')|	(-355.4920130195786, 'negative')|	positive|

The ASCII (.txt) files containing the HOG feature values for three of the sample images 
from the database and three of the test images:
in folder outputs/classify results

source code:
```
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Human_Detection import compute_HOG, convert_to_gray, gradient, gradient_angle, gradient_magnitude


def show_img(origin_image: np.ndarray, save_name):
    '''
    show the image and its HOG feature vector, for debugging purpose and report
    '''
    # convert the image to gray scale
    origin_image = convert_to_gray(origin_image)

    # get the gradient magnitude and angle
    gradient_x, gradient_y = gradient(origin_image)
    gradient_magnitude_image = gradient_magnitude(gradient_x, gradient_y)
    gradient_angle_image = gradient_angle(gradient_x, gradient_y)

    # save the 5 plot in a same figure with original image, gradient_x, gradient_y, gradient magnitude, gradient angle
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(origin_image, cmap=plt.cm.gray)
    axs[0].set_title('Input image')
    axs[1].imshow(gradient_x, cmap=plt.cm.gray)
    axs[1].set_title('Gradient x')
    axs[2].imshow(gradient_y, cmap=plt.cm.gray)
    axs[2].set_title('Gradient y')
    axs[3].imshow(gradient_magnitude_image, cmap=plt.cm.gray)
    axs[3].set_title('Gradient magnitude')
    axs[4].imshow(gradient_angle_image, cmap=plt.cm.gray)
    axs[4].set_title('Gradient angle')
    # plt.show()
    for ax in axs:
        ax.set_axis_off()
    fig.savefig(f"outputs/{save_name}.png")

    # compute the HOG feature vector
    block_histogram = compute_HOG(origin_image)
    # reshape the HOG feature vector
    HOG_feature = block_histogram.reshape(7524)
    # show the image
    plt.figure(figsize=(8, 4))
    plt.subplot(121).set_axis_off()
    plt.imshow(origin_image, cmap=plt.cm.gray)
    plt.title('Input image')
    # show the HOG feature vector
    plt.subplot(122).set_axis_off()
    plt.plot(HOG_feature)
    plt.title('Histogram of Oriented Gradients')
    # save the figure
    plt.savefig(f"outputs/{save_name}_HOG.png")


if __name__ == "__main__":
    # debug_num = [0, 1, 2, 3, 5, 20, 35, 95, 169, 171, 178, 179, 180]
    # for num in debug_num:
    #     cal_portions_index(num)
    train_folder = "/Users/zehuajiang/My/CS6643 Computer Vision Spring 2023/Project_2_Human_Detection/Image Data 2/Database images"
    test_folder = "/Users/zehuajiang/My/CS6643 Computer Vision Spring 2023/Project_2_Human_Detection/Image Data 2/Test images"
    for img in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img)
        output_path = os.path.join("outputs", img.split(".")[0] + ".txt")
        image = cv2.imread(img_path)
        show_img(image, save_name=img.split(".")[0])
```

All corresponding results can be found in the folder `results`.
\
All codes are available in the repository https://github.com/JiangZehua/Human_Detection

work done by Zehua Jiang and Sanyu Feng
