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
'''
compute the HOG (Histograms of Oriented Gradients) feature from an input image 
and then classify the HOG feature vector into human or no-human by using a 3-nearest neighbor (NN) classifier.
'''
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np

####################### helper functions for computing HOG features #######################
def convert_to_gray(image: np.ndarray):
    '''
    convert the image to gray scale, bmp file's color channel is BGR

    I = Round(0.299R + 0.587G + 0.114B)
    '''
    result = np.zeros((image.shape[0], image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            result[i, j] = round(0.299 * image[i, j, 2] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 0])
    return result


def gradient(image: np.ndarray):
    '''
    use the sobel's operator to compute the gradient of the image
    '''
    # sobel's operator
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    # compute the gradient, the border is set to 0
    gradient_x = np.zeros(image.shape)
    gradient_y = np.zeros(image.shape)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            gradient_x[i, j] = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_x)
            gradient_y[i, j] = np.sum(image[i - 1:i + 2, j - 1:j + 2] * sobel_y)
    return gradient_x, gradient_y


def gradient_magnitude(gradient_x: np.ndarray, gradient_y: np.ndarray):
    '''
    compute the gradient magnitude
    '''
    result = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    # normalize the result to 0-255
    result = result / np.max(result) * 255
    return result


def gradient_angle(gradient_x: np.ndarray, gradient_y: np.ndarray):
    '''
    compute the gradient angle
    '''
    result = np.arctan(gradient_y / gradient_x)
    # if both gradient_x and gradient_y are 0, then the angle is will be nan, so we need to set it to 0
    result[np.isnan(result)] = 0
    # convert the angle to degree
    result = (result + np.pi/2) / np.pi * 180
    return result


def cal_portions_index(num):
    '''
    calculate how far away from the bin centers, and use bin_pos to calculate the portion of spliting the magnitude to two bins
    '''
    bin_centers = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170, 190]) # 190 is for calculating the portion only
    bin_pos = num / 20 - 0.5
    bin_index_1 = int(np.floor(bin_pos))
    bin_index_2 = int(np.ceil(bin_pos))

    if bin_pos < 0:
        bin_index_1 = 8
        bin_index_2 = 0
        portion_2 = num + 180 - bin_centers[bin_index_1]
        portion_1 = bin_centers[bin_index_2] - num
    else:
        portion_1 = bin_centers[bin_index_2] - num
        portion_2 = num - bin_centers[bin_index_1]

    if bin_index_2 > 8:
        bin_index_2 = 0

    ## for debugging purpose
    # print(f"-------{num}-------")
    # print(f"bin_index_1: {bin_index_1}")
    # print(f"bin_index_2: {bin_index_2}")
    # print(f"portion_1: {portion_1}")
    # print(f"portion_2: {portion_2}")

    return portion_1 / 20, portion_2 / 20 , bin_index_1, bin_index_2


####################### main functions for getting HOG features #######################
def compute_HOG(image: np.ndarray):
    '''
    compute the normalized HOG feature vector, using L2 normalization
    '''
    cell_size = 8
    block_size = 2
    # compute the gradient
    gradient_x, gradient_y = gradient(image)
    # compute the gradient magnitude
    gradient_magnitude_image = gradient_magnitude(gradient_x, gradient_y)
    # compute the gradient angle
    gradient_angle_image = gradient_angle(gradient_x, gradient_y)
    # if the angle is larger than 180, then subtract 180
    gradient_angle_image = gradient_angle_image % 180

    # compute the histogram
    histogram = np.zeros((image.shape[0] // cell_size, image.shape[1] // cell_size, 9)) # 9 bins
    for i in range(image.shape[0] // cell_size):
        for j in range(image.shape[1] // cell_size):
            for k in range(cell_size):
                for l in range(cell_size):
                    # split bewteen two closest bins based on ditance to bin centers
                    portion_1, portion_2, bin_index_1, bin_index_2 = cal_portions_index(gradient_angle_image[i * cell_size + k, j * cell_size + l]) 

                    # compute the histogram for each cell
                    histogram[i, j, int(bin_index_1)] += gradient_magnitude_image[i * cell_size + k, j * cell_size + l] * portion_1
                    histogram[i, j, int(bin_index_2)] += gradient_magnitude_image[i * cell_size + k, j * cell_size + l] * portion_2
    assert histogram.shape == (20, 12, 9)

    # compute the overlapping blocks
    block = np.zeros((image.shape[0] // cell_size - block_size + 1, image.shape[1] // cell_size - block_size + 1, 36))
    for i in range(image.shape[0] // cell_size - block_size + 1):
        for j in range(image.shape[1] // cell_size - block_size + 1):
            for k in range(block_size):
                for l in range(block_size):
                    block[i, j, k * block_size : k * block_size + 9] = histogram[i + k, j + l]
    assert block.shape == (19, 11, 36)

    # compute the normalized HOG feature vector
    result = np.zeros((image.shape[0] // cell_size - block_size + 1, image.shape[1] // cell_size - block_size + 1, 36))
    for i in range(image.shape[0] // cell_size - block_size + 1):
        for j in range(image.shape[1] // cell_size - block_size + 1):
            result[i, j] = block[i, j] / np.sqrt(np.sum(block[i, j] ** 2)) # L2 normalization
            # replace the nan with 0 which caused by the L2 normalization when the sum of the block is 0
            result[np.isnan(result)] = 0
    assert ((image.shape[0] // cell_size - block_size + 1) * (image.shape[1] // cell_size - block_size + 1) * 36 == 7524)

    return result


def get_HOG_feature(image, output_path=None, save=False):
    '''
    get the HOG feature vector for the image
    '''
    # convert the image to gray scale
    image = convert_to_gray(image)
    # compute the HOG feature vector
    block_histogram = compute_HOG(image)
    # reshape the HOG feature vector
    HOG_feature = block_histogram.reshape(7524)
    # write the HOG feature vector to a txt file, each value separated by line break
    if save:
        with open(output_path, "w") as f:
            for i in range(HOG_feature.shape[0]):
                f.write(str(HOG_feature[i]) + "\n")
    return HOG_feature



####################### helper functions for classification #######################
def histogram_intersection(input_hist, sample_hist):
    '''
    compute the histogram intersection
    '''
    # Compare two arrays and returns a new array containing the element-wise minimas
    minima = np.minimum(input_hist, sample_hist) 
    intersection = 1 - (np.sum(minima))
    return intersection


def hellinger_distance(input_hist, sample_hist):
    '''
    compute the Hellinger distance
    '''
    # Compute the square root of the element-wise product
    root_product = np.sqrt(np.multiply(input_hist, sample_hist))
    
    # Compute the Hellinger distance
    distance = 1 - np.sum(root_product)
    return distance



####################### main functions for classification #######################
def three_nn(datasetfolder, img, database, algo):
    '''
    use a 3-nearest neighbor (NN) classifier to classify the HOG feature vector into human or no-human

    input_image: the input image from the test folder np.ndarray
    database: the list of tuples of the database images and their labels and names
    algo: the algorithm to use for classification, "histogram_intersection" or "hellinger_distance"

    return: tuples of the test images' name 
            and their 1st NN's (distance, label), 
                        2nd NN's (distance, label), 
                        3rd NN's (distance, label), 
            and the majority class
    '''
    # read the input image
    input_image = cv2.imread(os.path.join(datasetfolder, img))

    if algo == "histogram_intersection":
        distance = histogram_intersection
    elif algo == "hellinger_distance":
        distance = hellinger_distance
    else:
        raise NotImplementedError # we only have two algorithms for calculating the distance
    
    # get the HOG feature vector for the input image
    input_hist = get_HOG_feature(input_image)
    # compute the distance between the input image and the database images
    distances = []
    for i in range(len(database)):
        distances.append((distance(input_hist, get_HOG_feature(database[i][0])), database[i][1], database[i][2]))
    # sort the distances
    distances.sort(key=lambda x: x[0])
    # get the 1st, 2nd, 3rd NN's (distance, label)
    first_nn = distances[0] # the 1st NN
    second_nn = distances[1] # the 2nd NN
    third_nn = distances[2] # the 3rd NN
    # get the majority class
    if first_nn[1] == second_nn[1] or first_nn[1] == third_nn[1]:
        majority_class = first_nn[1]
    elif second_nn[1] == third_nn[1]:
        majority_class = second_nn[1]
    else:
        raise ValueError("The 1st, 2nd, 3rd NN's labels should have at least two same labels")
    # return the result
    return (img, (first_nn[0],first_nn[1]), (second_nn[0],second_nn[1]), (third_nn[0],third_nn[1]), majority_class)
    
    


def main(test_path, database_path, algo):
    '''
    main function for classification
    
    database: the list of tuples of the database images and their labels and names
    result: the list of tuples of the test images' name and their 1st NN's (distance, label), 2nd NN's (distance, label), 3rd NN's (distance, label), and the majority class
    '''
    # read the database images and have them in a tuple with their labels
    database = [] 
    result = [] 
    for label in os.listdir(database_path):
        for image in os.listdir(os.path.join(database_path, label)):
            image_path = os.path.join(database_path, label, image)
            database.append((cv2.imread(image_path), label, image))
    # classify the input image
    for img in os.listdir(test_path):
        result.append(three_nn(test_path, img, database, algo))
        print(f"Finish classifying {img}")
    # write the result to a txt file
    with open(f"outputs/{algo}_result.txt", "w") as f:
        for i in range(len(result)):
            f.write(f"{result[i][0]}\t{result[i][1]}\t{result[i][2]}\t{result[i][3]}\t{result[i][4]}\n")
    return result
    


if __name__ == "__main__":
    train_folder = "/Users/zehuajiang/My/CS6643 Computer Vision Spring 2023/Project_2_Human_Detection/Image Data 2/Database images"
    test_folder = "/Users/zehuajiang/My/CS6643 Computer Vision Spring 2023/Project_2_Human_Detection/Image Data 2/Test images"
    
    ################### for debugging purpose: debugging spliting the magnitude to two bins ###################
    # debug_num = [0, 1, 2, 3, 5, 20, 35, 95, 169, 171, 178, 179, 180]
    # for num in debug_num:
    #     cal_portions_index(num)


    # ################### for getting HOG features for selected images ###################
    selected_training_images_path = ["/Users/zehuajiang/My/CS6643 Computer Vision Spring 2023/Project_2_Human_Detection/Image Data 2/Database images/positive/DB2.bmp",
                                    "/Users/zehuajiang/My/CS6643 Computer Vision Spring 2023/Project_2_Human_Detection/Image Data 2/Database images/positive/DB9.bmp",
                                   "/Users/zehuajiang/My/CS6643 Computer Vision Spring 2023/Project_2_Human_Detection/Image Data 2/Database images/negative/DB15.bmp",]
    train_img_list = ["DB2.bmp", "DB9.bmp",  "DB15.bmp"]
    test_img_list = ["T2.bmp", "T5.bmp", "T10.bmp"]

    for i, img_path in enumerate(selected_training_images_path):
        output_path = os.path.join("outputs", train_img_list[i].split(".")[0] + ".txt")
        image = cv2.imread(img_path)
        get_HOG_feature(image, output_path=output_path, save=True)
    for img in test_img_list:
        img_path = os.path.join(test_folder, img)
        output_path = os.path.join("outputs", img.split(".")[0] + ".txt")
        image = cv2.imread(img_path)
        get_HOG_feature(image, output_path=output_path, save=True)

    ################### for classification ###################
    # get the HOG feature vector for the test images
    algos = ["histogram_intersection", "hellinger_distance"]
    for algo in algos:
        print(f"Start classifying using {algo}")
        main(test_folder, train_folder, algo)
    


```


All corresponding results can be found in the folder `results`.
\
All codes are available in the repository https://github.com/JiangZehua/Human_Detection

work done by Zehua Jiang and Sanyu Feng
