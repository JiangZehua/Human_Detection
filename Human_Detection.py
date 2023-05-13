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
        raise NotImplementedError
    
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
    
    ################### for debugging purpose ###################
    # debug_num = [0, 1, 2, 3, 5, 20, 35, 95, 169, 171, 178, 179, 180]
    # for num in debug_num:
    #     cal_portions_index(num)


    # ################### for getting HOG features for selected images ###################
    # selected_training_images_path = ["/Users/zehuajiang/My/CS6643 Computer Vision Spring 2023/Project_2_Human_Detection/Image Data 2/Database images/positive/DB2.bmp",
    #                                 "/Users/zehuajiang/My/CS6643 Computer Vision Spring 2023/Project_2_Human_Detection/Image Data 2/Database images/positive/DB9.bmp",
    #                                "/Users/zehuajiang/My/CS6643 Computer Vision Spring 2023/Project_2_Human_Detection/Image Data 2/Database images/positive/DB15.bmp",]
    # train_img_list = ["DB2.bmp", "DB9.bmp",  "DB15.bmp"]
    # test_img_list = ["T2.bmp", "T5.bmp", "T10.bmp"]

    # for i, img_path in enumerate(selected_training_images_path):
    #     output_path = os.path.join("outputs", train_img_list[i].split(".")[0] + ".txt")
    #     image = cv2.imread(img_path)
    #     get_HOG_feature(image, output_path=output_path, save=True)
    # for img in test_img_list:
    #     img_path = os.path.join(test_folder, img)
    #     output_path = os.path.join("outputs", img.split(".")[0] + ".txt")
    #     image = cv2.imread(img_path)
    #     get_HOG_feature(image, output_path=output_path, save=True)

    ################### for classification ###################
    # get the HOG feature vector for the test images
    algos = ["histogram_intersection", "hellinger_distance"]
    for algo in algos:
        main(test_folder, train_folder, algo)
    


