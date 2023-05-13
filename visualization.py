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