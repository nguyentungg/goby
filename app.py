import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import Detector
import cv2
import Core.Helper as help
import train_data as train

# Set directories
BASE_DIR = "Testing/Input"
list_imgs = os.listdir("Testing/Input\\")


def excute():
    # create detector object
    detector = Detector()

    # loop over images
    for im in list_imgs:
        # read image
        img = cv2.imread(os.path.join(BASE_DIR, im))
        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # get predictions and draw them on image
        predictions = detector.get_people_names(img, speed_up=False, downscale_by=1)
        annoted_image = detector.draw_results(img, predictions)

        # convert back to BGR (since using cv2)
        image = cv2.cvtColor(annoted_image, cv2.COLOR_RGB2BGR)
        # Save image and show the annoted iamge
        cv2.imwrite(f"Testing/Output\\{im.split('.')[0]}_infered.png", image)
        cv2.imshow("image", image)
        cv2.waitKey(3)


def main():
    path = 'dataset'
    print("\t**********************************************")
    print("\t***  Market Enterprise Vietnam AI Lab  ***")
    print("\t**********************************************")
    print("\t Press the keyboard to use the feature    ")
    print("\t r - Download the weight.")
    print("\t c - Crop images in the raw dataset.")
    print("\t t - Training the model.")
    print("\t d - Detect the image.")
    print("\t e - Exit.")
    print("\t ------------------ADD-ON-------------------")
    print("\t s - Save the pretrain model.")
    print("\t x - Export trained model.")

    choice = input("\t What would you like to do? ")
    if choice == "r":
        help.get_pretrain_model()
    elif choice == "c":
        help.crop_image('Dataset/Raw', 'Dataset/Crop')
    elif choice == "t":
        files = os.listdir("DataBase")
        train.training_model('Dataset/Crop', files)
    elif choice == "d":
        excute()
    elif choice == "e":
        return


if __name__ == '__main__':
    main()
