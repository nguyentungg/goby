import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import Detector
import Core.Helper as help
import train_data as train

# create detector object
detector = Detector(True)


def init():
    global detector
    if os.path.isfile('./DataBase/DataBase.json'):
        detector = Detector()


def main():
    init()
    path = 'dataset'
    print("\t**********************************************")
    print("\t***  Market Enterprise Vietnam AI Lab  ***")
    print("\t**********************************************")
    print("\t Press the keyboard to use the feature    ")
    print("\t r - Download the weight.")
    print("\t c - Crop images in the raw dataset.")
    print("\t t - Training the model.")
    print("\t d - Detect the image.")
    print("\t v - Detect by the camera.")
    print("\t e - Exit.")
    print("\t ------------------ADD-ON-------------------")
    print("\t x - Export pre-trained model.")

    choice = input("\t What would you like to do? ")
    if choice == "r":
        help.get_pretrain_model()
    elif choice == "c":
        help.crop_image('Dataset/Raw', 'Dataset/Crop')
    elif choice == "t":
        files = os.listdir("DataBase")
        train.training_model('Dataset/Crop', files)
    elif choice == "d":
        help.detectImages("Testing/Input", detector)
    elif choice == "v":
        help.detectCamera(detector)
    elif choice == "e":
        return
    elif choice == "x":
        help.export_model()


if __name__ == '__main__':
    main()
