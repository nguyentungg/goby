import os
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import Detector
import Core.Helper as help
import train_data as train
import face_recognition

# create detector object
detector = Detector(False)


def init():
    global detector
    if os.path.isfile('./DataBase/DataBase.json'):
        detector = Detector()

def remove_junk_images(source_dir, detector):
    employee_folder = os.listdir(source_dir)
    for employee in employee_folder:
        employee_path = os.path.join(source_dir, employee)
        if os.path.isfile(employee_path):
            return

        source_list = os.listdir(employee_path)

        # Start to crop images
        for file in source_list:
            # create file path
            file_path = os.path.join(employee_path, file)
            print(f'{file_path}')
            # Read file
            img = cv2.imread(file_path)
            result = detector.is_face(img)
            img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result_facerecog = face_recognition.face_locations(img)
            print(f'{result} -- {result_facerecog}')

def main():
    init()

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
        help.crop_images('Dataset/Raw', 'Dataset/Crop', detector)
        # remove_junk_images('Dataset/Crop', detector)
    elif choice == "t":
        train.training_model('Dataset/Crop')
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
