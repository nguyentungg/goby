from mtcnn import MTCNN
import cv2
from zipfile import ZipFile
import gdown
import os
from utils import Detector
from datetime import datetime


def crop_images(source_dir, dest_dir, detector):
    if detector.get_status() is False:
        detector = Detector()
        print("Ultis is loaded")
    print("[INFO] Cropping faces...")
    employee_folder = os.listdir(source_dir)
    for employee in employee_folder:
        employee_path = os.path.join(source_dir, employee)
        if os.path.isfile(employee_path):
            return

        employee_dest = os.path.join(dest_dir, employee)
        if os.path.isdir(employee_dest) == False:
            os.mkdir(employee_dest)

        source_list = os.listdir(employee_path)

        # Start to crop images
        for file in source_list:
            # create file path
            f_path = os.path.join(employee_path, file)
            # create image save path
            dest_path = os.path.join(employee_dest, file)
            print(f'{f_path} => {dest_path}')
            # Read file
            img = cv2.imread(f_path)
            # Crop face and write to a image
            detector.crop_image(img, dest_path)


def image_preprocessing(image_path):
    # read image
    img = cv2.imread(os.path.join(image_path))
    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_pretrain_model():
    url = 'https://drive.google.com/uc?id=1SH-9ApSaD78OS-AdKqZeyLMLHe-_GA8F'
    output_file = "model_weights.zip"
    gdown.download(url, output_file, quiet=False)
    # opening the zip file in READ mode
    with ZipFile(output_file, 'r') as zip:
        zip.printdir()
        print('Extracting all the files now...')
        zip.extractall()
        print('weights extracted Done!')

    os.remove(output_file)

def export_model():
    detector = Detector()
    detector.save_model()

def markAttendance(name, csv_path, is_record=False):
    if csv_path is None or name is None:
        return False
    with open(csv_path, 'r+') as file:
        myDataList = file.readline()
        nameList = []
        attend_time = None
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            attend_time = now.strftime('%d/%m/%y %H:%M:%S')
            if is_record:
                file.writelines(f'\n{name},{attend_time}')
        return attend_time

# Use this function to detecting a list of image
def detectImages(images_dir, detector):
    # create detector object
    if detector.get_status() is False:
        detector = Detector()
        print("Ultis is loaded")

    list_imgs = os.listdir(images_dir)
    # loop over images
    for im in list_imgs:
        # read image
        img = cv2.imread(os.path.join(images_dir, im))

        # resize image
        # img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

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
        # Show the image one by one
        cv2.waitKey(3)
        # Wait for key press to show the next image
        # cv2.waitKey(0)

def detectCamera(detector, camera_number=0):
    if detector.get_status() is False:
        detector = Detector()
        print("Ultis is loaded")

    cap = cv2.VideoCapture(camera_number)
    while True:
        success, img = cap.read()

        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # get predictions and draw them on image
        predictions = detector.get_people_names(img, speed_up=False, downscale_by=1)
        annoted_image = detector.draw_results(img, predictions, (255, 238, 0))

        # convert back to BGR (since using cv2)
        image = cv2.cvtColor(annoted_image, cv2.COLOR_RGB2BGR)

        cv2.imshow('Webcam', image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break