from mtcnn import MTCNN
import cv2
from zipfile import ZipFile
import gdown
import os
from utils import Detector
from datetime import datetime

def crop_image(source_dir, dest_dir, mode=1): # mode = 1 means 1 face per image
    if os.path.isdir(dest_dir) == False:
        os.mkdir(dest_dir)
    detector = MTCNN()

    print("[INFO] quantifying faces...")
    employee_folder = os.listdir(source_dir)

    uncropped_file_list = []
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
            f_path = os.path.join(employee_path, file)
            dest_path = os.path.join(employee_dest, file)

            img = cv2.imread(f_path)
            data = detector.detect_faces(img)
            if data == []:
                uncropped_file_list.append(f_path)
            else:
                if mode == 1:  # Detect the box with the largest area
                    for i, faces in enumerate(data):  # Iterate through all the faces found
                        box = faces['box']  # Get the box for each face
                        biggest = 0
                        area = box[3] * box[2]
                        if area > biggest:
                            biggest = area
                            bbox = box
                    bbox[0] = 0 if bbox[0] < 0 else bbox[0]
                    bbox[1] = 0 if bbox[1] < 0 else bbox[1]
                    img = img[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]
                    cv2.imwrite(dest_path, img)
                    print(f'Saved: {dest_path}')
                else:
                    for i, faces in enumerate(data):  # Iterate through all the faces found
                        box = faces['box']
                        if box != []:
                            # Return all faces found in the image
                            box[0] = 0 if box[0] < 0 else box[0]
                            box[1] = 0 if box[1] < 0 else box[1]
                            cropped_img = img[box[1]: box[1] + box[3], box[0]: box[0] + box[2]]
                            fname = os.path.splitext(file)[0]
                            fext = os.path.splitext(file)[1]
                            fname = fname + str(i) + fext
                            save_path = os.path.join(employee_dest, fname)
                            cv2.imwrite(save_path, cropped_img)
                            print(f'Saved: {save_path}')

    return uncropped_file_list

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