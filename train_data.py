import sys

from Core.FaceRecognizer import FaceRecognizer
from imutils import paths
import cv2
import json
import os


def add_person_to_database(Name, feature_vector):
    # load DataBase.json File
    with open(os.path.join("DataBase", "DataBase.json"), "r") as file:
        data = json.load(file)

    # append the record
    data[Name] = feature_vector

    # Save updated record to DataBase.json
    with open(os.path.join("DataBase", "DataBase.json"), "w") as file:
        json.dump(data, file)
        print(f"{Name} is added to DataBase")


def classifier(images_folder, recognizer):
    if images_folder is None:
        return "Please specify the images folder "
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(images_folder))

    for (i, imagePath) in enumerate(imagePaths):

        print("[INFO] processing image {}/{} ({})".format(i + 1, len(imagePaths), imagePath))
        name = imagePath.split(os.path.sep)[-2]
        print(f'Training Name: {name}')
        try:
            image = cv2.imread(imagePath)
            face_embed = recognizer.get_face_embedding(image).flatten().tolist()
            add_person_to_database(name, face_embed)

        except IndexError as e:
            print(f"\r Error: Cannot encode image:\n")


def init_database():
    data_path = os.path.join(os.path.curdir, 'DataBase', 'DataBase.json')
    if os.path.isfile(data_path):
        print('[NOTE] DataBase.json is exits', file=sys.stderr)
        return
    with open(data_path, 'w') as f:
        empty_json = {}
        json.dump(empty_json, f)
        print('[NOTE] DataBase.json created', file=sys.stderr)


def check_database(data_path):
    if os.path.isfile(data_path):
        return True
    return False


def training_model(processed_dir, model_path='./DataBase/DataBase.json'):
    if not check_database(model_path):
        init_database()
    recognizer = FaceRecognizer()
    classifier(processed_dir, recognizer)
