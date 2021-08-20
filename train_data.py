from Core.FaceRecognizer import FaceRecognizer
from imutils import paths
import cv2
import json
import os

def check_model(files):
    if "DataBase.json" not in files:
        with open(os.path.join("DataBase", "DataBase.json"), "x") as file:
            empty_json = {}
            json.dump(empty_json, file)


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

def training_model(processed_dir, model_path="DataBase"):
    check_model(model_path)
    recognizer = FaceRecognizer()
    classifier(processed_dir, recognizer)
