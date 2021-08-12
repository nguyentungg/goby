import Core.Helper as hp
from utils import Detector
from deepface import DeepFace
# source_dir = r'input' # directory with files to crop
# dest_dir = r'output' # directory where cropped images get stored
#
# uncropped_files_list = hp.crop_image(source_dir, dest_dir) # mode=1 means 1 face per image
# if len(uncropped_files_list) > 0:
#     for f in uncropped_files_list:
#         print(f)

choice = input("\t What would you like to do? ")
if choice == "s":
    obj = DeepFace.analyze(img_path = "Testing/Input/tung.jpg", actions = ['age', 'gender', 'race', 'emotion'])
    print(obj)
elif choice == "c":
    pass
