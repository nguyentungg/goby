import Core.Helper as hp
from utils import Detector
# source_dir = r'input' # directory with files to crop
# dest_dir = r'output' # directory where cropped images get stored
#
# uncropped_files_list = hp.crop_image(source_dir, dest_dir) # mode=1 means 1 face per image
# if len(uncropped_files_list) > 0:
#     for f in uncropped_files_list:
#         print(f)
detector = Detector()
choice = input("\t What would you like to do? ")
if choice == "s":
    detector.save_model()
elif choice == "c":
    pass
