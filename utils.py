from Core.FaceDetector import FaceDetector
from Core.FaceRecognizer import FaceRecognizer
from datetime import datetime
import joblib
import json
import cv2


class Detector():

    def __init__(self, load=True):
        self.load = load
        self.FaceDetect = None
        self.FaceRecog = None
        if self.load is True:
            self.FaceDetect = FaceDetector()
            self.FaceRecog = FaceRecognizer()

    def load_detect(self):
        self.FaceDetect = FaceDetector()

    def load_recog(self):
        self.FaceRecog = FaceRecognizer()

    def get_people_names_svc(self, Model_dir, decode_json_dir, image, speed_up=True, downscale_by=4):
        """
		Arguments:
            model_dir - model directory for svc_classifier
            decode_json_dir - directory for json file containing data to decode svc
                                classifier output
            image - image on which to predict
            speed_up - bool whether to downscale image or not
            downscale_by - bigger the number faster the reults but lower accuracy
		Output:
            results - a lsit with following format
            [(confidence, person_name, box_co-ordinates),
            (confidence, person_name, box_co-ordinates), .........]

            box_co-ordinates = [xmin, ymin, xmax, ymax]
		"""

        # load svc classifier
        with open(Model_dir, "rb") as file:
            svc_clf = joblib.load(file)

            # load decode json
        with open(decode_json_dir, "r") as file:
            class_decode = json.load(file)

            # get bounding boxes for faces
        face_bboxes = self.FaceDetect.detect_faces(image, speed_up=speed_up,
                                                   scale_factor=downscale_by)

        # get face crops for faces
        Face_crops = self.FaceDetect.crop_faces(image, face_bboxes)

        # Store the result as a tuple in a list
        results = []
        for face_crop, box in zip(Face_crops, face_bboxes):
            # get face embedding for the face crop
            face_embd = self.FaceRecog.get_face_embedding(face_crop)
            # get svc_classifier output
            class_id = svc_clf.predict(face_embd)
            # confidence always 100 (only added to make it compatable with draw function)
            confidence = 100
            person_name = class_decode[str(class_id[0])]
            results.append((confidence, person_name, box))
        return results

    def get_people_names(self, image, speed_up=True, downscale_by=4):
        """
		Arguments:
            image - numpy array of image
            speed_up - bool whether to downscale image or not
            downscale_by - bigger the number faster the reults but lower accuracy
		Output:
            results - a lsit with following format
            [(distance, person_name, box_co-ordinates),
            (distance, person_name, box_co-ordinates), .........]

            box_co-ordinates = [xmin, ymin, xmax, ymax]
		"""
        # get bounding boxes for faces
        face_bboxes = self.FaceDetect.detect_faces(image, speed_up=speed_up,
                                                   scale_factor=downscale_by)
        # get face crops according to the bounding boxes
        Face_crops = self.FaceDetect.crop_faces(image, face_bboxes)

        # store the results in tuple format in list
        results = []
        for face_crop, box in zip(Face_crops, face_bboxes):
            # get face embedding
            face_embd = self.FaceRecog.get_face_embedding(face_crop)
            # get person_name and distance
            person_name, distance = self.FaceRecog.Whoisit(face_embd)
            results.append((distance, person_name, box))

        return results

    def get_people_only_names(self, image, speed_up=True, downscale_by=4):
        """
		Arguments:
            image - numpy array of image
            speed_up - bool whether to downscale image or not
            downscale_by - bigger the number faster the reults but lower accuracy
		Output:
            results - a lsit with following format
            [(distance, person_name, the time of verification),
            (distance, person_name, the time of verification), .........]

            box_co-ordinates = [xmin, ymin, xmax, ymax]
		"""
        # get bounding boxes for faces
        face_bboxes = self.FaceDetect.detect_faces(image, speed_up=speed_up,
                                                   scale_factor=downscale_by)
        # get face crops according to the bounding boxes
        Face_crops = self.FaceDetect.crop_faces(image, face_bboxes)

        # store the results in tuple format in list
        results = []
        for face_crop, box in zip(Face_crops, face_bboxes):
            # get face embedding
            face_embd = self.FaceRecog.get_face_embedding(face_crop)
            # get person_name and distance
            person_name, distance = self.FaceRecog.Whoisit(face_embd)
            # get the time of verification
            now = datetime.now()
            attend_time = now.strftime('%d/%m/%y %H:%M:%S')

            results.append((person_name, attend_time, distance))

        return results

    def draw_results(self, image, infer_results,
                     color=(255, 0, 0), box_thickness=None,
                     font_size=None, font_thickness=None,
                     offset=None):
        """
		Arguments:
            image - numpy array of image(RGB)
            infer_results  - result list from .get_people_name() and .get_people_name_svc() methods
            color - color of the bounding box as well as name
            box_thickness - thickess of the bounding box
            font_size - Size of the font
            font_thickness - thickness of the font
            offset - distance between top edge of box and alphabets of name
            (Leaving the above 4 option to None will automatically calculate best valus for all)
        Output:
            A seperate image instance with face boxes and person name drawn on to
            the image (a numpy array)
		"""
        # make deep copy of image
        img = image.copy()

        # Calculate best fraw setting and set if None is not 
        # provided
        settings = self.get_draw_settings(image.shape)
        if offset == None:
            offset = settings[0]
        if font_size == None:
            font_size = settings[1]
        if font_thickness == None:
            font_thickness = settings[2]
        if box_thickness == None:
            box_thickness = settings[3]

        # loop over results
        for result in infer_results:
            dist, name, box = result
            x1, y1, x2, y2 = box
            # draw bounding box
            img = cv2.rectangle(img, (x1, y1), (x2, y2),
                                color=color, thickness=box_thickness)
            # generate text to put over box
            text = "{} {:.2f}".format(name, dist)
            # put the text on image

            img = cv2.putText(img, text, (x1, y1 - offset),
                              cv2.FONT_HERSHEY_SIMPLEX, font_size,
                              color, font_thickness, cv2.LINE_AA)
        return img

    def get_draw_settings(self, image_shape):
        """
		Arguments:
            image_shape - shape of the image
		Output:
            Best setting for the image calculated by
            empherical relations formed from several best settings
		"""
        width, _, _ = image_shape
        offset = round(width / 150)
        font_size = round(width / 800, 2)
        font_thickness = round(width / 400)
        box_thickness = round(width / 300)
        return offset, font_size, font_thickness, box_thickness

    def save_model(self, path=None):
        """
        Output:
            Save the Facenet pre-tain model from current weights
        Note:
            Useful for convert pre-train model to another version
        """
        self.FaceRecog.export_model(path)

    def get_status(self):
        """
        Output:
            Return the status of this utils when init two objects FaceRecog and FaceDetection
            - True: mean FaceDetect and FaceRecog is loaded
            - False: mean FaceDetect and FaceRecog wasn't load
        """
        return self.load

    def recog_isload(self):
        """
        Output:
            Return the status of this utils when init two objects FaceRecog and FaceDetection
            - True: mean FaceRecog is loaded
            - False: mean FaceRecog wasn't loaded
        """
        return False if self.FaceRecog is None else True

    def crop_image(self, image, save_dir, speed_up=True, downscale_by=4):
        """
        Output:
            Detecting, croping face and write to a image file.
        """
        # convert to RGB (blue image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get bounding boxes for faces
        face_bboxes = self.FaceDetect.detect_faces(image, speed_up=speed_up,
                                                   scale_factor=downscale_by)
        # get face crops according to the bounding boxes
        Face_crops = self.FaceDetect.crop_faces(image, face_bboxes)
        for face_crop, box in zip(Face_crops, face_bboxes):
            # convert back to BGR (since using cv2)
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_dir, face_crop)

    def is_face(self, image, speed_up=True, downscale_by=4):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.FaceDetect.detect_faces(image, speed_up=speed_up,
                                                   scale_factor=downscale_by)
        return result

