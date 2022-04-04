import cv2
import numpy as np

class faces:
    def __str__(self):
        return "Picture contains {} faces.".format(self.faces_count)

    def __repr__(self):
        return self.image

    def __len__(self):
        return 1

    def __init__(self, path, info=True):
        # Declare object's variables.
        self._info = info
        self.faces: list = []
        self.faces_count: int = 0
        self._haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
        self.image_path: str = "Path is not assigned yet."
        self.faces_coordinates: np.ndarray = np.array([])
        self.highlighted_faces = None
        self.resize_percentage: int = 5
        # Set the image path.
        self.image_path = path

        # Read the image from path and assign it.
        self.image = cv2.imread(path)

        if self._info:
            print("[INFO]: {} image loaded.".format(self.image_path))

    def find_faces(self):
        # Resize the image
        if self.image.shape[1] > 1898:
            self.image = cv2.resize(self.image, ((self.image.shape[1]//100) * self.resize_percentage, (self.image.shape[0]//100) * self.resize_percentage))
        # Grayscale Image
        gray_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Detect faces_count by using opencv and haarcascade.
        self.faces_coordinates = self._haarcascade.detectMultiScale(gray_img, 1.2, 4)

        # Set faces_count count.
        self.faces_count = len(self.faces_coordinates)

        if self._info:
            print("[INFO]: this image contains {} faces.".format(self.faces_count))

    def highlight_faces(self, display=True):
        # Drawing rectangles around faces_count.
        self.highlighted_faces = self.image
        for (x, y, w, h) in self.faces_coordinates:
            cv2.rectangle(self.highlighted_faces, (x, y), (x+w, y+h), (50, 205, 50), 2)

        # Display the highlighted faces_count image.
        if display:

            cv2.imshow("Faces", self.highlighted_faces)

            cv2.waitKey()

    def extract_faces(self):
        # Crop faces in image.
        for (x, y, w, h) in self.faces_coordinates:
            self.faces.append(self.image[y+3:y+h-3, x+3:x+w-3])
