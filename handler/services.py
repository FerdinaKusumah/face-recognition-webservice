import base64
import io

import face_recognition
from PIL import Image, ImageDraw


class Prediction:

    def __init__(self, train_model, distance_threshold):
        self.__model = train_model
        self.__distance_threshold = distance_threshold

    @staticmethod
    def show_prediction_labels_on_image(img_path, predictions, image_extension):
        pil_image = Image.open(io.BytesIO(img_path)).convert("RGB")
        draw = ImageDraw.Draw(pil_image)

        for name, (top, right, bottom, left) in predictions:
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # There's a bug in Pillow where it blows up with non-UTF-8 text
            # when using the default bitmap font
            name = name.encode("UTF-8")

            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        # Remove the drawing library from memory as per the Pillow docs
        del draw

        # convert pil to base64 encode image
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())

        # Display the resulting image
        return f'data:image/{image_extension};base64,{img_str.decode()}'

    def predict_image(self, data: bytes, image_extension: str):
        """Make prediction based on data upload
        :param image_extension: str
        :param data: bytes
        :return:
        """
        # get data model from database
        data_model = self.__model
        # Load image file and find face locations
        x_img = face_recognition.load_image_file(io.BytesIO(data))
        x_face_locations = face_recognition.face_locations(x_img)

        # If no faces are found in the image, return an empty result.
        if len(x_face_locations) == 0:
            return []

        # Find encodings for faces in the test image
        faces_encodings = face_recognition.face_encodings(x_img, known_face_locations=x_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = data_model.kneighbors(faces_encodings, n_neighbors=5)
        are_matches = [closest_distances[0][i][0] <= self.__distance_threshold for i in range(len(x_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        results = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                   zip(data_model.predict(faces_encodings), x_face_locations, are_matches)]

        return self.show_prediction_labels_on_image(data, results, image_extension)


class Helper:
    @staticmethod
    def allowed_file(filename: str) -> bool:
        """Check if filename is allowed
        :return:
        """
        # function to validate acceptable files
        ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
