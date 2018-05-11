import os
from sklearn import svm

import numpy as np
import cv2
from skimage import feature

from django.conf import settings
from sklearn.externals import joblib

PERSON_TEMPLATES_DIR = 'person_templates'


def get_hist(image_bytes, raise_exception=False):
    image = np.frombuffer(image_bytes, dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    face_cascade = cv2.CascadeClassifier(
        os.path.join(settings.BASE_DIR, "app/cascades/haarcascade_frontalface_default.xml"))
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30),
    )
    faces = list(faces)
    if len(faces) > 0:
        rects = [w * h for (x, y, w, h) in faces]
        max_rect = np.argmax(rects)
        (x, y, w, h) = faces[max_rect]
        face_img = image[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (300, 300))
        face_img = cv2.equalizeHist(cv2.GaussianBlur(face_img, (5, 5), 0))
        hist = feature.hog(face_img, orientations=34, pixels_per_cell=(7, 7), cells_per_block=(1, 1),
                           feature_vector=True)
        return hist
    else:
        if raise_exception:
            raise Exception("No face found.")
        else:
            return {'error': "No face found."}


def build_classifier(user, images):
    templates_dir = os.path.join(settings.BASE_DIR, PERSON_TEMPLATES_DIR)

    # Save the new template for user
    # user_template = np.array([get_hist(image, raise_exception=True) for image in images])
    user_template = []
    for image in images:
        hist = get_hist(image, raise_exception=False)
        if not isinstance(hist, dict):
            user_template.append(hist)

    np.save(
        os.path.join(templates_dir, str(user.pk)),
        np.array(user_template)
    )

    labels = [0]
    templates = [user_template[0]]

    for f in os.listdir(templates_dir):
        if f.endswith(".npy"):
            # Read the template
            pk, _ = f.split(".")
            template = np.load(os.path.join(templates_dir, f))
            labels.extend([pk for _ in range(len(template))])
            templates.extend(template.tolist())

    clf = svm.SVC()
    clf.fit(templates, labels)
    joblib.dump(clf, settings.CLASSIFIER_PATH)
