from django.core.management.base import BaseCommand, CommandError

from django.conf import settings

import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

from app.utils import get_hist

PERSON_TEMPLATES_DIR = 'person_templates'
PERSON_IMAGES_DIR = 'person_images'


class Command(BaseCommand):
    def handle(self, *args, **options):
        templates_dir = os.path.join(settings.BASE_DIR, PERSON_TEMPLATES_DIR)
        images_dir = os.path.join(settings.BASE_DIR, PERSON_IMAGES_DIR)

        for dir in [d for d in os.listdir(images_dir) if d != '.DS_Store']:
            pk = int(dir)
            person_images = os.path.join(images_dir, dir)
            user_template = []
            for image in [d for d in os.listdir(person_images) if d != '.DS_Store']:
                image_path = os.path.join(images_dir, dir, image)
                hist = get_hist(open(image_path, 'rb').read(), raise_exception=False)
                if not isinstance(hist, dict):
                    user_template.append(hist)

            np.save(
                os.path.join(templates_dir, str(pk)),
                np.array(user_template)
            )

        labels = []
        templates = []

        for f in os.listdir(templates_dir):
            if f.endswith(".npy"):
                # Read the template
                pk, _ = f.split(".")
                template = np.load(os.path.join(templates_dir, f))
                labels.extend([pk for _ in range(len(template))])
                templates.extend(template.tolist())
        clf = svm.SVC(probability=True)
        print(labels)
        clf.fit(templates, labels)
        joblib.dump(clf, settings.CLASSIFIER_PATH)
