import cv2
import torch
from torchvision import transforms
import numpy as np

import os
import time
import pandas as pd
from deepface import DeepFace
from deepface.commons import functions, recognition_draw

from facenet_pytorch import MTCNN
from retinaface import RetinaFace


def get_detected_faces(face_objs: list):
    faces = []
    for face_obj in face_objs:
        facial_area = face_obj["facial_area"]
        faces.append((facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]))
    return faces


def simplified_recognition(img, pivot_img_size):
    resolution_x = img.shape[1]
    resolution_y = img.shape[0]

    face_objs = DeepFace.extract_faces(
        img_path=img,
        target_size=target_size,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
    )
    faces = get_detected_faces(face_objs)
    freeze_img = img.copy()
    opacity = 0.5

    for detected_face in faces:
        x = detected_face[0]
        y = detected_face[1]
        w = detected_face[2]
        h = detected_face[3]

        cv2.rectangle(
            img, (x, y), (x + w, y + h), (67, 67, 67), 1
        )

        custom_face = freeze_img[y: y + h, x: x + w]
        dfs = DeepFace.find(
            img_path=custom_face,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            silent=silent,
        )
        if len(dfs) > 0:
            # directly access 1st item because custom face is extracted already
            df = dfs[0]

            if df.shape[0] > 0:
                candidate = df.iloc[0]
                cosine_score = candidate['ArcFace_cosine']
                label = candidate["identity"]
                folder_name = label.split('/')[-2:][0]

                if cosine_score >= cosine_threshold:
                    print(f'Name: {folder_name} - Score: {cosine_score}')

                    # to use this source image as is
                    display_img = cv2.imread(label)
                    # to use extracted face
                    source_objs = DeepFace.extract_faces(
                        img_path=label,
                        target_size=(pivot_img_size, pivot_img_size),
                        detector_backend=detector_backend,
                        enforce_detection=False,
                        align=False,
                    )

                    if len(source_objs) > 0:
                        # extract 1st item directly
                        source_obj = source_objs[0]
                        display_img = source_obj["face"]
                        display_img *= 255
                        display_img = display_img[:, :, ::-1]
                    # --------------------
                    label = label.split("/")[-1]

                    try:
                        freeze_img, label = recognition_draw.img_and_label(x, y, w, h, freeze_img, display_img, pivot_img_size, label, resolution_x, resolution_y, text_color, opacity)
                    except Exception as err:  # pylint: disable=broad-except
                        print(str(err))
                        label = 'Unknown'
                else:
                    print(f'Name: Unknown - Suspect {folder_name} with Score: {cosine_score}')

        cv2.imshow("img", freeze_img)


def mtncc_detect(frame):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)
    boxes, _, points_list = mtcnn.detect(frame, landmarks=True)
    if boxes is not None:
        return True
    else:
        return False


if __name__ == "__main__":
    db_path = './dataset/canvas-asia-face-img'
    model_name = "ArcFace"
    detector_backend = "retinaface"
    distance_metric = "cosine"
    enable_face_analysis = False
    enforce_detection = False
    silent = True
    source = 0

    cosine_threshold = 0.3
    text_color = (255, 255, 255)
    pivot_img_size = 112
    target_size = functions.find_target_size(model_name=model_name)

    DeepFace.build_model(model_name=model_name)
    print(f"facial recognition model {model_name} is just built")

    cap = cv2.VideoCapture(source)
    while True:
        _, img = cap.read()
        if img is None:
            break

        if mtncc_detect(frame=img):
            simplified_recognition(img, pivot_img_size)
        else:
            cv2.imshow("img", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()