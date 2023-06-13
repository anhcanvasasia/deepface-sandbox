import cv2
import torch
from torchvision import transforms
import numpy as np

import os
import time
import pandas as pd
from deepface import DeepFace
from deepface.commons import functions

from facenet_pytorch import MTCNN
from retinaface import RetinaFace


def get_detected_faces(face_objs: list):
    faces = []
    for face_obj in face_objs:
        facial_area = face_obj["facial_area"]
        faces.append((facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]))
    return faces


if __name__ == "__main__":
    db_path = './dataset/canvas-asia-face-img'
    model_name = "ArcFace"
    detector_backend = "retinaface"
    distance_metric = "cosine"
    enable_face_analysis = False
    enforce_detection = False
    silent = True
    source = 0
    time_threshold = 5
    frame_threshold = 5

    text_color = (255, 255, 255)
    pivot_img_size = 112
    target_size = functions.find_target_size(model_name=model_name)

    DeepFace.build_model(model_name=model_name)
    print(f"facial recognition model {model_name} is just built")

    freeze = False
    face_detected = False
    face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
    freezed_frame = 0
    tic = time.time()

    cap = cv2.VideoCapture(source)
    while True:
        _, img = cap.read()

        if img is None:
            break

        resolution_x = img.shape[1]
        resolution_y = img.shape[0]
        tic = time.time()

        face_objs = DeepFace.extract_faces(
            img_path=img,
            target_size=target_size,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
        )
        faces = get_detected_faces(face_objs)
        freeze_img = img.copy()
        base_img = img.copy()

        for detected_face in faces:
            x = detected_face[0]
            y = detected_face[1]
            w = detected_face[2]
            h = detected_face[3]

            cv2.rectangle(
                img, (x, y), (x + w, y + h), (67, 67, 67), 1
            )

            custom_face = base_img[y: y + h, x: x + w]
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
                    label = candidate["identity"]

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
                        if (
                                y - pivot_img_size > 0
                                and x + w + pivot_img_size < resolution_x
                        ):
                            # top right
                            freeze_img[
                            y - pivot_img_size: y,
                            x + w: x + w + pivot_img_size,
                            ] = display_img

                            overlay = freeze_img.copy()
                            opacity = 0.4
                            cv2.rectangle(
                                freeze_img,
                                (x + w, y),
                                (x + w + pivot_img_size, y + 20),
                                (46, 200, 255),
                                cv2.FILLED,
                            )
                            cv2.addWeighted(
                                overlay,
                                opacity,
                                freeze_img,
                                1 - opacity,
                                0,
                                freeze_img,
                            )

                            cv2.putText(
                                freeze_img,
                                label,
                                (x + w, y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                text_color,
                                1,
                            )

                            # connect face and text
                            cv2.line(
                                freeze_img,
                                (x + int(w / 2), y),
                                (x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                                (67, 67, 67),
                                1,
                            )
                            cv2.line(
                                freeze_img,
                                (x + 3 * int(w / 4), y - int(pivot_img_size / 2)),
                                (x + w, y - int(pivot_img_size / 2)),
                                (67, 67, 67),
                                1,
                            )

                        elif (
                                y + h + pivot_img_size < resolution_y
                                and x - pivot_img_size > 0
                        ):
                            # bottom left
                            freeze_img[
                            y + h: y + h + pivot_img_size,
                            x - pivot_img_size: x,
                            ] = display_img

                            overlay = freeze_img.copy()
                            opacity = 0.4
                            cv2.rectangle(
                                freeze_img,
                                (x - pivot_img_size, y + h - 20),
                                (x, y + h),
                                (46, 200, 255),
                                cv2.FILLED,
                            )
                            cv2.addWeighted(
                                overlay,
                                opacity,
                                freeze_img,
                                1 - opacity,
                                0,
                                freeze_img,
                            )

                            cv2.putText(
                                freeze_img,
                                label,
                                (x - pivot_img_size, y + h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                text_color,
                                1,
                            )

                            # connect face and text
                            cv2.line(
                                freeze_img,
                                (x + int(w / 2), y + h),
                                (
                                    x + int(w / 2) - int(w / 4),
                                    y + h + int(pivot_img_size / 2),
                                ),
                                (67, 67, 67),
                                1,
                            )
                            cv2.line(
                                freeze_img,
                                (
                                    x + int(w / 2) - int(w / 4),
                                    y + h + int(pivot_img_size / 2),
                                ),
                                (x, y + h + int(pivot_img_size / 2)),
                                (67, 67, 67),
                                1,
                            )

                        elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                            # top left
                            freeze_img[
                            y - pivot_img_size: y, x - pivot_img_size: x
                            ] = display_img

                            overlay = freeze_img.copy()
                            opacity = 0.4
                            cv2.rectangle(
                                freeze_img,
                                (x - pivot_img_size, y),
                                (x, y + 20),
                                (46, 200, 255),
                                cv2.FILLED,
                            )
                            cv2.addWeighted(
                                overlay,
                                opacity,
                                freeze_img,
                                1 - opacity,
                                0,
                                freeze_img,
                            )

                            cv2.putText(
                                freeze_img,
                                label,
                                (x - pivot_img_size, y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                text_color,
                                1,
                            )

                            # connect face and text
                            cv2.line(
                                freeze_img,
                                (x + int(w / 2), y),
                                (
                                    x + int(w / 2) - int(w / 4),
                                    y - int(pivot_img_size / 2),
                                ),
                                (67, 67, 67),
                                1,
                            )
                            cv2.line(
                                freeze_img,
                                (
                                    x + int(w / 2) - int(w / 4),
                                    y - int(pivot_img_size / 2),
                                ),
                                (x, y - int(pivot_img_size / 2)),
                                (67, 67, 67),
                                1,
                            )

                        elif (
                                x + w + pivot_img_size < resolution_x
                                and y + h + pivot_img_size < resolution_y
                        ):
                            # bottom righ
                            freeze_img[
                            y + h: y + h + pivot_img_size,
                            x + w: x + w + pivot_img_size,
                            ] = display_img

                            overlay = freeze_img.copy()
                            opacity = 0.4
                            cv2.rectangle(
                                freeze_img,
                                (x + w, y + h - 20),
                                (x + w + pivot_img_size, y + h),
                                (46, 200, 255),
                                cv2.FILLED,
                            )
                            cv2.addWeighted(
                                overlay,
                                opacity,
                                freeze_img,
                                1 - opacity,
                                0,
                                freeze_img,
                            )

                            cv2.putText(
                                freeze_img,
                                label,
                                (x + w, y + h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                text_color,
                                1,
                            )

                            # connect face and text
                            cv2.line(
                                freeze_img,
                                (x + int(w / 2), y + h),
                                (
                                    x + int(w / 2) + int(w / 4),
                                    y + h + int(pivot_img_size / 2),
                                ),
                                (67, 67, 67),
                                1,
                            )
                            cv2.line(
                                freeze_img,
                                (
                                    x + int(w / 2) + int(w / 4),
                                    y + h + int(pivot_img_size / 2),
                                ),
                                (x + w, y + h + int(pivot_img_size / 2)),
                                (67, 67, 67),
                                1,
                            )
                    except Exception as err:  # pylint: disable=broad-except
                        print(str(err))
            cv2.rectangle(freeze_img, (10, 10), (90, 50), (67, 67, 67), -10)
            cv2.putText(
                freeze_img,
                label,
                (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )

            cv2.imshow("img", freeze_img)

        # img = cv2.flip(img, 1)
        # cv2.imshow("img", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
            break

    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()