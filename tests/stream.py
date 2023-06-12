from deepface import DeepFace

DeepFace.stream(db_path="dataset/canvas-asia-face-img",
                model_name='ArcFace',
                detector_backend='retinaface',
                enable_face_analysis=False,
                time_threshold=1,
                frame_threshold=1)