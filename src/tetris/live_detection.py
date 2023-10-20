import pathlib
import pickle

import cv2
import mediapipe as mp
import pandas as pd
import yaml
from sanic import Sanic
from sanic.response import html

from core.functions import *
from core.model import *
from gesture_control.data_preprocessor import DataPreprocessor

import time

root_dir = pathlib.Path(__file__).parent.parent.parent


# Methods
# TODO: use this to not trigger double events
def allDectectionsEqual(detections):
    detectionSet = set(detections)
    return len(detectionSet) == 1


def load_params(pickle_path):
    with open(pickle_path, "rb") as f:
        model = pickle.load(f)
        scaler = pickle.load(f)
        pca = pickle.load(f)

    return model, scaler, pca


def getKeypointNamesWithPostfix(keypointNames):
    """
    helper function to create list of all keypoint names (each has x and y)
    :param keypointNames:
    :return:
    """

    result = []
    for keypointName in keypointNames:
        result.append(f"{keypointName}_x")
        result.append(f"{keypointName}_y")
    return result


#############################################


# PARAMETERS
################################################
# Maps model to command
LABEL_MAP = {
    0: "idle",
    1: "right",
    2: "left",
    3: "rotate",
    4: "down",
    5: "up",
    6: "flip_table",
}
THRESHOLD = 0.5
NUM_FRAMES = 22
STRIDE = 1
FRAME_RATE = 33

model, scaler, pca = load_params(
    root_dir / "weights" / "all_gestures_model_lessKeypoints" / "model45.pkl"
)
preprocessor = DataPreprocessor(21)
################################################

show_video = True
show_data = True

tetris_root_path = pathlib.Path(__file__).parent

# you can find more information about sanic online https://sanicframework.org,
# but you should be good to go with this example code
app = Sanic("tetris_server")
app.static("/static", tetris_root_path)

app.static("/script.js", tetris_root_path / "script.js", name="script.js")
app.static("/style.css", tetris_root_path / "style.css", name="style.css")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


@app.route("/")
async def index(request):
    return html(open(tetris_root_path.joinpath("tetris.html"), "r").read())


@app.websocket("/events")
async def emitter(_request, ws):
    print("websocket connection opened")

    KEYPOINT_LIST = [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ]

    script_dir = pathlib.Path(__file__).parent
    mp_pose = mp.solutions.pose

    with open(script_dir.joinpath("keypoint_mapping.yml"), "r") as yaml_file:
        mappings = yaml.safe_load(yaml_file)
        KEYPOINT_NAMES = mappings["face"]
        KEYPOINT_NAMES += mappings["body"]

    buffer = []
    last_detection = 0

    while True:
        cap = cv2.VideoCapture(index=0)

        success = True
        # find parameters for Pose here: https://google.github.io/mediapipe/solutions/pose.html#solution-apis
        with mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose:
            while cap.isOpened() and success:
                # get pose
                success, image = cap.read()
                if not success:
                    break

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                if results.pose_landmarks == None:
                    continue

                # draw pose
                if show_video:
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )
                    cv2.imshow("MediaPipe Pose", image)
                    #cv2.waitKey(FRAME_RATE)
                    cv2.waitKey(1)

                # Fill buffer
                keypoints = [
                    [
                        results.pose_landmarks.landmark[
                            KEYPOINT_NAMES.index(joint_name)
                        ].x,
                        results.pose_landmarks.landmark[
                            KEYPOINT_NAMES.index(joint_name)
                        ].y,
                    ]
                    for joint_name in KEYPOINT_NAMES
                ]
                keypoints_array = np.array(keypoints).reshape(-1)
                if len(buffer) < NUM_FRAMES:
                    buffer.append(keypoints_array)
                    continue
                else:
                    buffer = buffer[1:]
                    buffer.append(keypoints_array)

                # Preprocessing
                buffer_df = pd.DataFrame(
                    buffer, columns=getKeypointNamesWithPostfix(KEYPOINT_NAMES)
                )
                buffer_array = preprocessor.preprocess_data(
                    buffer_df, including_ground_truth=False
                )

                buffer_array = buffer_array.reshape(-1)
                buffer_array = scaler.transform(buffer_array)
                buffer_array = pca.transform(buffer_array)

                # Trigger event
                model_output = Softmax().apply(model(buffer_array))
                confidence = np.max(model_output)
                if confidence >= THRESHOLD:
                    current_detection = np.argmax(model_output)
                else:
                    current_detection = 0

                print(LABEL_MAP[current_detection])

                if current_detection != 0 and last_detection == current_detection:
                    await ws.send(LABEL_MAP[current_detection])
                    time.sleep(0.5)
                    buffer = []

                last_detection = current_detection


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
