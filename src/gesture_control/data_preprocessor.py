from pathlib import Path

import numpy as np
import pandas as pd

from gesture_control.data_scaler import DataScaler

num_frames = 21
idle_stepwidth = 3

LABEL_DICT = {
    "idle": 0,
    "swipe_left": 1,
    "swipe_right": 2,
    "rotate": 3,
    # "swipe_down": 4,
    # "swipe_up": 5,
    # "flip_table": 6,
}


class DataPreprocessor:
    def __init__(
        self,
        including_frames=21,
        data_scaler=None,
        relative_to_first_frame=False,
        percentage_majority=0.5,
    ):
        self.including_frames = including_frames
        self.relative_to_first_frame = relative_to_first_frame
        self.percentage_majority = percentage_majority
        if data_scaler is None:
            self.data_scaler = DataScaler()
        else:
            self.data_scaler = data_scaler

    def preprocess_data(self, frames, including_ground_truth=True):
        x = []
        keys = [
            "left_thumb",
            "right_thumb",
            "left_elbow",
            "right_elbow",
            "left_shoulder",
            "right_shoulder",
        ]
        y = []

        for index in range(
            0, len(frames) - self.including_frames, self.including_frames
        ):
            row = []
            scale_vector = self.data_scaler.get_scale_value(
                frames["left_shoulder_x"][index : index + self.including_frames],
                frames["left_shoulder_y"][index : index + self.including_frames],
                frames["right_shoulder_x"][index : index + self.including_frames],
                frames["right_shoulder_y"][index : index + self.including_frames],
            )
            for key in keys:
                x_values, y_values = self.get_x_and_y_values_for_key(frames, index, key)
                scaled_x = np.multiply(x_values, scale_vector)
                scaled_y = np.multiply(y_values, scale_vector)
                row.extend(scaled_x)
                row.extend(scaled_y)
            row.extend(self.get_distances_hand_to_mouth(frames, index))
            x.append(row)
            if including_ground_truth:
                y.append(self.get_most_occuring_ground_truth(frames, index))
        if including_ground_truth:
            return np.array(x), np.array(y)
        return np.array(x)

    def get_most_occuring_ground_truth(self, frames, index):
        unique, pos = np.unique(
            frames["ground_truth"][index : index + self.including_frames],
            return_inverse=True,
        )
        counts = np.bincount(pos)
        if (counts[counts.argmax()] / self.including_frames) > self.percentage_majority:
            maxpos = counts.argmax()
            ground_truth = unique[maxpos]
            return ground_truth
        # return "idle"
        return 0

    def get_x_and_y_values_for_key(self, frames, index, key):
        x_key = key + "_x"
        y_key = key + "_y"
        x_value = frames[x_key]
        y_value = frames[y_key]
        previous_x = x_value.iloc[index]
        previous_y = y_value.iloc[index]
        x_values = []
        y_values = []
        for i in range(index + 1, index + self.including_frames + 1):
            x_values.append(x_value.iloc[i] - previous_x)
            y_values.append(y_value.iloc[i] - previous_y)
            if not self.relative_to_first_frame:
                previous_x = x_value.iloc[i]
                previous_y = y_value.iloc[i]
        return x_values, y_values

    def get_distances_hand_to_mouth(self, frames, index):
        distances = []
        distances_right_x = []
        distances_right_y = []
        distances_left_x = []
        distances_left_y = []
        for i in range(index, index + self.including_frames):
            distances_right_x.append(
                frames["right_mouth_x"].iloc[i] - frames["right_thumb_x"].iloc[i]
            )
            distances_right_y.append(
                frames["right_mouth_y"].iloc[i] - frames["right_thumb_y"].iloc[i]
            )
            distances_left_x.append(
                frames["left_mouth_x"].iloc[i] - frames["left_thumb_x"].iloc[i]
            )
            distances_left_y.append(
                frames["left_mouth_y"].iloc[i] - frames["left_thumb_y"].iloc[i]
            )
        distances.extend(distances_right_x)
        distances.extend(distances_right_y)
        distances.extend(distances_left_x)
        distances.extend(distances_left_y)
        return distances


class GestureSampler:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data_preprocessor = DataPreprocessor()

    @staticmethod
    def get_consecutive_segments_from_df(df):
        df_copy = df.copy()
        df_copy["Group"] = (
            df_copy["ground_truth"] != df_copy["ground_truth"].shift()
        ).cumsum()

        # Use groupby to get the indices of each segment
        consecutive_segments = df_copy.groupby("Group").groups.values()

        # Convert the results to a list for easy access
        consecutive_segments = [segment.tolist() for segment in consecutive_segments]

        return consecutive_segments

    @staticmethod
    def _get_value_at_index(df, idx: int):
        return df.iloc[idx]["ground_truth"]

    def extract_gestures(self, df, consecutive_segments):
        gesture_indices_list = []

        for segments_list in consecutive_segments:
            value = self._get_value_at_index(df, segments_list[0])

            if value == 0:
                idx = segments_list[0]
                while self._get_value_at_index(idx + num_frames - 1) == 0:
                    gesture_indices_list.append(((idx, idx + num_frames - 1), value))
                    idx += idle_stepwidth

                    # check if index out of bounds
                    if idx + num_frames - 1 >= len(df):
                        break

            else:  # gesture
                center_index = segments_list[len(segments_list) // 2]
                start_index = center_index - num_frames // 2
                end_index = start_index + num_frames - 1

                # check if index out of bounds
                if end_index >= len(df):
                    continue

                if start_index < 0:
                    continue

                if self._get_value_at_index(df, end_index) != value:
                    print(
                        "warning: num_frames may to be too large, idle frames added to gesture"
                    )
                    # continue

                gesture_indices_list.append(((start_index, end_index), value))

        return gesture_indices_list

    @staticmethod
    def filter_df_by_index_intervals(df, index_intervals):
        mask = pd.Series(False, index=df.index)

        for start, end in index_intervals:
            mask |= (df.index >= start) & (df.index <= end)

        # Apply the mask to the DataFrame
        filtered_df = df.loc[mask]

        return filtered_df

    @staticmethod
    def tuple_to_df(X, y):
        df = pd.DataFrame(X)
        df["ground_truth"] = y
        return df

    def sample_gestures(self):
        dfs = [p for p in self.data_path.glob("*.csv")]
        dfs = [pd.read_csv(p) for p in dfs]

        filtered_dfs = []
        for idx in range(len(dfs)):
            df = dfs[idx]
            consecutive_segments_list = self.get_consecutive_segments_from_df(df)
            gesture_indices_list = self.extract_gestures(
                df=df, consecutive_segments=consecutive_segments_list
            )
            gestures_indices = [interval for interval, __ in gesture_indices_list]

            filtered_df = self.filter_df_by_index_intervals(df, gestures_indices)
            filtered_dfs.append(filtered_df)

        df = pd.concat(filtered_dfs)

        print(df.shape)

        X, y = self.data_preprocessor.preprocess_data(df)

        df = self.tuple_to_df(X, y)
        df = df.replace(LABEL_DICT)

        return df


if __name__ == "__main__":
    data_path = "../../data/gesture_data/validation_data/mandatory/"
    sampler = GestureSampler(data_path)
    df = sampler.sample_gestures()
    df.to_csv(
        "../../data/gesture_data/validation_preprocessed_mandatory_lessKeypoints.csv", index=False
    )
