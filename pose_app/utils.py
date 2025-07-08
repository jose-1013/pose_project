import pandas as pd
import numpy as np

CSV_COLUMN_MAPS = {
    "squat": {
        "NOSE": "Head", "LEFT_SHOULDER": "LS", "RIGHT_SHOULDER": "RS",
        "LEFT_ELBOW": "LE", "RIGHT_ELBOW": "RE",
        "LEFT_WRIST": "LW", "RIGHT_WRIST": "RW",
        "LEFT_HIP": "LH", "RIGHT_HIP": "RH",
        "LEFT_KNEE": "LK", "RIGHT_KNEE": "RK",
        "LEFT_ANKLE": "LA", "RIGHT_ANKLE": "RA"
    },
    "cat": {
        "NOSE": "Head", "LEFT_SHOULDER": "LShoulder", "RIGHT_SHOULDER": "RShoulder",
        "LEFT_ELBOW": "LElbow", "RIGHT_ELBOW": "RElbow",
        "LEFT_WRIST": "LWrist", "RIGHT_WRIST": "RWrist",
        "LEFT_HIP": "LHip", "RIGHT_HIP": "RHip",
        "LEFT_KNEE": "LKnee", "RIGHT_KNEE": "Rknee",
        "LEFT_ANKLE": "LAnkle", "RIGHT_ANKLE": "RAnkle"
    }
}

ANGLE_JOINTS = [
    ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_WRIST"),
    ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_HIP"),
    ("LEFT_KNEE", "LEFT_HIP", "LEFT_ANKLE"),
    ("RIGHT_KNEE", "RIGHT_HIP", "RIGHT_ANKLE"),
    ("LEFT_HIP", "LEFT_KNEE", "LEFT_SHOULDER"),
    ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_SHOULDER"),
    ("NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER")
]

def load_reference_pose(csv_path, row_index, pose_type):
    df = pd.read_csv(csv_path)
    column_map = CSV_COLUMN_MAPS[pose_type]
    row = df.iloc[row_index]

    pose = {}
    for joint, col_prefix in column_map.items():
        pose[joint] = (row[f"{col_prefix}_x"], row[f"{col_prefix}_y"])
    return pose

def normalize_pose(pose):
    xs, ys = zip(*pose.values())
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return {
        joint: (
            (x - x_min) / (x_max - x_min + 1e-6),
            (y - y_min) / (y_max - y_min + 1e-6)
        )
        for joint, (x, y) in pose.items()
    }

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def get_joint_angle_differences(ref_pose, user_pose):
    angle_diffs = {}
    for center, a, b in ANGLE_JOINTS:
        if all(j in ref_pose and j in user_pose for j in [a, center, b]):
            ref_angle = calculate_angle(ref_pose[a], ref_pose[center], ref_pose[b])
            user_angle = calculate_angle(user_pose[a], user_pose[center], user_pose[b])
            angle_diffs[center] = round(abs(ref_angle - user_angle), 2)
        else:
            angle_diffs[center] = None  # �먮뒗 -1 �먮뒗 "MISSING"
    return angle_diffs