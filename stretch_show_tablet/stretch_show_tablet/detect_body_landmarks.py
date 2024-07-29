import sys

import cv2
from stretch_deep_perception import body_landmark_detector as bl
from stretch_deep_perception import deep_learning_model_options as do

from . import toggleable_detection_node as tdn


def main():
    print("cv2.__version__ =", cv2.__version__)
    print("Python version (must be > 3.0):", sys.version)
    assert int(sys.version[0]) >= 3

    models_directory = do.get_directory()
    print(
        "Using the following directory for deep learning models:",
        models_directory,
    )
    use_neural_compute_stick = do.use_neural_compute_stick()

    detector = bl.BodyLandmarkDetector(
        models_directory, use_neural_compute_stick=use_neural_compute_stick
    )

    default_marker_name = "body_landmarks"
    node_name = "DetectBodyLandmarksNode"
    topic_base_name = "body_landmarks"
    fit_plane = True
    node = tdn.ToggleableDetectionNode(
        detector, default_marker_name, node_name, topic_base_name, fit_plane
    )
    node.main()
    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print('interrupt received, so shutting down')


if __name__ == "__main__":
    main()
