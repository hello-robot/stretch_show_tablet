import rclpy
import rclpy.logging
import rclpy.time
from std_msgs.msg import String
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


class DataRecorder:
    def __init__(self):
        self.node = rclpy.create_node("data_recorder")
        # sub
        self.sub_face_landmarks = self.node.create_subscription(
            String,
            "/faces/landmarks_3d",
            callback=self.callback_face_landmarks,
            qos_profile=1,
        )

        self.sub_body_landmarks = self.node.create_subscription(
            String,
            "/body_landmarks/landmarks_3d",
            callback=self.callback_body_landmarks,
            qos_profile=1,
        )

        # tf
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(buffer=self.tf_buffer, node=self.node)

        # state
        self.face_landmarks = None
        self.body_landmarks = None
        self.latest_camera_pose = None

    # callbacks
    def callback_face_landmarks(self, msg: String):
        # print('callback_f')
        self.face_landmarks = msg.data

    def callback_body_landmarks(self, msg: String):
        # print('callback_b')
        self.body_landmarks = msg.data

    def clear_landmarks(self):
        self.face_landmarks = None
        self.body_landmarks = None

    def write_pose_data(self, face_filename, body_filename):
        # print("Writing pose data to " + face_filename + " and " + body_filename)
        try:
            face_file = open(face_filename, "w")
            body_file = open(body_filename, "w")
        except Exception:
            return

        face_file.write(self.face_landmarks)
        body_file.write(self.body_landmarks)

        face_file.close()
        body_file.close()

    def write_camera_data(self, camera_filename):
        # print("Writing camera data to " + camera_filename)
        try:
            camera_file = open(camera_filename, "w")
        except Exception:
            return

        camera_file.write(str(self.latest_camera_pose))

        camera_file.close()

    # main
    def main(self):
        # config
        data_dir = "/home/hello-robot/ament_ws/src/stretch_tablet/data/"
        max_i = 20

        # loop
        i = 0
        # rate = self.node.create_rate(10.0, self.node.get_clock())

        while rclpy.ok():
            # print(self.face_marker_array)
            # print(self.body_marker_array)
            if self.face_landmarks is not None and self.body_landmarks is not None:
                # get camera pose
                try:
                    t = self.tf_buffer.lookup_transform(
                        "camera_color_optical_frame", "odom", rclpy.time.Time()
                    )

                    camera_pos = t.transform.translation
                    camera_ori = t.transform.rotation
                    self.latest_camera_pose = [
                        [camera_pos.x, camera_pos.y, camera_pos.z],
                        [
                            camera_ori.x,
                            camera_ori.y,
                            camera_ori.z,
                            camera_ori.w,
                        ],
                    ]
                except Exception as ex:
                    print(ex)

                # set up data
                face_file = data_dir + "face_" + str(i) + ".json"
                body_file = data_dir + "body_" + str(i) + ".json"
                camera_file = data_dir + "camera_" + str(i) + ".json"
                self.write_pose_data(face_file, body_file)
                self.write_camera_data(camera_file)
                self.clear_landmarks()

                i += 1

            if i >= max_i:
                return

            rclpy.spin_once(self.node, timeout_sec=0.1)


def main():
    rclpy.init()
    DataRecorder().main()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
