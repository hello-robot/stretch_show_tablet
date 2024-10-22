import json
from enum import Enum

import cv2
import message_filters
import rclpy
import ros2_numpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String
from std_srvs.srv import SetBool
from stretch_deep_perception import detection_2d_to_3d as d2
from stretch_deep_perception.detection_node import DetectionNode
from visualization_msgs.msg import MarkerArray


class Camera(Enum):
    HEAD = 1
    GRIPPER = 2


class ToggleableDetectionNode(DetectionNode):
    def __init__(
        self,
        detector,
        default_marker_name,
        node_name,
        topic_base_name,
        fit_plane,
        min_box_side_m=None,
        max_box_side_m=None,
        modify_3d_detections=None,
    ):
        super().__init__(
            detector,
            default_marker_name,
            node_name,
            topic_base_name,
            fit_plane,
            min_box_side_m,
            max_box_side_m,
            modify_3d_detections,
        )

        self.toggled_on = False

    def toggle_callback(self, req: SetBool.Request, res: SetBool.Response):
        self.logger.info(f"Toggle detection service: {req.data}")
        self.toggled_on = req.data
        res.success = True
        res.message = "Detection toggled on" if req.data else "Detection toggled off"
        self.logger.info(f"Toggle detection service returning {res}")
        return res

    def image_callback(self, ros_rgb_image, ros_depth_image, rgb_camera_info):
        if not self.toggled_on:
            return

        self.rgb_image = ros2_numpy.numpify(ros_rgb_image)
        self.rgb_image_timestamp = ros_rgb_image.header.stamp
        self.depth_image = ros2_numpy.numpify(ros_depth_image)
        self.depth_image_timestamp = ros_depth_image.header.stamp
        self.camera_info = rgb_camera_info
        self.image_count = self.image_count + 1

        # OpenCV expects bgr images, but numpify by default returns rgb images.
        self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)

        # TODO: Check if this operation can be handled by a ROS 2 method instead of
        # doing it manually
        ############
        time_diff_nanosec = abs(
            self.rgb_image_timestamp.nanosec - self.depth_image_timestamp.nanosec
        )
        time_diff_sec = abs(
            self.rgb_image_timestamp.sec - self.depth_image_timestamp.sec
        )
        time_diff = time_diff_sec + time_diff_nanosec * 0.000001
        ############

        if time_diff > 0.0001:
            self.logger.warn(
                (
                    "The rgb image and the depth image were not taken at the same time. "
                    f"The time difference between their timestamps = {time_diff} s"
                ),
                throttle_duration_sec=1.0,
            )

        # Rotate the image by 90deg to account for camera
        # orientation. In the future, this may be performed at the
        # image source.
        detection_box_image = cv2.rotate(self.rgb_image, cv2.ROTATE_90_CLOCKWISE)

        debug_input = False
        if debug_input:
            self.logger.info("DetectionNode.image_callback: received an image!")
            self.logger.info(
                "DetectionNode.image_callback: detection_box_image.shape =",
                detection_box_image.shape,
            )
            cv2.imwrite(
                "./output_images/deep_learning_input_"
                + str(self.image_count).zfill(4)
                + ".png",
                detection_box_image,
            )

        debug_output = False
        detections_2d, output_image = self.detector.apply_to_image(
            detection_box_image, draw_output=debug_output
        )

        if debug_output:
            self.logger.info(
                "DetectionNode.image_callback: processed image with deep network!"
            )
            self.logger.info(
                f"DetectionNode.image_callback: output_image.shape = {output_image.shape}"
            )
            cv2.imwrite(
                "./output_images/deep_learning_output_"
                + str(self.image_count).zfill(4)
                + ".png",
                output_image,
            )

        if output_image is not None:
            output_image = ros2_numpy.msgify(Image, output_image, encoding="bgr8")
            if output_image is not None:
                self.visualize_object_detections_pub.publish(output_image)

        detections_3d = d2.detections_2d_to_3d(
            detections_2d,
            self.rgb_image,
            self.camera_info,
            self.depth_image,
            fit_plane=self.fit_plane,
            min_box_side_m=self.min_box_side_m,
            max_box_side_m=self.max_box_side_m,
        )

        if self.modify_3d_detections is not None:
            detections_3d = self.modify_3d_detections(detections_3d)

        self.marker_collection.update(detections_3d, self.rgb_image_timestamp)

        if len(detections_3d) > 0:
            landmarks_3d = detections_3d[0]["landmarks_3d"]
        else:
            landmarks_3d = None
        landmark_string = String()
        landmark_string.data = json.dumps(landmarks_3d)
        self.landmark_3d_pub.publish(landmark_string)

        marker_array = self.marker_collection.get_ros_marker_array(
            self.landmark_color_dict
        )
        include_axes = True
        include_z_axes = False
        axes_array = None
        axes_scale = 4.0
        if include_axes or include_z_axes:
            axes_array = self.marker_collection.get_ros_axes_array(
                include_z_axes, include_axes, axes_scale=axes_scale
            )

        if self.publish_marker_point_clouds:
            for marker in self.marker_collection:
                marker_points = marker.get_marker_point_cloud()
                self.add_point_array_to_point_cloud(marker_points)
                publish_plane_points = False
                if publish_plane_points:
                    plane_points = marker.get_plane_fit_point_cloud()
                    self.add_point_array_to_point_cloud(plane_points)
            self.publish_point_cloud()
        self.visualize_markers_pub.publish(marker_array)
        if axes_array is not None:
            self.visualize_axes_pub.publish(axes_array)

    def main(self, camera: Camera = Camera.HEAD):
        rclpy.init()
        self.node = rclpy.create_node(self.node_name)
        name = self.node.get_name()
        self.node.get_logger().info("{0} started".format(name))

        if camera == Camera.HEAD:
            topic_prefix = "/camera/"
            rgb_topic = topic_prefix + "color/image_raw"
            depth_topic = topic_prefix + "aligned_depth_to_color/image_raw"
        elif camera == Camera.GRIPPER:
            topic_prefix = "/gripper_camera/"
            # rgb_topic = topic_prefix + 'color/image_rect_raw'
            rgb_topic = topic_prefix + "image_raw"
            # depth_topic = topic_prefix + 'aligned_depth_to_color/image_raw'
            depth_topic = topic_prefix + "depth/image_rect_raw"
        else:
            raise ValueError

        self.rgb_topic_name = rgb_topic
        self.rgb_image_subscriber = message_filters.Subscriber(
            self.node,
            Image,
            self.rgb_topic_name,
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self.depth_topic_name = depth_topic
        self.depth_image_subscriber = message_filters.Subscriber(
            self.node,
            Image,
            self.depth_topic_name,
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self.camera_info_subscriber = message_filters.Subscriber(
            self.node,
            CameraInfo,
            topic_prefix + "color/camera_info",
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self.synchronizer = message_filters.TimeSynchronizer(
            [
                self.rgb_image_subscriber,
                self.depth_image_subscriber,
                self.camera_info_subscriber,
            ],
            10,
        )
        self.synchronizer.registerCallback(self.image_callback)

        self.visualize_markers_pub = self.node.create_publisher(
            MarkerArray, "/" + self.topic_base_name + "/marker_array", 1
        )
        self.visualize_axes_pub = self.node.create_publisher(
            MarkerArray, "/" + self.topic_base_name + "/axes", 1
        )
        self.visualize_point_cloud_pub = self.node.create_publisher(
            PointCloud2, "/" + self.topic_base_name + "/point_cloud2", 1
        )

        self.visualize_object_detections_pub = self.node.create_publisher(
            Image,
            "/" + self.topic_base_name + "/color/image_with_bb",
            qos_profile=QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )

        self.landmark_3d_pub = self.node.create_publisher(
            String, "/" + self.topic_base_name + "/landmarks_3d", 1
        )

        # toggle
        self.toggle_service = self.node.create_service(
            SetBool,
            "/" + self.topic_base_name + "/detection/toggle",
            self.toggle_callback,
        )

        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            self.logger.info("interrupt received, so shutting down")

        self.node.destroy_node()
        rclpy.shutdown()
