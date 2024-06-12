#!/usr/bin/env python3
import rclpy
import rclpy.logging

import rclpy.time
from visualization_msgs.msg import MarkerArray, Marker

import time
import json

def marker_to_dict(marker: Marker):
    marker_dict = {
        'header': {
            'stamp': {
                'secs': marker.header.stamp.sec,
                'nsecs': marker.header.stamp.nanosec
            },
            'frame_id': marker.header.frame_id
        },
        'ns': marker.ns,
        'id': marker.id,
        'type': marker.type,
        'action': marker.action,
        'pose': {
            'position': {
                'x': marker.pose.position.x,
                'y': marker.pose.position.y,
                'z': marker.pose.position.z
            },
            'orientation': {
                'x': marker.pose.orientation.x,
                'y': marker.pose.orientation.y,
                'z': marker.pose.orientation.z,
                'w': marker.pose.orientation.w
            }
        },
        'scale': {
            'x': marker.scale.x,
            'y': marker.scale.y,
            'z': marker.scale.z
        },
        'color': {
            'r': marker.color.r,
            'g': marker.color.g,
            'b': marker.color.b,
            'a': marker.color.a
        },
        'lifetime': {
            'secs': marker.lifetime.sec,
            'nsecs': marker.lifetime.nanosec
        },
        'frame_locked': marker.frame_locked,
        'points': [{'x': p.x, 'y': p.y, 'z': p.z} for p in marker.points],
        'colors': [{'r': c.r, 'g': c.g, 'b': c.b, 'a': c.a} for c in marker.colors],
        'text': marker.text,
        'mesh_resource': marker.mesh_resource,
        'mesh_use_embedded_materials': marker.mesh_use_embedded_materials
    }
    
    return marker_dict

class DataRecorder():
    def __init__(self):
        self.node = rclpy.create_node('data_recorder')
        # sub
        self.sub_face_markers = self.node.create_subscription(
            MarkerArray,
            "/faces/marker_array",
            callback=self.callback_face_markers,
            qos_profile=1
        )

        self.sub_body_markers = self.node.create_subscription(
            MarkerArray,
            "/body_landmarks/marker_array",
            callback=self.callback_body_markers,
            qos_profile=1
        )

        # state
        self.face_marker_array = None
        self.body_marker_array = None

    # callbacks
    def callback_face_markers(self, msg: MarkerArray):
        print('callback_f')
        self.face_marker_array = msg

    def callback_body_markers(self, msg: MarkerArray):
        print('callback_b')
        self.body_marker_array = msg
    
    def write_data(self, face_filename, body_filename):
        print("Writing to " + face_filename + " and " + body_filename)
        try:
            face_file = open(face_filename, 'w')
            body_file = open(body_filename, 'w')
        except:
            return
        
        face_markers = self.face_marker_array.markers
        body_markers = self.body_marker_array.markers

        face_markers = [marker_to_dict(m) for m in face_markers]
        body_markers = [marker_to_dict(m) for m in body_markers]

        json.dump(face_markers, face_file)
        json.dump(body_markers, body_file)

        face_file.close()
        body_file.close()

    # main
    def main(self):
        # config
        data_dir = "/home/hello-robot/ament_ws/src/stretch_tablet/data/"
        max_i = 20

        # loop
        i = 0
        rate = self.node.create_rate(10., self.node.get_clock())

        while rclpy.ok():
            print(self.face_marker_array)
            print(self.body_marker_array)
            if self.face_marker_array is not None and self.body_marker_array is not None:
                print('b')
                face_file = data_dir + "face_" + str(i) + ".json"
                body_file = data_dir + "body_" + str(i) + ".json"
                self.write_data(face_file, body_file)
                i += 1
            
            if i >= max_i:
                return
            
            rclpy.spin_once(self.node, timeout_sec=0.1)

def main():
    rclpy.init()
    DataRecorder().main()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
