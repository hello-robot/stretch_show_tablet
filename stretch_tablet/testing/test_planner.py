import sophuspy as sp
from scipy.spatial.transform import Rotation as R

from stretch_tablet.human import Human
from stretch_tablet.planner import TabletPlanner

TEST_SCAN = {
    "right_eye": [-0.02386079281333919, -0.05803722054846452, 0.7124],
    "right_hip": [0.4987571135666532, -0.015583955766162247, 0.9360000000000002],
    "left_wrist": [0.4023346307026806, -0.33542882378085376, 0.821],
    "right_elbow": [0.24138610106688924, 0.11303700113913348, 0.9822],
    "left_elbow": [0.32286812358238137, -0.3681137274379407, 0.901],
    "neck": [0.11286025223914913, -0.14247963308675224, 0.8003],
    "right_wrist": [0.3762840947559113, 0.13324634459159845, 0.9179999999999999],
    "left_hip": [0.5115954453467944, -0.23255626003048543, 0.8949],
    "left_shoulder": [0.11121028978607148, -0.301615081844942, 0.7886],
    "right_ear": [0.006737427207304832, -0.05826352682069391, 0.7235],
    "left_ear": [-0.057472474495927676, -0.16064603035623903, 0.75325],
    "right_shoulder": [0.12189979012310445, -0.002999073386167257, 0.8644000000000001],
    "nose": [-0.005712366377572615, -0.09333307102307334, 0.6948],
    "left_eye": [-0.04474005305506239, -0.11568067603361239, 0.7005999999999999],
}

TEST_SCAN = {
    "nose": [-0.25424847822690794, -0.0948932145273591, 1.539],
    "neck": [-0.05267583191736175, -0.10025754829206361, 1.626],
    "right_shoulder": [-0.05371250265620282, 0.04520064334474946, 1.658],
    "right_elbow": [0.16585468701556957, 0.19035949742242653, 1.67],
    "right_wrist": [0.3596054272018213, 0.11131587345427005, 1.564],
    "left_shoulder": [-0.05486242691590267, -0.31498128299356315, 1.638],
    "left_elbow": [0.16585468701556957, -0.32296807625473867, 1.67],
    "left_wrist": [0.0, -0.0, 0.0],
    "right_hip": [0.36121491440796755, 0.04110420501803008, 1.571],
    "right_knee": [0.5430184210512212, 0.15262956110696355, 1.339],
    "right_ankle": [0.7730677260036696, 0.03914844622621244, 1.436],
    "left_hip": [0.3619046946391731, -0.23355733303529994, 1.574],
    "left_knee": [0.5276078908048086, -0.251605668986476, 1.301],
    "left_ankle": [0.7691540730944139, -0.2540233668098272, 1.321],
    "right_eye": [-0.25424847822690794, -0.09320371252588527, 1.539],
    "left_eye": [-0.0, -0.0, 0.0],
    "right_ear": [-0.26812558814962417, -0.028803909839001662, 1.623],
    "left_ear": [-0.0, -0.0, 0.0],
}

TEST_CAM = [
    [1.2736559937283527, -0.0384095063226762, 0.2600588997210326],
    [
        0.530770859840495,
        0.5714883669360042,
        -0.41583753305099946,
        -0.4677205222214006,
    ],
]


def test_planner():
    human = Human()
    human.pose_estimate.set_body_estimate_camera_frame(TEST_SCAN)
    cam = sp.SE3(R.from_quat(TEST_CAM[1]).as_matrix(), TEST_CAM[0])
    human.pose_estimate.set_body_estimate_robot_frame(TEST_SCAN, cam)

    planner = TabletPlanner()
    r = planner._get_head_shoulder_orientation(human)
    print(r)


test_planner()
