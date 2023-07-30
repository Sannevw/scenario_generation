import matplotlib.pyplot as plt
import os
import pybullet as p
import pybullet
import pybullet_data
import math
import numpy as np
import time


class KitchenRenderer(object):
    FOLDER_PATH = './urdf/'
    TABLE_HEIGHT = 0.94
    # WORKSPACE = np.array([np.array([-0.5,0.5])*1.5,np.array([-0.5,0.5])*2.2]) # min/max x and y coordinate
    WORKSPACE = np.array([np.array([-0.5, 0.5]) * 1.3, np.array([-0.5, 0.5]) * 2.])  # min/max x and y coordinate

    def __init__(self, gui=False):
        if gui:
            p.connect(p.GUI,
                      options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0 --fullscreen')
            p.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=90, cameraPitch=-45,
                                         cameraTargetPosition=[0, .35, 0.6])
        else:
            p.connect(p.DIRECT,
                      options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')

        self.p = p

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        self.initialize_scene()
        self.compute_camera_matrices()

    def scale_to_workspace(self, pose):
        W = self.WORKSPACE
        x_n = pose[0] * (W[0, 1] - W[0, 0]) + W[0, 0]
        y_n = pose[1] * (W[1, 1] - W[1, 0]) + W[1, 0]
        return np.array([x_n, y_n, pose[2] * 2 * np.pi])

    def compute_camera_matrices(self, top_view=False):
        if top_view:
            self.viewMatrix = p.computeViewMatrix(
                cameraEyePosition=[0.0, 0, 4.6],
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=[-1, 0, 0])

            self.projectionMatrix = p.computeProjectionMatrixFOV(
                fov=25.0,
                aspect=1.5,
                nearVal=0.1,
                farVal=30.1)
        else:
            self.viewMatrix = p.computeViewMatrix(
                cameraEyePosition=[1.0, 0, 1.6],
                cameraTargetPosition=[-0.2, 0, 0.2],
                cameraUpVector=[-1, 0, 0])

            self.projectionMatrix = p.computeProjectionMatrixFOV(
                fov=100.0,
                aspect=1.5,
                nearVal=0.1,
                farVal=30.1)

    def get_image(self, top_view=False):
        self.compute_camera_matrices(top_view)

        (_, _, px, _, _) = p.getCameraImage(width=1920, height=1080, projectionMatrix=self.projectionMatrix,
                                            viewMatrix=self.viewMatrix)  # , viewMatrix=view_matrix,
        px = np.array(px)
        if px.ndim < 3:
            px = px.reshape(1080, 1920, 4) 
        return px

    def get_image_and_depth(self, top_view=False):
        self.compute_camera_matrices(top_view)

        (_, _, px, de, _) = p.getCameraImage(width=1920, height=1080, projectionMatrix=self.projectionMatrix,
                                             viewMatrix=self.viewMatrix,
                                             renderer=pybullet.ER_TINY_RENDERER)  # , viewMatrix=view_matrix,
        return px, de

    def place_panda(self, angle=None):
        # Franka Panda robot
        urdfRootPath = pybullet_data.getDataPath()
        pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"), [0, 0, self.TABLE_HEIGHT],
                              useFixedBase=True)
        pandaEndEffectorIndex = 11
        if angle is not None:
            p.resetJointState(pandaUid, 0, angle)
            p.resetJointState(pandaUid, 1, 0.2)
            p.resetJointState(pandaUid, 2, 0.8)
            p.resetJointState(pandaUid, 3, 0.0)
            p.resetJointState(pandaUid, 4, -.5)
            p.resetJointState(pandaUid, 5, .5)

    def place_object(self):
        Kanelbulle_path = self.FOLDER_PATH + "/kanel/kanel.urdf"
        self.objectid8 = p.loadURDF(Kanelbulle_path, [0.5, 0., self.TABLE_HEIGHT], p.getQuaternionFromEuler([0, 0, 0]),
                                    globalScaling=1.7)
        texture_path = self.FOLDER_PATH + "/kanel/kanel.png"
        textureId = p.loadTexture(texture_path)
        p.changeVisualShape(self.objectid8, -1, textureUniqueId=textureId)

    def draw_workspace(self):
        T = self.TABLE_HEIGHT
        W = self.WORKSPACE
        p.addUserDebugLine([W[0, 0], W[1, 0], T], [W[0, 0], W[1, 1], T], lineColorRGB=[0, 0, 1], lineWidth=2.0,
                           lifeTime=0)
        p.addUserDebugLine([W[0, 0], W[1, 1], T], [W[0, 1], W[1, 1], T], lineColorRGB=[0, 0, 1], lineWidth=2.0,
                           lifeTime=0)
        p.addUserDebugLine([W[0, 1], W[1, 1], T], [W[0, 1], W[1, 0], T], lineColorRGB=[0, 0, 1], lineWidth=2.0,
                           lifeTime=0)
        p.addUserDebugLine([W[0, 1], W[1, 0], T], [W[0, 0], W[1, 0], T], lineColorRGB=[0, 0, 1], lineWidth=2.0,
                           lifeTime=0)

    def initialize_scene(self):

        # Load floor
        planeUid = p.loadURDF(self.FOLDER_PATH + "/floor/floor.urdf", useFixedBase=True)

        # Load kitchen
        # kitchen_path = self.FOLDER_PATH + "/kitchen/kitchen_description/urdf/kitchen_part_right_gen_convex.urdf"
        # kitchen = p.loadURDF(kitchen_path, [-2, 0., 1.477], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

        # # Load walls
        # wall_path = self.FOLDER_PATH + "/kitchen_walls/model_normalized.urdf"
        # o_wall = p.loadURDF(wall_path, [-2, -3.5, 1.477], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        # p.changeVisualShape(o_wall, -1, rgbaColor=[0.7, 0.7, 0.7, 1])
        # o_wall2 = p.loadURDF(wall_path, [-2, 3.5, 1.477], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        # p.changeVisualShape(o_wall2, -1, rgbaColor=[0.7, 0.7, 0.7, 1])
        # flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        # Load table
        tableUid = p.loadURDF("table/table.urdf", [0., 0, 0], p.getQuaternionFromEuler([0, 0, math.pi / 2]),
                              globalScaling=1.5)


class YCBObjectLoader(object):
    # settings per object: [0: urdf, 1: scale, 2: pos_offset, 3: angle_offset]
    SETTINGS = {'fork': ['./urdf/030_fork/fork.sdf', 1., np.array([0, 0, 0]), np.array([0, -np.pi / 2, 0])],
                'knife': ['./urdf/032_knife/knife.sdf', 1., np.array([0, 0, 0]), np.array([0, np.pi / 2, -np.pi / 2])],
                # np.array([np.pi/2, -np.pi / 2, -np.pi])],
                'plate': ['./urdf/plate/plate.urdf', 1., np.array([0, 0, 0]), np.array([0, 0, 0])],
                'mug': ['./urdf/025_mug/model_normalized.urdf', 1., np.array([0, 0, 0.05]), np.array([0, 0, 0])],
                'cracker': ['./urdf/003_cracker_box/model_normalized.urdf', 1., np.array([0, 0, 0.1]),
                            np.array([0, 0, 0])],
                'can': ['./urdf/002_master_chef_can/model_normalized.urdf', 1., np.array([0, 0, 0.05]),
                        np.array([0, 0, 0])],
                'bottle': ['./urdf/bottle/bottle.urdf', 0.05, np.array([0, 0, 0.1]), np.array([np.pi / 2, 0, 0])],
                'bulle': ['./urdf/kanel/kanel.urdf', 1.7, np.array([0, 0, 0.1]), np.array([0, 0, 0])]}

    def __init__(self):
        pass

    @classmethod
    def load(cls, name):
        s = cls.SETTINGS[name]
        obj = YCBObject(name, s[0], s[1:])
        if 'bulle' in name:
            texture_path = s[0].replace('.urdf', '.png')
            textureId = p.loadTexture(texture_path)
            p.changeVisualShape(obj.id, -1, textureUniqueId=textureId)
        return obj


class YCBObject(object):
    TABLE_HEIGHT = 0.94

    def __init__(self, name, path, defaults=[1., np.array([0, 0, 0]), np.array([0, 0, 0])], **kwargs):
        self.name = name

        # save defaults
        self.d_scl = defaults[0]  # global scaling
        self.d_pos = defaults[1]  # position offset
        self.d_rot = defaults[2]  # rotation offset

        # load object from file
        if '.urdf' in path:
            self.id = p.loadURDF(path, globalScaling=self.d_scl, **kwargs)
        else:
            self.id = p.loadSDF(path, globalScaling=self.d_scl, **kwargs)[0]

        # estimate center of object to center of bbox offset
        self.offset = np.array([0, 0, 0])
        self.offset = self.compute_offset()

    def compute_offset(self):
        pos = self.pose
        bbox = np.mean(self.bbox, axis=0)
        return pos[:2] - bbox

    @property
    def pose(self):
        pos, w = p.getBasePositionAndOrientation(self.id)
        euler = p.getEulerFromQuaternion(w)
        return np.array([pos[0] - self.offset[0], pos[1] - self.offset[1], euler[-1]])

    @pose.setter
    def pose(self, pose):
        new_pos = np.array([pose[0] + self.offset[0], pose[1] + self.offset[1], self.TABLE_HEIGHT]) + self.d_pos
        new_angle = np.array([0, 0, pose[2]]) + self.d_rot
        p.resetBasePositionAndOrientation(self.id, new_pos, p.getQuaternionFromEuler(new_angle))

    @property
    def bbox(self):
        col = p.getAABB(self.id, -1)
        bbox = np.array(
            [[col[0][0], col[0][1]], [col[0][0], col[1][1]], [col[1][0], col[1][1]], [col[1][0], col[0][1]]])
        return bbox

    def plot(self):
        bbox = np.concatenate((self.bbox, [self.bbox[0, :]]))
        plt.plot(bbox[:, 0], bbox[:, 1], '-b')

    def debug_plot(self):
        bbox = np.concatenate((self.bbox, [self.bbox[0, :]]))
        T = self.TABLE_HEIGHT
        for i in range(len(bbox) - 1):
            p.addUserDebugLine([bbox[i, 0], bbox[i, 1], T], [bbox[i + 1, 0], bbox[i + 1, 1], T], lineColorRGB=[0, 0, 1],
                               lineWidth=2.0, lifeTime=0)
