
import pybullet
import gym, gym.spaces, gym.utils
import numpy as np
import os, inspect
import traceback

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet_data

class XmlBasedRobot:
    """Base class for XML-based robots."""
    self_collision = True

    def __init__(self, robot_name, action_dim, obs_dim, self_collision):
        self.parts = None
        self.objects = []
        self.jdict = None
        self.ordered_joints = None
        self.robot_body = None

        high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-high, high)
        high = np.inf * np.ones([obs_dim])
        self.observation_space = gym.spaces.Box(-high, high)

        self.robot_name = robot_name
        self.self_collision = self_collision

    def addToScene(self, bullet_client, bodies):
        """Add robot to PyBullet scene."""
        self._p = bullet_client

        if bodies is None:
            return self.parts, self.jdict, self.ordered_joints, self.robot_body

        if isinstance(bodies, int):
            bodies = [bodies]

        parts = self.parts or {}
        joints = self.jdict or {}
        ordered_joints = self.ordered_joints or []

        for i in range(len(bodies)):
            if self._p.getNumJoints(bodies[i]) == 0:
                part_name, robot_name = self._p.getBodyInfo(bodies[i])
                self.robot_name = robot_name.decode("utf8")
                part_name = part_name.decode("utf8")
                parts[part_name] = BodyPart(self._p, part_name, bodies, i, -1)
            
            for j in range(self._p.getNumJoints(bodies[i])):
                self._p.setJointMotorControl2(bodies[i], j, pybullet.POSITION_CONTROL, 
                                           positionGain=0.1, velocityGain=0.1, force=0)
                
                joint_info = self._p.getJointInfo(bodies[i], j)
                joint_name = joint_info[1].decode("utf8")
                part_name = joint_info[12].decode("utf8")

                parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)

                if part_name == self.robot_name:
                    self.robot_body = parts[part_name]

                if i == 0 and j == 0 and self.robot_body is None:
                    parts[self.robot_name] = BodyPart(self._p, self.robot_name, bodies, 0, -1)
                    self.robot_body = parts[self.robot_name]

                if joint_name[:6] == "ignore":
                    Joint(self._p, joint_name, bodies, i, j).disable_motor()
                    continue

                if joint_name[:8] != "jointfix":
                    joints[joint_name] = Joint(self._p, joint_name, bodies, i, j)
                    ordered_joints.append(joints[joint_name])
                    joints[joint_name].power_coef = 100.0

        return parts, joints, ordered_joints, self.robot_body

    def robot_specific_reset(self, bullet_client):
        """Robot-specific reset logic"""
        pass

    def reset_pose(self, position, orientation):
        """Reset robot pose"""
        self.parts[self.robot_name].reset_pose(position, orientation)

    def calc_state(self):
        """Calculate robot state"""
        pass

class MJCFBasedRobot(XmlBasedRobot):
    """Base class for MJCF-based robots."""
    def __init__(self, model_xml, robot_name, action_dim, obs_dim, self_collision=True):
        XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision)
        self.model_xml = model_xml
        self.doneLoading = 0

    def reset(self, bullet_client):
        
        self._p = bullet_client
        
        try:
            if self.objects:
                for obj in self.objects:
                    self._p.removeBody(obj)
                    
            self.ordered_joints = []
            self.objects = self._p.loadMJCF(self.model_xml)
            if not self.objects:
                print("failed generating MJCF model")
                return self.calc_state()

            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self._p, self.objects)

            print("processing robet specific reset...")
            self.robot_specific_reset(self._p)            
            state = self.calc_state()
            return state
            
        except Exception as e:
            print(f"reset robor failed: {str(e)}")
            print(f"current working Directory: {os.getcwd()}")
            print(f"model file: {self.model_xml}")
            traceback.print_exc()
            raise

    def reset_design(self, bullet_client, design):
        """Reset robot design parameters"""
        raise NotImplementedError

    def calc_potential(self):
        """Calculate robot potential"""
        return 0
class BodyPart:
    """Class representing a robot body part."""
    def __init__(self, bullet_client, body_name, bodies, body_Index, bodyPartIndex):
        self.bodies = bodies
        self._p = bullet_client
        self.body_Index = body_Index
        self.bodyPartIndex = bodyPartIndex
        self.initialPosition = self.current_position()
        self.initialOrientation = self.current_orientation()
        self.bp_pose = Pose_Helper(self)

    def state_fields_of_pose_of(self, body_id, link_id=-1):
        """Get state fields of pose."""
        if link_id == -1:
            pos, orn = self._p.getBasePositionAndOrientation(body_id)
        else:
            pos, orn, _, _, _, _ = self._p.getLinkState(body_id, link_id)
        return np.array([*pos, *orn])

    def get_position(self):
        """Get current position."""
        return self.current_position()

    def get_velocity(self):
        """Get current velocity."""
        if self.bodyPartIndex == -1:
            lin_vel, ang_vel = self._p.getBaseVelocity(self.bodies[self.body_Index])
        else:
            _, _, _, _, _, _, lin_vel, ang_vel = self._p.getLinkState(
                self.bodies[self.body_Index], 
                self.bodyPartIndex, 
                computeLinkVelocity=1
            )
        return np.array(lin_vel), np.array(ang_vel)

    def get_pose(self):
        """Get current pose."""
        return self.state_fields_of_pose_of(self.bodies[self.body_Index], self.bodyPartIndex)

    def speed(self):
        """Get linear velocity."""
        lin_vel, _ = self.get_velocity()
        return lin_vel

    def speed_angular(self):
        """Get angular velocity."""
        _, ang_vel = self.get_velocity()
        return ang_vel

    def current_velocity(self):
 
        if self.bodyPartIndex == -1:
            lin_vel, _ = self._p.getBaseVelocity(self.bodies[self.body_Index])
        else:
            _, _, _, _, _, _, lin_vel, _ = self._p.getLinkState(
                self.bodies[self.body_Index], 
                self.bodyPartIndex, 
                computeLinkVelocity=1
            )
        return np.array(lin_vel)

    def speed_angular(self):

        if self.bodyPartIndex == -1:
            _, ang_vel = self._p.getBaseVelocity(self.bodies[self.body_Index])
        else:
            _, _, _, _, _, _, _, ang_vel = self._p.getLinkState(
                self.bodies[self.body_Index], 
                self.bodyPartIndex, 
                computeLinkVelocity=1
            )
            
    def current_position(self):
        """Get current position."""
        return self.get_pose()[:3]

    def current_orientation(self):
        """Get current orientation."""
        return self.get_pose()[3:]

    def reset_position(self, position):
        """Reset position."""
        self._p.resetBasePositionAndOrientation(
            self.bodies[self.body_Index],
            position,
            self.current_orientation()
        )

    def reset_orientation(self, orientation):
        """Reset orientation."""
        self._p.resetBasePositionAndOrientation(
            self.bodies[self.body_Index],
            self.current_position(),
            orientation
        )

    def reset_velocity(self, linearVelocity=[0,0,0], angularVelocity=[0,0,0]):
        """Reset velocity."""
        self._p.resetBaseVelocity(
            self.bodies[self.body_Index],
            linearVelocity,
            angularVelocity
        )

    def reset_pose(self, position, orientation):
        """Reset pose."""
        self._p.resetBasePositionAndOrientation(
            self.bodies[self.body_Index],
            position,
            orientation
        )

    def pose(self):
        """Get pose helper."""
        return self.bp_pose

class Joint:
    """Class representing a robot joint."""
    def __init__(self, bullet_client, joint_name, bodies, body_Index, joint_index):
        self.bodies = bodies
        self._p = bullet_client
        self.body_Index = body_Index
        self.joint_index = joint_index
        self.joint_name = joint_name
             
        joint_info = self._p.getJointInfo(self.bodies[self.body_Index], self.joint_index)
        self.lower_Limit = joint_info[8]  
        self.upper_Limit = joint_info[9]  
        self.power_coef = 0
 
        self.joint_type = joint_info[2]
        self.damping = joint_info[6]
        self.friction = joint_info[7]

    def set_motor_torque(self, torque):
   
        self._p.setJointMotorControl2(
            self.bodies[self.body_Index],
            self.joint_index,
            controlMode=self._p.TORQUE_CONTROL,
            force=torque
        )

    def get_state(self):
        """Get joint state."""
        pos, vel, forces, torque = self._p.getJointState(self.bodies[self.body_Index], self.joint_index)
        return pos, vel

    def reset_current_position(self, position, velocity):
 
        self._p.resetJointState(
            self.bodies[self.body_Index],
            self.joint_index,
            targetValue=position,
            targetVelocity=velocity
        )
    def current_velocity(self):
        """Get current velocity."""
        _, vel = self.get_state()
        return vel

    def current_relative_position(self):

        pos, vel = self.get_state()
        pos_mid = 0.5 * (self.lower_Limit + self.upper_Limit) 
        return (
            2 * (pos - pos_mid) / (self.upper_Limit - self.lower_Limit),
            0.1 * vel
        )

    def reset_position(self, position, velocity):
 
        self._p.resetJointState(
            self.bodies[self.body_Index],
            self.joint_index,
            targetValue=position,
            targetVelocity=velocity
        )

    def reset_current_position(self, position, velocity):
 
        self._p.resetJointState(
            self.bodies[self.body_Index],
            self.joint_index,
            targetValue=position,
            targetVelocity=velocity
        )
        
    def disable_motor(self):

        self._p.setJointMotorControl2(
            self.bodies[self.body_Index],
            self.joint_index,
            controlMode=self._p.POSITION_CONTROL,
            targetPosition=0,
            targetVelocity=0,
            positionGain=0.1,
            velocityGain=0.1,
            force=0
        )

    def set_position(self, position):
        """Set target position."""
        self._p.setJointMotorControl2(
            self.bodies[self.body_Index],
            self.joint_index,
            self._p.POSITION_CONTROL,
            targetPosition=position
        )

    def set_velocity(self, velocity):
        """Set target velocity."""
        self._p.setJointMotorControl2(
            self.bodies[self.body_Index],
            self.joint_index,
            self._p.VELOCITY_CONTROL,
            targetVelocity=velocity
        )

    def set_torque(self, torque):
        """Set joint torque."""
        self._p.setJointMotorControl2(
            self.bodies[self.body_Index],
            self.joint_index,
            self._p.TORQUE_CONTROL,
            force=torque
        )

    def disable_motor(self):
        """Disable joint motor."""
        self._p.setJointMotorControl2(
            self.bodies[self.body_Index],
            self.joint_index,
            controlMode=self._p.POSITION_CONTROL,
            force=0
        )

    def is_at_limit(self):
        """Check if joint is at limit."""
        pos = self.current_position()
        return pos <= self.lower_Limit + 0.05 or pos >= self.upper_Limit - 0.05

class Pose_Helper:
    """Helper class for robot pose operations."""
    def __init__(self, body_part):
        self.body_part = body_part

    def xyz(self):
        """Get position."""
        return self.body_part.current_position()

    def rpy(self):
        """Get Euler angles."""
        return self.body_part._p.getEulerFromQuaternion(self.body_part.current_orientation())

    def orientation(self):
        """Get quaternion orientation."""
        return self.body_part.current_orientation()

    def get_velocity(self):
        """Get velocity."""
        return self.body_part.get_velocity()
