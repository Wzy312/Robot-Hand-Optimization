import pybullet as p
import pybullet_data
from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from .contact_model import ContactModel
import gym, gym.spaces, gym.utils
import numpy as np
from .robot_locomotors import AllegroHand
import os, inspect
import time
import traceback
from gym import spaces
from pybullet_utils import bullet_client

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

class HandBaseBulletEnv(MJCFBaseBulletEnv):
    def create_single_player_scene(self, bullet_client):
        try:
            self.timestep = 1.0/480.0
            self.frame_skip = 1
            self.gravity = -9.81
            self.target_object = bullet_client.loadURDF(
                os.path.join(currentdir, "assets/sphere/sphere.urdf"),
                basePosition=self.initial_object_position,
                globalScaling=0.04
            )
            bullet_client.changeDynamics(
                self.target_object, 
                -1,
                mass=0.1,
                lateralFriction=1.0,
                spinningFriction=0.1,
                rollingFriction=0.1,
                restitution=0.0,
                contactStiffness=1000,
                collisionMargin=0.00001,
                contactDamping=50,
                linearDamping=0.8,
                angularDamping=0.8,
                localInertiaDiagonal=[0.001, 0.001, 0.001],  
                contactProcessingThreshold=0.0001
            )
            
            print("Scene created successfully")
            return True
            
        except Exception as e:
            print(f"Scene creation failed: {str(e)}")
            traceback.print_exc()
            return False
            
    def reset(self):
        print("\n====== Environment reset started ======")
        try:
            # Reset basic state
            self.current_step = 0
            self.gravity_enabled = False
            self.grasp_started = False 
            self.contact_count = 0

            # Use safe reset method
            if hasattr(self, '_p'):
                try:
                    # Save target object ID before removing all objects
                    target_obj_id = getattr(self, 'target_object', -1)
                    
                    # Get all object IDs
                    all_bodies = []
                    for i in range(self._p.getNumBodies()):
                        all_bodies.append(self._p.getBodyUniqueId(i))
                    
                    print(f"Number of objects in current scene: {len(all_bodies)}")
                    
                    # Remove objects one by one to avoid batch reset
                    for body_id in all_bodies:
                        try:
                            # Skip target object, handle separately later
                            if body_id == target_obj_id:
                                continue
                                
                            self._p.removeBody(body_id)
                            print(f"Successfully removed object ID: {body_id}")
                        except Exception as e:
                            print(f"Error removing object {body_id}: {str(e)}")
                    
                    # Now safely reset simulation
                    print("Performing complete simulation reset...")
                    self._p.resetSimulation()
                    
                except Exception as e:
                    print(f"Error during simulation reset: {str(e)}")
                    import traceback
                    traceback.print_exc()

            # Initialize physics parameters    
            self._p.setTimeStep(1./240)
            self._p.setGravity(0, 0, 0)
            
            # Reset robot first to ensure proper initialization
            print("Resetting robot...")
            state = self.robot.reset(self._p)
            print(f"Robot reset complete, state vector length: {len(state)}")

            # Sphere position - place at appropriate height
            self.initial_object_position = [0, 0, 0.1]
            try:
                print(f"Loading target object at position: {self.initial_object_position}")
                self.target_object = self._p.loadURDF(
                    os.path.join(currentdir, "assets/sphere/sphere.urdf"),
                    basePosition=self.initial_object_position,
                    globalScaling=1
                )
                print(f"Load successful, object ID: {self.target_object}")
            except Exception as e:
                print(f"Failed to load sphere: {str(e)}")
                # Create simplified sphere
                sphereRadius = 0.045
                colSphereId = self._p.createCollisionShape(self._p.GEOM_SPHERE, radius=sphereRadius)
                self.target_object = self._p.createMultiBody(
                    baseMass=0.1,
                    baseCollisionShapeIndex=colSphereId,
                    basePosition=self.initial_object_position,
                    useMaximalCoordinates=False
                )
                print(f"Created simplified sphere, ID: {self.target_object}")

             # Configure sphere physics properties
            self._p.changeDynamics(
                self.target_object, 
                -1,
                mass=0.1,
                lateralFriction=1.0,
                spinningFriction=0.1,
                rollingFriction=0.1,
                restitution=0.0,
                contactStiffness=1000,
                collisionMargin=0.00001,
                contactDamping=50,
                linearDamping=0.8,
                angularDamping=0.8,
                localInertiaDiagonal=[0.001, 0.001, 0.001],  # Reduce inertia
                contactProcessingThreshold=0.0001
            )

            # Hand position should be above the sphere, palm facing down
            hand_init_pos = [
                self.initial_object_position[0],      # X coordinate aligned with sphere
                self.initial_object_position[1],      # Y coordinate aligned with sphere
                self.initial_object_position[2] + 0.3 # Z coordinate: 15cm above sphere
            ]
            
            # Correction: Use quaternion ensuring palm faces down
            hand_init_orn = [0, 1, 0, 0]            
            print(f"Setting hand position: {hand_init_pos}, orientation: {hand_init_orn}")           
            # Reset robot position
            self.robot.robot_body.reset_pose(hand_init_pos, hand_init_orn)
            
            # Verify position is correctly set
            actual_pos, actual_orn = self._p.getBasePositionAndOrientation(
                self.robot.robot_body.bodies[self.robot.robot_body.body_Index]
            )
            print(f"Actual hand position: {actual_pos}, orientation: {actual_orn}")

            # Set fingers to pre-grasp posture - ensure fingers bend downward
            self.relax_hand()

            # Stabilize simulation
            print("Stabilizing simulation...")
            for _ in range(100):
                self._p.stepSimulation()

            # Adjust camera
            self._p.resetDebugVisualizerCamera(
                cameraDistance=0.6,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.2]
            )
                
            print("Environment reset successful")
            return state

        except Exception as e:
            print(f"\nEnvironment reset failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(self.robot.observation_space.shape)

    def relax_hand(self):
    
        pass  
        
    def _approach_phase(self):
   
        return False 
        
    def _grasp_phase(self, action):
     
        return action  
        
    def _lift_phase(self):
      
        pass  
        
    def _compute_reward(self, object_pos, linear_vel, contact_count, contact_points):

        pass
    
    def camera_adjust(self):
        """Adjust camera view to ensure both target sphere and robot hand are in view"""
        try:   # Get actual position of target sphere
            object_pos, _ = self._p.getBasePositionAndOrientation(self.target_object)
            # Get actual position of robot hand body
            hand_pos, _ = self._p.getBasePositionAndOrientation(self.robot.robot_body.bodies[0])
            # Calculate observation target point as midpoint between them
            target_pos = [
                (hand_pos[0] + object_pos[0]) / 2,
                (hand_pos[1] + object_pos[1]) / 2,
                (hand_pos[2] + object_pos[2]) / 2
            ]
            # Adjust camera parameters
            self._p.resetDebugVisualizerCamera(
                cameraDistance=0.8,  # Increase camera distance
                cameraYaw=90,
                cameraPitch=-40,     # Slightly lower view angle for better grasping observation
                cameraTargetPosition=target_pos
            )
        except Exception as e:
            print(f"Camera went wrong: {str(e)}")
    
    def _check_termination(self):
        """Check whether current episode should be terminated"""
        # Get object position
        object_pos, _ = self._p.getBasePositionAndOrientation(self.target_object)
        
        # Get contact information
        contacts = self._p.getContactPoints(
            self.robot.robot_body.bodies[0],
            self.target_object
        )
        
        # Calculate grasp quality
        grasp_quality = 0
        if contacts:
            contact_model_result = self.robot.contact_model.compute_contact_model(
                self._p,
                self.robot.robot_body.bodies[0],
                self.target_object
            )
            if contact_model_result:
                grasp_quality = self.robot.contact_model.evaluate_grasp_quality(
                    contact_model_result
                )
        
        # Termination conditions:
        # 1. Object successfully lifted to sufficient height with stable grasp
        if object_pos[2] > 0.25 and grasp_quality > 0.3:
            return True, "Grasp successful"
            
        # 2. Maximum step limit reached
        if self.current_step >= self.max_steps:
            return True, "Maximum steps reached"
                
        # 3. Object dropped below ground
        if object_pos[2] < 0:
            return True, "Object dropped"
            
        return False, ""
    
    def reset_design(self, design):
        """Base class reset_design method to ensure consistent parameter signature"""
        if self.robot is not None:
            self._p.resetSimulation()
            self.scene = None
            self.robot.reset_design(self._p, design)
            return self.reset()
        return None
        
    def render_camera_image(self, resolution=(320, 240)):
        """Render camera image"""
        width, height = resolution
        
        # Get current view information
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=0.5,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        
        # Set projection parameters
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width)/height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Render image
        _, _, rgba, _, _ = self._p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self._p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to RGB format
        rgb = rgba[:, :, :3] / 255.0
        return rgb
        
    def __del__(self):
        """Destructor"""
        if hasattr(self, '_p'):
            try:
                self._p.disconnect()
            except Exception as e:
                print(f"Error disconnecting PyBullet: {e}")

class AllegroHandBulletEnv(HandBaseBulletEnv):
    def __init__(self, render=False, design=None):
        """Initialize AllegroHand environment with optimized finger configuration and physics parameters"""
        print("Initializing AllegroHandBulletEnv...")
        
        # Create robot instance
        self.robot = AllegroHand(design)
        self._render = render
        
        # Call parent class initialization
        super().__init__(self.robot, render)        
        import pybullet as p
        if not hasattr(self, 'target_object') or self.target_object is None:
            # Set a default value in advance, will be updated in reset()
            self.target_object = -1  
        if not hasattr(self, '_p'):
            if render:
                self.physicsClientId = p.connect(p.GUI)
            else:
                self.physicsClientId = p.connect(p.DIRECT)
            self._p = p
        
        # Modification: Redefine active grasp joints to ensure only controllable joints are included
        self.active_grasp_joints = [
            1, 3, 6, 8,     # if
            11, 13, 16, 18, # mf
            21, 23, 26, 28, # rf
            32, 34, 36, 38  # th
        ]
        
        # Initialize environment parameters
        self.current_step = 0
        self.max_steps = 1000
        self.gravity_enabled = False 
        self.stable_grasp = False
        self.contact_count = 0
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=[self.robot.observation_dim],
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            -1, 1,
            shape=(len(self.active_grasp_joints),),
            dtype=np.float32
        )
        
        # Modification: Optimize grasp parameters
        self.grasp_distance_margin = 0.005  
        self.grasp_time_limit = 1.5        
        self.target_grasp_velocity = 0.05   
        self.max_grasp_force = 35.0        
        self.target_object_position = [0, 0, 0.1]  
        self.start_pos_x = 0.0
        self.start_pos_y = 0.0
        self.start_pos_z = 0.6 
        self.finger_groups = {
            'thumb': [32, 34, 36, 38],  
            'index': [1, 3, 6, 8],         
            'middle': [11, 13, 16, 18],    
            'ring': [21, 23, 26, 28]       
        }
        self.grasp_angles = {          
            32: 0.3,  
            34: 0.1,   
            36: 0.9,  
            38: 0.9,      
            1: 0.1,    
            3: 0.6,  
            6: 0.9,   
            8: 0.8,               
            11: 0.1,   
            13: 0.6,  
            16: 0.9,   
            18: 0.8,     
            21: 0.1,  
            23: 0.6,   
            26: 0.9,  
            28: 0.8,   
        }       
        print("Initialization complete - grasp parameters optimized")

    def relax_hand(self):
        
        body_id = self.robot.robot_body.bodies[self.robot.robot_body.body_Index]
        preset_angles = {

            "rfj0_twist": 0.15,    
            "rfj1": -0.65,            
            "rfj2": 0.2,          
            "rfj3":  0.2,           

            "mfj0_twist": 0.00,      
            "mfj1":  -0.65,           
            "mfj2":  0.2,            
            "mfj3":  0.20,           

            "ffj0_twist": 0.15,     
            "ffj1":  -0.65,            
            "ffj2":  0.2,            
            "ffj3":  0.20,           
            
            "thj1_twist": 2.1,     
            "thj1_bend":  0.4,              
            "thj2":  0.06,        
            "thj3":  0.05        
        }
        
        for joint_name, target_angle in preset_angles.items():
            found = False
            for joint_idx in range(self._p.getNumJoints(body_id)):
                info = self._p.getJointInfo(body_id, joint_idx)
                if info[1].decode('utf-8') == joint_name:
                    # Get joint limits
                    lower_limit = info[8]
                    upper_limit = info[9]
                    
                    # Ensure angle is within valid range
                    clamped_angle = max(lower_limit, min(upper_limit, target_angle))
                    
                    # Reset joint state
                    self._p.resetJointState(body_id, joint_idx, targetValue=clamped_angle)
                    
                    # Use greater force to ensure fingers maintain set posture
                    self._p.setJointMotorControl2(
                        body_id,
                        joint_idx,
                        controlMode=self._p.POSITION_CONTROL,
                        targetPosition=clamped_angle,
                        force=30.0,  # Increase force
                        positionGain=0.8,
                        velocityGain=0.3
                    )
                    
                    print(f"Joint '{joint_name}' (index {joint_idx}) set to {clamped_angle:.2f}")
                    found = True
                    break
                    
            if not found:
                print(f"Warning: Joint '{joint_name}' not found")
        
        # Simulate more steps to stabilize finger posture
        for _ in range(10):
            self._p.stepSimulation()

    def _analyze_grasping_potential(self, palm_id):
        print("\n===== Grasp Potential Analysis =====")
        
        # Get sphere position
        ball_pos, _ = self._p.getBasePositionAndOrientation(self.target_object)
        ball_radius = 0.045 
        
        for finger_name, joints in self.finger_groups.items():
            # Only check fingertip joints
            distal_joint = joints[-1]  # Assume the last one is the fingertip joint
            
            # Get joint information
            joint_info = self._p.getJointInfo(palm_id, distal_joint)
            joint_name = joint_info[1].decode('utf-8')
            
            # Get joint position
            link_state = self._p.getLinkState(palm_id, distal_joint)
            tip_pos = link_state[0]  # Position in world coordinate system
            
            # Calculate distance to sphere center
            distance = np.linalg.norm(np.array(tip_pos) - np.array(ball_pos))
            
            # Calculate fingertip distance to sphere surface
            surface_distance = distance - ball_radius
            
            print(f"  {finger_name} fingertip({joint_name}) to sphere: {surface_distance:.3f}m")
            
            # Suggest grasp angle
            if surface_distance > 0:
                # Calculate direction from fingertip to sphere center
                direction = np.array(ball_pos) - np.array(tip_pos)
                direction = direction / np.linalg.norm(direction)
                
                # Suggested bending angle - simple linear mapping
                suggested_angle = min(1.0, surface_distance * 3.0)
                print(f"  Suggested {finger_name} fingertip bending angle: {suggested_angle:.2f} rad")
        
        print("=======================")

    def _detect_contacts_in_detail(self):
        
        palm_id = self.robot.robot_body.bodies[self.robot.robot_body.body_Index]
        
        # Get all possible contact points
        all_contact_points = self._p.getContactPoints()
        
        # Classify contact points
        hand_object_contacts = []
        hand_self_contacts = []
        object_environment_contacts = []
        
        for cp in all_contact_points:
            bodyA = cp[1]
            bodyB = cp[2]
            
            if (bodyA == palm_id and bodyB == self.target_object) or (bodyA == self.target_object and bodyB == palm_id):
                hand_object_contacts.append(cp)
            elif bodyA == palm_id and bodyB == palm_id:
                hand_self_contacts.append(cp)
            elif bodyA == self.target_object or bodyB == self.target_object:
                object_environment_contacts.append(cp)
        
        print(f"\nDetailed contact analysis:")
        print(f"- Hand-sphere contacts: {len(hand_object_contacts)}")
        print(f"- Hand self-contacts: {len(hand_self_contacts)}")
        print(f"- Sphere-environment contacts: {len(object_environment_contacts)}")
        
        if hand_object_contacts:
            print("\nHand-sphere contact details:")
            for i, cp in enumerate(hand_object_contacts[:3]):  # Only show first 3
                link_idx = cp[3] if cp[1] == palm_id else cp[4]
                link_name = "Unknown"
                try:
                    link_name = self._p.getJointInfo(palm_id, link_idx)[12].decode('utf-8')
                except:
                    pass
                    
                distance = cp[8]
                normal_force = cp[9]
                print(f"  Contact point {i+1}: Link={link_name}, Distance={distance:.4f}, Normal force={normal_force:.2f}N")
        
        return len(hand_object_contacts)        
    
    def _approach_phase(self, params=None):
   
        # Use default parameters
        if params is None:
            approach_speed = 0.25
            approach_distance = 0.1
            approach_force = 0.6
            vertical_threshold = 0.135 # Stricter vertical distance threshold
            horizontal_threshold = 0.11 # Stricter horizontal distance threshold
            speed_threshold = 0.05  # Stricter speed threshold
            stability_time = 20  # Simulation steps required to maintain stable state
        else:
            # print(f"Using strategy-provided approach parameters: {params}")
            approach_speed = params.get('approach_speed', 0.25)
            approach_distance = params.get('approach_distance', 0.07)
            approach_force = params.get('approach_force', 0.6)
            vertical_threshold = params.get('vertical_threshold', 0.135)
            horizontal_threshold = params.get('horizontal_threshold', 0.11)
            speed_threshold = params.get('speed_threshold', 0.05)
            stability_time = params.get('stability_time', 20)
        
        # Get sphere and palm positions
        ball_pos, _ = self._p.getBasePositionAndOrientation(self.target_object)
        palm_id = self.robot.robot_body.bodies[self.robot.robot_body.body_Index]
        current_pos, current_orn = self._p.getBasePositionAndOrientation(palm_id)
        
        # Get current velocity
        lin_vel, ang_vel = self._p.getBaseVelocity(palm_id)
        speed = np.linalg.norm(lin_vel)
        
        # New: Track stability state counter
        if not hasattr(self, '_stability_counter'):
            self._stability_counter = 0
        
        grasp_offset = approach_distance  # Use parameterized vertical height offset
        thumb_offset = 0.04  # Small offset in X direction, away from thumb
        
        # Adjust target position considering thumb position
        desired_pos = [
            ball_pos[0] + thumb_offset,  # X-axis offset, away from thumb
            ball_pos[1],                 # Y coordinate aligned
            ball_pos[2] + grasp_offset   # Specified cm above sphere
        ]
        
        # Calculate distance and direction
        error = np.array(desired_pos) - np.array(current_pos)
        distance = np.linalg.norm(error)
        
        # Calculate vertical and horizontal distances in detail for grasp judgment
        vertical_distance = abs(current_pos[2] - desired_pos[2])
        horizontal_distance = np.sqrt((current_pos[0] - desired_pos[0])**2 + 
                                    (current_pos[1] - desired_pos[1])**2)
        
        if self.current_step % 1000 == 0:
            print(f"\n[Approach phase] Status info:")
            print(f"- Sphere position: {[f'{x:.3f}' for x in ball_pos]}")
            print(f"- Palm position: {[f'{x:.3f}' for x in current_pos]}")
            print(f"- Target position: {[f'{x:.3f}' for x in desired_pos]}")
            print(f"- Vertical distance: {vertical_distance:.3f}m, Horizontal distance: {horizontal_distance:.3f}m")
            print(f"- Current speed: {speed:.3f}m/s, Stability count: {self._stability_counter}/{stability_time}")
            print(f"- Approach parameters: Speed={approach_speed:.3f}, Distance={approach_distance:.3f}, Force={approach_force:.3f}")
        
        # Use PD controller to control palm movement
        kp = approach_speed  # Use parameterized position proportional gain
        kd = 2  # Damping proportional gain
        
        # Calculate PD control force
        p_term = kp * error
        d_term = -kd * np.array(lin_vel)
        force = p_term + d_term
        
        # Limit force magnitude
        max_force = approach_force  # Use parameterized maximum force
        force_norm = np.linalg.norm(force)
        if force_norm > max_force:
            force = force / force_norm * max_force
            
        # Apply force - only when stable state not reached
        if distance > 0.01 and self._stability_counter < stability_time:
            self._p.applyExternalForce(
                palm_id,
                -1,
                forceObj=force.tolist(),
                posObj=current_pos,
                flags=self._p.WORLD_FRAME
            )
            
            # Suppress rotation
            if np.linalg.norm(ang_vel) > 0.01:
                anti_torque = [-ang_vel[0] * 2.0, -ang_vel[1] * 2.0, -ang_vel[2] * 2.0]
                self._p.applyExternalTorque(
                    palm_id,
                    -1,
                    torqueObj=anti_torque,
                    flags=self._p.WORLD_FRAME
                )
 
        if vertical_distance < vertical_threshold and horizontal_distance < horizontal_threshold and speed < speed_threshold:
            # Position and speed conditions met, increase stability count
            self._stability_counter += 1
            
            # If stable for long enough, can start grasping
            if self._stability_counter >= stability_time:
                print(f"\n*** Palm has stably reached pre-grasp position ({self._stability_counter} steps), ready to enter grasp phase ***")
                print(f"- Final position: Vertical distance={vertical_distance:.4f}m, Horizontal distance={horizontal_distance:.4f}m")
                print(f"- Final speed: {speed:.4f}m/s")
                
                # Reset stability counter
                self._stability_counter = 0
                
                # Stabilize for a few steps
                self._p.resetBaseVelocity(palm_id, [0, 0, 0], [0, 0, 0])
                for _ in range(10):
                    self._p.stepSimulation()
                    
                return True
        else:
            # Conditions not met, reset stability counter
            self._stability_counter = 0
        
        return False
        
    def _grasp_phase(self, params=None):
        """      
        Args:
            params: Grasp parameter dictionary, if None use default values
        """
        # Use default parameters
        if params is None:
            finger_force = 30.0
            position_gain = 0.3
            velocity_gain = 0.1
            finger_closure_rate = 0.05
        else:
            # Modification: Add parameter validation to ensure strategy output is used correctly
            # print(f"Using strategy-provided grasp parameters: {params}")
            finger_force = params.get('finger_force', 30.0)
            position_gain = params.get('position_gain', 0.3)
            velocity_gain = params.get('velocity_gain', 0.1)
            finger_closure_rate = params.get('finger_closure_rate', 0.05)
        
        # Get current state
        palm_id = self.robot.robot_body.bodies[self.robot.robot_body.body_Index]
        obj_pos, _ = self._p.getBasePositionAndOrientation(self.target_object)
        palm_pos, _ = self._p.getBasePositionAndOrientation(palm_id)
        
        # print(f"Grasp phase: Sphere position {[f'{x:.3f}' for x in obj_pos]}, Palm position {[f'{x:.3f}' for x in palm_pos]}")
        # print(f"Grasp parameters: Force={finger_force:.2f}, Position gain={position_gain:.2f}, Velocity gain={velocity_gain:.2f}, Closure rate={finger_closure_rate:.3f}")
        
        # Check finger distance to sphere
        min_distance = float('inf')
        for joint_idx in range(self._p.getNumJoints(palm_id)):
            link_state = self._p.getLinkState(palm_id, joint_idx)
            if link_state:
                link_pos = link_state[0]
                dist = np.linalg.norm(np.array(link_pos) - np.array(obj_pos))
                min_distance = min(min_distance, dist)
        
        # print(f"Nearest finger to sphere distance: {min_distance:.4f}m")
        
        # Target grasp angles
        grasp_angles = {
            # Index finger - target angles
            "rfj0_twist": 0.25,       # Keep slight twist
            "rfj1": 0.6,             # Base bend
            "rfj2": 0.8,             # Modified to positive to avoid sudden change
            "rfj3": 1,              # Modified to positive to avoid sudden change
            
            # Middle finger - target angles
            "mfj0_twist": 0.0,       # Keep neutral
            "mfj1": 0.5,             # Base bend
            "mfj2": 0.8,             # Modified to positive to avoid sudden change
            "mfj3": 1,              # Modified to positive to avoid sudden change
            
            # Ring finger - target angles
            "ffj0_twist": -0.25,      # Slight inward twist
            "ffj1": 0.6,             # Base bend
            "ffj2": 0.8,             # Modified to positive to avoid sudden change
            "ffj3": 1,              # Modified to positive to avoid sudden change
   
            "thj1_twist": 2.2,       # Keep neutral
            "thj1_bend": 0.6,        # Larger bend, but not to limit
            "thj2": 0.6,             # Larger bend, but not to limit
            "thj3": 0.6              # Larger bend, but not to limit
        }
        
        # Modification: Track processed joints and their angles
        processed_joints = []
        
        # Smoothly apply grasp angles - Key improvement: Use smaller force, gradual closure
        for joint_name, target_angle in grasp_angles.items():
            for joint_idx in range(self._p.getNumJoints(palm_id)):
                info = self._p.getJointInfo(palm_id, joint_idx)
                if info[1].decode('utf-8') == joint_name:
                    # Get joint limits
                    lower_limit = info[8]
                    upper_limit = info[9]
                    
                    # Ensure angle is within valid range
                    clamped_angle = max(lower_limit, min(upper_limit, target_angle))
                    
                    # Get current joint angle
                    current_pos = self._p.getJointState(palm_id, joint_idx)[0]
                    
                    # Calculate smooth transition target
                    # Move no more than set percentage of distance per step to avoid sudden change
                    blend_factor = finger_closure_rate  # [Modification] Use parameterized closure rate
                    smooth_target = current_pos + (clamped_angle - current_pos) * blend_factor
                    
                    # Apply moderate force position control
                    self._p.setJointMotorControl2(
                        palm_id,
                        joint_idx,
                        controlMode=self._p.POSITION_CONTROL,
                        targetPosition=smooth_target,
                        force=finger_force,  # [Modification] Use parameterized force
                        positionGain=position_gain,  # [Modification] Use parameterized position gain
                        velocityGain=velocity_gain   # [Modification] Use parameterized velocity gain
                    )
                    
                    # Record processed joints
                    processed_joints.append((joint_name, current_pos, smooth_target))
                    break
        
        # Modification: Output processed joint count and details
        # print(f"Processed {len(processed_joints)} joints for grasping")
        # for joint_name, current_pos, target_pos in processed_joints[:5]:  # Only show first 5
        #     print(f"Joint '{joint_name}' current value: {current_pos:.2f}, target value: {target_pos:.2f}")
        
        # Check grasp status
        contact_points = self._p.getContactPoints(palm_id, self.target_object)
        contact_count = len(contact_points) if contact_points else 0
        
        # print(f"Contact point count: {contact_count}")
        
        # If there are enough contact points, enable gravity to test stability, but use very small gravity
        if contact_count >= 3 and not self.stable_grasp:
            print("\n*** Detected enough contact points, enabling slight gravity test ***")
            self.stable_grasp = True
            
            # Safer gravity test - don't use setGravity, but manually apply very small force
            self._p.setGravity(0, 0, 0)  # Turn off global gravity
            
            # Apply very small downward force to sphere
            obj_mass = 0.05  # Assume very small mass (50g)
            gravity_force = [0, 0, -1.0 * obj_mass]  # Use very small gravity acceleration
            
            self._p.applyExternalForce(
                self.target_object,
                -1,
                forceObj=gravity_force,
                posObj=obj_pos,
                flags=self._p.WORLD_FRAME
            )

    def _lift_phase(self, params=None):
 
        if not self.stable_grasp:
            return
            
        # Use default parameters
        if params is None:
            lift_force = 15.0
            force_scale = 1.0
            target_height = 0.35
        else:
            # Modification: Print and validate strategy parameters
            # print(f"Using strategy-provided lift parameters: {params}")
            lift_force = params.get('lift_force', 15.0)
            force_scale = params.get('force_scale', 1.0)
            target_height = params.get('target_height', 0.35)
            
        # Get palm and sphere positions
        palm_id = self.robot.robot_body.bodies[self.robot.robot_body.body_Index]
        palm_pos, _ = self._p.getBasePositionAndOrientation(palm_id)
        obj_pos, _ = self._p.getBasePositionAndOrientation(self.target_object)
        
        # Check contacts
        contact_points = self._p.getContactPoints(palm_id, self.target_object)
        contact_count = len(contact_points)
        
        if contact_count == 0:
            print("Warning: Lost contact during lift phase!")
            self.stable_grasp = False
            return
        
        # [Modification]: Fix finger contact situation analysis
        finger_contacts = {group: 0 for group in self.finger_groups}
        
        # Get all contact points
        for contact in contact_points:
            # Determine which link the contact point is on
            link_index = contact[3] if contact[1] == palm_id else contact[4]
            
            # Get link information
            try:
                link_info = self._p.getJointInfo(palm_id, link_index)
                link_name = link_info[12].decode('utf-8') if link_info[12] else "unknown"
                
                # Determine which finger it belongs to based on link name
                if 'th' in link_name.lower():
                    finger_contacts['thumb'] += 1
                elif 'rf' in link_name.lower():
                    finger_contacts['index'] += 1
                elif 'mf' in link_name.lower():
                    finger_contacts['middle'] += 1
                elif 'ff' in link_name.lower():
                    finger_contacts['ring'] += 1
            except Exception as e:
                print(f"Error analyzing link {link_index}: {e}")
        
        # Dynamically adjust grip force based on contact situation
        force_scale_adjusted = min(1.5, max(1.0, 3.0 / contact_count)) * force_scale
        
        # Modification: Record joint control information
        joint_controls = []
        
        # Apply grip force to all joints
        for joint in self.active_grasp_joints:
            # Get joint information
            joint_info = self._p.getJointInfo(palm_id, joint)
            joint_name = joint_info[1].decode('utf-8')
            
            # Get current position
            current_pos = self._p.getJointState(palm_id, joint)[0]
            
            # Maintain current position with greater force
            force = 70.0 * force_scale_adjusted
            
            # Apply position control
            self._p.setJointMotorControl2(
                palm_id,
                joint,
                controlMode=self._p.POSITION_CONTROL,
                targetPosition=current_pos,
                force=force,
                positionGain=0.8,
                velocityGain=0.1
            )
            
            # Record joint control information
            joint_controls.append((joint_name, current_pos, force))
        
        # Modification: Output joint control information
        # print(f"Lift phase controlled {len(joint_controls)} joints")
        # for joint_name, pos, force in joint_controls[:3]:  # Only show first 3
        #     print(f"Joint '{joint_name}': Position={pos:.3f}, Force={force:.1f}")
        
        # Determine required lift force
        current_height = obj_pos[2]
        target_height_adjusted = target_height
        
        # Adjust lift force based on current height
        lift_force_adjusted = lift_force
        if current_height < target_height_adjusted:
            lift_factor = max(0.5, min(1.5, (target_height_adjusted - current_height) * 10))
            lift_force_adjusted *= lift_factor
        
        # Apply lift force
        lift_force_vector = [0, 0, lift_force_adjusted]
        
        self._p.applyExternalForce(
            palm_id,
            -1,
            forceObj=lift_force_vector,
            posObj=palm_pos,
            flags=self._p.WORLD_FRAME
        )
        
        # Modification: Always output lift status
        # print(f"[Lift phase] Current height: {current_height:.3f}m, Target height: {target_height_adjusted:.3f}m")
        # print(f"[Lift phase] Lift force: {lift_force_adjusted:.1f}N, Force scale: {force_scale_adjusted:.2f}")
        # print(f"[Lift phase] Contact situation: {finger_contacts}, Total: {contact_count}")

    def _detect_contacts(self):
        """
        Enhanced contact point detection, check contact between each finger segment and sphere
        """
        palm_id = self.robot.robot_body.bodies[self.robot.robot_body.body_Index]
        contact_dict = {}
        total_contacts = 0
        
        # Check contact points for each finger segment
        for joint_idx in range(self._p.getNumJoints(palm_id)):
            joint_info = self._p.getJointInfo(palm_id, joint_idx)
            joint_name = joint_info[1].decode('utf-8')
            
            # Get contact points for current link
            contacts = self._p.getContactPoints(
                bodyA=palm_id,
                bodyB=self.target_object,
                linkIndexA=joint_idx
            )
            
            if contacts:
                contact_dict[joint_name] = len(contacts)
                total_contacts += len(contacts)
                
                # Output detailed contact point information
                for cp in contacts:
                    distance = cp[8]  # Contact distance
                    normal_force = cp[9]  # Normal force
                    lateral_friction1 = cp[10]  # Lateral friction force 1
                    lateral_friction2 = cp[11]  # Lateral friction force 2
                    
                    print(f"Contact point: Link '{joint_name}'")
                    print(f"  Distance: {distance:.4f}")
                    print(f"  Normal force: {normal_force:.2f}N")
                    print(f"  Friction force: {lateral_friction1:.2f}N, {lateral_friction2:.2f}N")
        
        if total_contacts > 0:
            print(f"\nTotal {total_contacts} contact points detected:")
            for joint, count in contact_dict.items():
                print(f"  {joint}: {count}")
        else:
            print("No contact points detected")
        
        return total_contacts, contact_dict
        
        return contact_exists
        
    def step(self, action):
        """Execute one step interaction, integrate state transition, reward calculation and termination check
        
        Modification: Ensure reward calculated by _compute_reward is correctly returned
        """
        try:
            # Safety check
            if not hasattr(self, 'target_object') or self.target_object is None:
                print("Error: target_object does not exist, attempting to reset environment")
                return np.zeros(self.observation_space.shape), 0.0, True, {'error': 'target_object missing'}
            
            # Check if object is still in simulation
            try:
                object_pos, object_orn = self._p.getBasePositionAndOrientation(self.target_object)
            except Exception as e:
                print(f"Failed to get target object state: {e}, attempting to reset environment")
                return np.zeros(self.observation_space.shape), 0.0, True, {'error': 'target_object state access failed'}
                
            if self.current_step % 100 == 0:
                print(f"\n====== Step {self.current_step} ======")
                print(f"Current phase: {'Stable grasp' if self.stable_grasp else ('Grasping' if self.grasp_started else 'Approaching')}")
            
            # [Modification]: Convert strategy action to grasp parameters
            grasp_params = self._convert_action_to_grasp_params(action)
            
            # Phase control logic
            if not self.grasp_started:
                # Approach phase - pass parameters
                if self._approach_phase(grasp_params):
                    self.grasp_started = True
                    print(f"*** Grasp phase triggered ***")
                    
            elif not self.stable_grasp:
                # Grasp phase - pass parameters
                self._grasp_phase(grasp_params)
                
            else:
                # Lift phase - pass parameters
                self._lift_phase(grasp_params)
                
            # Execute physics simulation
            for _ in range(5):  # Increase simulation steps for better stability
                self._p.stepSimulation()
                
            # Update object state
            try:
                new_obj_pos, new_obj_orn = self._p.getBasePositionAndOrientation(self.target_object)
                linear_vel, angular_vel = self._p.getBaseVelocity(self.target_object)
            except Exception as e:
                print(f"Failed to update object state: {e}")
                # Use safe values
                new_obj_pos = object_pos  # Use previously obtained position
                new_obj_orn = object_orn
                linear_vel = [0, 0, 0]
                angular_vel = [0, 0, 0]
            
            # Get contact information
            contact_points = self._p.getContactPoints(
                self.robot.robot_body.bodies[0],
                self.target_object
            )
            contact_count = len(contact_points) if contact_points else 0
            
            # Print contact information
            # print(f"Current contact point count: {contact_count}")
            # if contact_count > 0:
            #     print(f"Contact point details: {[f'Link:{cp[3]}' for cp in contact_points[:3]]}")
            
            # Modification: Explicitly call _compute_reward and use its return value
            reward = self._compute_reward(
                object_pos=new_obj_pos,
                linear_vel=linear_vel,
                contact_count=contact_count,
                contact_points=contact_points
            )
            
            # Debug information - helps track reward flow
            # print(f"[ENV DEBUG] Environment returned reward: {reward}")
            
            # Get new state
            try:
                state = self.robot.calc_state()
            except Exception as e:
                print(f"Failed to calculate state: {e}")
                state = np.zeros(self.observation_space.shape)
            
            # Check termination conditions
            done, done_reason = self._check_termination()
            if done:
                print(f"Episode ended: {done_reason}")
                
            # Build information dictionary
            info = {
                'done_reason': done_reason if done else None,
                'object_pos': new_obj_pos,
                'object_vel': linear_vel,
                'contact_count': contact_count,
                'phase': 'lift' if self.stable_grasp else ('grasp' if self.grasp_started else 'approach'),
                'grasp_params': grasp_params,
                # Add original reward for recording
                'orig_reward': reward
            }
            
            self.current_step += 1
            return state, reward, done, info
            
        except Exception as e:
            print(f"Step execution error: {str(e)}")
            traceback.print_exc()
            return np.zeros(self.observation_space.shape), 0.0, True, {'error': str(e)}
            
    def _convert_action_to_grasp_params(self, action):
        default_params = {
         
            'approach_speed': 0.25,        
            'approach_distance': 0.1,     
            'approach_force': 0.6,         
            'vertical_threshold': 0.13,              
         
            'finger_force': 30.0,         
            'position_gain': 0.3,          
            'velocity_gain': 0.1,         
            
            'lift_force': 15.0,            
        }
        
        if action is None or len(action) == 0:
            return default_params         
        action_dim = min(len(action), 8)
        grasp_params = default_params.copy()
        param_keys = list(default_params.keys())
        for i in range(action_dim):
            if i < len(param_keys):
                param_name = param_keys[i]
                default_val = default_params[param_name]
                grasp_params[param_name] = default_val * (1.0 + 0.3 * action[i])
        
        return grasp_params
    
    def reset_design(self, design):
        
        """Reset robot design parameters"""
        if self.robot is not None:
            self._p.resetSimulation()
            self.scene = None
            self.robot.reset_design(self._p, design)
            return self.reset()
        return None
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            self.camera_adjust()
            return None
            
        elif mode == 'rgb_array':
            return self.render_camera_image(
                resolution=(self._render_width, self._render_height)
            )
            
        else:
            raise ValueError(f"unsupported: {mode}")

    def _compute_reward(self, object_pos, linear_vel, contact_count, contact_points):
        rewards = {}
        palm_id = self.robot.robot_body.bodies[0]
        palm_pos, _ = self._p.getBasePositionAndOrientation(palm_id)
        
    
        rewards["Base reward"] = 0.1
        
        distance_to_ball = np.linalg.norm(np.array(palm_pos) - np.array(object_pos))
        ideal_distance = 0.05
        max_distance = 0.5  
        distance_reward = np.exp(-5.0 * max(0, distance_to_ball - ideal_distance) / (max_distance - ideal_distance))
        rewards["Approach reward"] = distance_reward
        
        # ===== 3. Touch reward =====
        # If there are contact points, give special touch reward
        touch_reward = 0.0
        if contact_count > 0:
            # Each contact point provides 0.2 reward, maximum 1.0
            touch_reward = min(1.0, contact_count * 0.2)
        rewards["Touch reward"] = touch_reward
        
        # ===== 4. Height reward =====
        base_height = 0.05  # Initial height
        target_height = 0.25  # Target height
        current_height = object_pos[2]
        
        # Normalize height to [0,1] interval
        height_norm = max(0.0, min(1.0, (current_height - base_height) / (target_height - base_height)))
        # Use linear function instead of Sigmoid to ensure obvious reward change
        rewards["Height reward"] = height_norm
        
        # ===== 5. Contact model reward =====
        # 5.1 Grasp quality evaluation
        grasp_quality = 0.0
        if hasattr(self.robot, 'contact_model') and contact_count > 0:
            contact_model_result = self.robot.contact_model.compute_contact_model(
                self._p, palm_id, self.target_object
            )
            if contact_model_result:
                grasp_quality = self.robot.contact_model.evaluate_grasp_quality(contact_model_result)
        rewards["Grasp quality"] = grasp_quality
        
        # 5.2 Contact force reward
        force_reward = 0.0
        if hasattr(self.robot, 'contact_model') and contact_count > 0:
            force_reward = self.robot.contact_model.compute_force_reward(
                self._p, palm_id, self.target_object
            )
        rewards["Contact force"] = force_reward
        
        # 5.3 Stability reward
        stability_reward = 0.0
        if hasattr(self.robot, 'contact_model') and contact_count > 0:
            # Collect joint information for stability evaluation
            joint_positions = []
            joint_velocities = []
            
            for joint_idx in self.active_grasp_joints:
                state = self._p.getJointState(palm_id, joint_idx)
                if state:
                    pos, vel = state[0], state[1]
                    joint_positions.append(pos)
                    joint_velocities.append(vel)
            
            stability_reward = self.robot.contact_model.compute_stability_reward(
                self._p, palm_id, self.target_object,
                joint_positions, joint_velocities, None,
                self.active_grasp_joints
            )
        rewards["Stability"] = stability_reward
        
        # ===== 6. Phase-based weight allocation =====
        # Determine current phase - approach, grasp or lift
        if contact_count == 0:  # Approach phase
            weights = {
                "Base reward": 0.05,
                "Approach reward": 0.60,  
                "Touch reward": 0.00,  
                "Height reward": 0.05,  
                "Grasp quality": 0.00,
                "Contact force": 0.00,
                "Stability": 0.00
            }
            current_phase = "Approach phase"
        elif current_height < 0.15:  # Grasp phase
            # Smooth transition - from initial grasp to prepare for lift
            transition = current_height / 0.15  # Transition value from 0 to 1
            
            # Initial grasp weights
            grasp_weights = {
                "Base reward": 0.05,
                "Approach reward": 0.20,  
                "Touch reward": 0.40,  
                "Height reward": 0.10,  
                "Grasp quality": 0.15,
                "Contact force": 0.10, 
                "Stability": 0.00
            }
            
            # Prepare to lift weights
            lift_prep_weights = {
                "Base reward": 0.05,
                "Approach reward": 0.10,
                "Touch reward": 0.20,
                "Height reward": 0.30,  
                "Grasp quality": 0.20,
                "Contact force": 0.10,
                "Stability": 0.05
            }
            
            # Smoothly blend weights
            weights = {}
            for key in grasp_weights:
                weights[key] = grasp_weights[key] * (1-transition) + lift_prep_weights[key] * transition
            
            current_phase = f"Grasp phase (transition: {transition:.2f})"
        else:  # Lift phase
            weights = {
                "Base reward": 0.05,
                "Approach reward": 0.05,
                "Touch reward": 0.10,
                "Height reward": 0.50,  # Height becomes main reward
                "Grasp quality": 0.15,
                "Contact force": 0.05,
                "Stability": 0.10
            }
            current_phase = "Lift phase"
        
        # ===== 7. Calculate weighted total reward =====
        total_reward = 0.0
        reward_details = []
        
        for key, value in rewards.items():
            weight = weights.get(key, 0.0)
            weighted_value = value * weight
            total_reward += weighted_value
            reward_details.append(f"{key}: {value:.3f} × {weight:.2f} = {weighted_value:.3f}")
        
        # Scale reward to make it more significant
        scaling_factor = 2.0
        final_reward = total_reward * scaling_factor
        
        # Print detailed reward information every 20 steps
        if self.current_step % 1000 == 0:
            print("\n===== Reward calculation details =====")
            print(f"Palm distance: {distance_to_ball:.3f}m, Object height: {current_height:.3f}m")
            print(f"Contact point count: {contact_count}, Current phase: {current_phase}")
            print("\n".join(reward_details))
            print(f"Total reward (unscaled): {total_reward:.3f} × {scaling_factor:.1f} = {final_reward:.1f}")
            print("=====================\n")
        
        return final_reward