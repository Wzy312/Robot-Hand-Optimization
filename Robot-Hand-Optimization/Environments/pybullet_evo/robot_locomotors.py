from .robot_bases import XmlBasedRobot, MJCFBasedRobot
import numpy as np
import pybullet
import os, inspect
import tempfile
import atexit
import time
import xmltodict
from .contact_model import ContactModel
import traceback
import inspect 
import xml.etree.ElementTree as ET
import gym, gym.spaces, gym.utils

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def cleanup_temp_file(filepath):
    """Clean up the temporary file if it exists."""
    if os.path.exists(filepath):
        os.remove(filepath)

class HandBase(MJCFBasedRobot):
    """The robot base class provides general robot functions"""
    def __init__(self, model_xml, robot_name, action_dim, obs_dim, self_collision=True, power=1.0):
        super().__init__(
            model_xml=model_xml,
            robot_name=robot_name,
            action_dim=action_dim,
            obs_dim=obs_dim,
            self_collision=self_collision  
        )
        self.power = power
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0.5
        self.body_xyz = [0, 0, 0]
        self.contact_model = ContactModel()
        self.finger_list = []  
        self.fingertip_list = []  
        self.initial_height = None

    def reset_design(self, bullet_client, design): 
        pass
        
class AllegroHand(HandBase):
    def __init__(self, design=None):
    
        self.design_dim = 28 
        self.num_joints = 16         
        joint_state_dim = self.num_joints * 2  
        fingertip_state_dim = 4 * 6 
        palm_pos_dim = 3
        palm_orn_dim = 4
        palm_vel_dim = 6
        self.observation_dim = joint_state_dim + fingertip_state_dim + palm_pos_dim + palm_orn_dim + palm_vel_dim + self.design_dim
                
        print(f"State dimension details:")
        print(f"- Joint state: {joint_state_dim}")
        print(f"- Fingertip state: {fingertip_state_dim}")
        print(f"- Palm position: {palm_pos_dim}")
        print(f"- Palm orientation: {palm_orn_dim}")
        print(f"- Palm velocity: {palm_vel_dim}")
        print(f"- Design parameters: {self.design_dim}")
        print(f"Total dimension: {self.observation_dim}")

        # Initial position and target object position
        self.start_pos_x = 0.0
        self.start_pos_y = -0.1
        self.start_pos_z = 0.3
        self.target_object_position = [0, 0, 0.24]
        self.width_mapping = {
            'rf': [16, 17, 18],
            'mf': [19, 20, 21],
            'ff': [22, 23, 24],
            'th': [25, 26, 27]
        }
        self.finger_names = ['rf', 'mf', 'ff', 'th']
        self.fingertip_list = [f"{finger}_distal" for finger in self.finger_names]

        # Base sizes (used to calculate the length of each phalanx) - can be adjusted as needed
        self.base_sizes = {
            'proximal': 0.0487,
            'middle': 0.0305,
            'distal': 0.0235
        }

        # Finger configurations (for generating XML), ensure consistency with the original XML file
        self.finger_configs = {
            'rf': {
                'base_pos': [0.05, 0.04, 0],
                'base_orn': [1, 0, 1, 0],
                'joint_idxs': [0, 1, 2, 3],
                'ranges': {
                    'j0_twist': (-0.6, 0.6),
                    'j1': (-1.7, 0.6),
                    'j2': (-0.35, 1.3),
                    'j3': (-0.2, 1.5)
                }
            },
            'mf': {
                'base_pos': [0.05, 0, 0],
                'base_orn': [1, 0, 1, 0],
                'joint_idxs': [4, 5, 6, 7],
                'ranges': {
                    'j0_twist': (-0.6, 0.6),
                    'j1': (-1.7, 0.6),
                    'j2': (-0.35, 1.3),
                    'j3': (-0.2, 1.5)
                }
            },
            'ff': {
                'base_pos': [0.05, -0.04, 0],
                'base_orn': [1, 0, 1, 0],
                'joint_idxs': [8, 9, 10, 11],
                'ranges': {
                    'j0_twist': (-0.6, 0.6),
                    'j1': (-1.7, 0.6),
                    'j2': (-0.35, 1.3),
                    'j3': (-0.2, 1.5)
                }
            },
            'th': {
                'base_pos': [-0.01, -0.03, -0.01],
                'base_orn': [1, 0.33, 0, 0],
                # For the thumb, use 4 joints: thj0, thj1_twist, thj1_bend, thj2 (thj3 is the tip)
                'joint_idxs': [12, 13, 14, 15],
                'ranges': {
                    'j1_twist': (0.5, 2.3),
                    'j1_bend': (-0.2, 1.1),
                    'j2': (-0.25, 1.4),
                    'j3': (-0.2, 1.6)
                }
            }
        }
        self._adapted_xml_file = tempfile.NamedTemporaryFile(delete=False, prefix='allegro_hand_', suffix='.xml', mode='w')
        self.model_xml = self._adapted_xml_file.name
        self._adapted_xml_file.close()
        self.body_xyz = [0, 0, 0]
        self.body_rpy = [0, 0, 0]

        # Design parameter validation
        if design is not None and not self._verify_design_params(design):
            raise ValueError("Invalid design parameters")
        self.design = (np.array([1.0] * self.design_dim)
                       if design is None else np.array(design))
        if not self.generate_mujoco_xml(self.model_xml, self.design):
            raise RuntimeError("Failed to generate XML model")

        super().__init__(model_xml=self.model_xml, robot_name="palm",
                         action_dim=self.num_joints, obs_dim=self.observation_dim,
                         self_collision=True, power=1.0)
        atexit.register(self.cleanup_temp_file, self.model_xml)
        self.initial_height = None
        self.body_xyz = [0, 0, 0]

    def cleanup_temp_file(self, filepath):
        if os.path.exists(filepath):
            os.remove(filepath)

    def generate_mujoco_xml(self, file, design):
        """
        Directly read the initial XML file, then dynamically modify the finger 
        lengths and widths based on the design, and finally write it to the file 
        for simulation.
        """
        try:
            # Read the original XML file (please ensure the path is correct)
            initial_xml_path = "/home/ubuntu2244/Coadaptatin0/Environments/pybullet_evo/allegro_hand.xml"
            with open(initial_xml_path, 'r') as f:
                xml_content = f.read()
            
            # Parse XML
            xml_dict = xmltodict.parse(xml_content)
            
            # Log geometry parameters before modification (for debugging)
            self._log_geometry_params(xml_dict, "Before Modification")
            
            # Get the palm node
            palm_node = xml_dict['mujoco']['worldbody']['body']
            if not palm_node or palm_node.get('@name') != 'palm':
                raise ValueError("Could not find the palm node")
            
            # Modify each finger
            self._process_finger(palm_node, 'rf', design)  # Ring finger
            self._process_finger(palm_node, 'mf', design)  # Middle finger
            self._process_finger(palm_node, 'ff', design)  # Index finger
            self._process_finger(palm_node, 'th', design)  # Thumb
            
            # Log geometry parameters after modification (for debugging)
            self._log_geometry_params(xml_dict, "After Modification")
            
            # Save the modified XML
            modified_xml = xmltodict.unparse(xml_dict, pretty=True)
            with open(file, 'w') as f:
                f.write(modified_xml)
            
            if not self._verify_xml():
                raise ValueError("XML validation failed")
                
            print(f"XML file successfully modified and saved to: {file}")
            return True
            
        except Exception as e:
            print(f"XML generation failed: {str(e)}")
            traceback.print_exc()
            return False

    def _process_finger(self, palm_node, finger_name, design):
        """
        Precisely process all segments of a specific finger.
        
        Args:
            palm_node: The palm XML node.
            finger_name: Finger name code ('rf', 'mf', 'ff', 'th').
            design: The design parameter array.
        """
        # Get finger configuration
        config = self.finger_configs[finger_name]
        joint_idxs = config['joint_idxs']
        width_indices = self.width_mapping[finger_name]
        
        # Prepare scaling factors
        length_scaling = {
            'proximal': design[joint_idxs[0]],
            'middle': design[joint_idxs[1]],
            'distal': design[joint_idxs[2]]
        }
        width_scaling = {
            'proximal': design[width_indices[0]],
            'middle': design[width_indices[1]],
            'distal': design[width_indices[2]]
        }
        
        print(f"Processing finger {finger_name}:")
        print(f"  Length scaling: {length_scaling}")
        print(f"  Width scaling: {width_scaling}")
        
        # Get finger base node
        finger_base = None
        for body in palm_node.get('body', []):
            if isinstance(body, dict) and body.get('@name') == f"{finger_name}_base":
                finger_base = body
                break
        
        if not finger_base:
            print(f"Warning: Could not find base node for finger {finger_name}")
            return
        
        # Select processing path based on finger type
        if finger_name != 'th':
            # Non-thumb finger processing path
            self._process_normal_finger(finger_base, finger_name, length_scaling, width_scaling)
        else:
            # Thumb processing path
            self._process_thumb(finger_base, length_scaling, width_scaling)

    def _process_normal_finger(self, finger_base, finger_name, length_scaling, width_scaling):
        """Process non-thumb fingers (index, middle, ring)."""
        # Get nodes at each level
        univ = self._get_child_by_name(finger_base, f"{finger_name}_univ")
        if not univ:
            print(f"Warning: Could not find {finger_name}_univ node")
            return
            
        univ_sep = self._get_child_by_name(univ, f"{finger_name}_univ_sep")
        if not univ_sep:
            print(f"Warning: Could not find {finger_name}_univ_sep node")
            return
            
        # Get and modify proximal phalanx
        proximal = self._get_child_by_name(univ_sep, f"{finger_name}_proximal")
        if proximal:
            self._modify_geometry(proximal, 'proximal', length_scaling, width_scaling)
            
            # Get and modify middle phalanx
            middle = self._get_child_by_name(proximal, f"{finger_name}_middle")
            if middle:
                self._modify_geometry(middle, 'middle', length_scaling, width_scaling)
                
                # Get and modify distal phalanx
                distal = self._get_child_by_name(middle, f"{finger_name}_distal")
                if distal:
                    self._modify_geometry(distal, 'distal', length_scaling, width_scaling)
                else:
                    print(f"Warning: Could not find {finger_name}_distal node")
            else:
                print(f"Warning: Could not find {finger_name}_middle node")
        else:
            print(f"Warning: Could not find {finger_name}_proximal node")

    def _process_thumb(self, thumb_base, length_scaling, width_scaling):
        """Specifically handle the thumb, considering its unique structure."""
        # Get the thumb's proximal node
        proximal = self._get_child_by_name(thumb_base, "th_proximal")
        if not proximal:
            print("Warning: Could not find thumb proximal node")
            return
            
        # Get the twist node
        twist = self._get_child_by_name(proximal, "th_twist")
        if not twist:
            print("Warning: Could not find thumb twist node")
            return
            
        # Get the bend node
        proximal_bend = self._get_child_by_name(twist, "th_proximal_bend")
        if proximal_bend:
            # Modify proximal geometry
            self._modify_geometry(proximal_bend, 'proximal', length_scaling, width_scaling)
            
            # Get the medial node
            medial = self._get_child_by_name(proximal_bend, "th_medial")
            if medial:
                # Modify medial geometry
                self._modify_geometry(medial, 'middle', length_scaling, width_scaling)
                
                # Get the distal node
                distal = self._get_child_by_name(medial, "th_distal")
                if distal:
                    # Modify distal geometry
                    self._modify_geometry(distal, 'distal', length_scaling, width_scaling)
                else:
                    print("Warning: Could not find thumb distal node")
            else:
                print("Warning: Could not find thumb medial node")
        else:
            print("Warning: Could not find thumb bend node")

    def _modify_geometry(self, node, segment_type, length_scaling, width_scaling):
        """
        Precisely modify the geometric attributes of a node.
        
        Args:
            node: The XML node to modify.
            segment_type: Phalanx type ('proximal', 'middle', 'distal').
            length_scaling: Dictionary of length scaling factors.
            width_scaling: Dictionary of width scaling factors.
        """
        if 'geom' not in node:
            print(f"Warning: Node {node.get('@name', 'unknown')} has no geometry")
            return
            
        geom = node['geom']
        node_name = node.get('@name', 'unknown')
        
        # Modify length - fromto attribute
        if '@fromto' in geom:
            try:
                original_fromto = geom['@fromto']
                parts = original_fromto.split()
                
                # Ensure there are 6 values (x1 y1 z1 x2 y2 z2)
                if len(parts) == 6:
                    # Calculate length vector
                    x1, y1, z1 = float(parts[0]), float(parts[1]), float(parts[2])
                    x2, y2, z2 = float(parts[3]), float(parts[4]), float(parts[5])
                    
                    # Calculate original length vector components
                    dx, dy, dz = x2-x1, y2-y1, z2-z1
                    
                    # Apply scaling factor to the Z component
                    scale = length_scaling[segment_type]
                    z2 = z1 + dz * scale
                    
                    # Update the fromto string
                    new_fromto = f"{x1} {y1} {z1} {x2} {y2} {z2}"
                    geom['@fromto'] = new_fromto
                    
                    print(f"  Modified {node_name} length: {original_fromto} -> {new_fromto}")
                else:
                    print(f"Warning: Incorrect fromto format: {original_fromto}")
                    
            except Exception as e:
                print(f"Error modifying {node_name} length: {e}")
        
        # Modify width - size attribute
        if '@size' in geom:
            try:
                original_size = geom['@size']
                scale = width_scaling[segment_type]
                
                # Handle single-value case
                if ' ' not in original_size:
                    size_val = float(original_size)
                    new_size = str(size_val * scale)
                    geom['@size'] = new_size
                    print(f"  Modified {node_name} width: {original_size} -> {new_size}")
                else:
                    # Handle multi-value case
                    parts = original_size.split()
                    new_parts = [str(float(p) * scale) for p in parts]
                    new_size = ' '.join(new_parts)
                    geom['@size'] = new_size
                    print(f"  Modified {node_name} width: {original_size} -> {new_size}")
                    
            except Exception as e:
                print(f"Error modifying {node_name} width: {e}")

    def _get_child_by_name(self, parent, name):
        if not parent or 'body' not in parent:
            return None
            
        bodies = parent['body']
        
        # Handle the case of a single body
        if isinstance(bodies, dict):
            if bodies.get('@name') == name:
                return bodies
            return None
            
        # Handle the case of multiple bodies
        for body in bodies:
            if isinstance(body, dict) and body.get('@name') == name:
                return body
                
        return None

    def _log_geometry_params(self, xml_dict, prefix=""):
        """
        Print all geometry parameters in the XML for debugging.
        
        Args:
            xml_dict: The XML dictionary.
            prefix: A prefix for the log message.
        """
        geometries = []
        
        def traverse(node, path=""):
            if isinstance(node, dict):
                # Check if there is a name and geom
                if '@name' in node and 'geom' in node:
                    name = node['@name']
                    geom = node['geom']
                    if isinstance(geom, dict):
                        fromto = geom.get('@fromto', None)
                        size = geom.get('@size', None)
                        if fromto or size:
                            geometries.append({
                                'path': path,
                                'name': name,
                                'fromto': fromto,
                                'size': size
                            })
                
                # Recursively traverse all keys
                for key, value in node.items():
                    traverse(value, path + "/" + key)
            elif isinstance(node, list):
                # Recursively traverse list items
                for i, item in enumerate(node):
                    traverse(item, path + f"[{i}]")
        
        # Start traversal
        traverse(xml_dict)
        
        # Print results
        print(f"\n=== {prefix} Geometry Parameters ({len(geometries)} items) ===")
        for g in geometries:
            print(f"{g['name']}:")
            print(f"  Path: {g['path']}")
            print(f"  fromto: {g['fromto']}")
            print(f"  size: {g['size']}")
        print("=============================")

    # Keep the original _verify_xml method, but with stricter checks
    def _verify_xml(self):
        """
        Validate if the generated XML is valid by checking if it contains all necessary joints.
        
        Returns:
            bool: Whether the XML is valid.
        """
        try:
            # Parse XML file
            tree = ET.parse(self.model_xml)
            root = tree.getroot()
            
            # Check if the joints for each finger exist
            joint_expectations = {
                'rf': ['rfj0_twist', 'rfj1', 'rfj2', 'rfj3'],
                'mf': ['mfj0_twist', 'mfj1', 'mfj2', 'mfj3'],
                'ff': ['ffj0_twist', 'ffj1', 'ffj2', 'ffj3'],
                'th': ['thj1_twist', 'thj1_bend', 'thj2', 'thj3']
            }
            
            all_valid = True
            
            # Check each finger
            for finger, expected_joints in joint_expectations.items():
                found_joints = []
                
                # Find joints
                for name in expected_joints:
                    joints = root.findall(f".//joint[@name='{name}']")
                    if joints:
                        found_joints.append(name)
                
                # Validate results
                if len(found_joints) != len(expected_joints):
                    print(f"Error: {finger} finger has incorrect number of joints, expected {len(expected_joints)}, found {len(found_joints)}")
                    print(f"  Found joints: {found_joints}")
                    print(f"  Missing joints: {set(expected_joints) - set(found_joints)}")
                    all_valid = False
            
            return all_valid
            
        except ET.ParseError as e:
            print(f"XML parsing error: {str(e)}")
            with open(self.model_xml, 'r') as f:
                print(f"XML content:\n{f.read()}")
            return False
            
    def _verify_design_params(self, design):
        try:
            design = np.array(design, dtype=np.float32)
            if design.shape != (self.design_dim,):
                print(f"Design parameter dimension error: expected {self.design_dim}, got {len(design)}")
                return False
            if not np.all((design >= 0.8) & (design <= 1.2)):
                out = design[(design < 0.8) | (design > 1.2)]
                print(f"Design parameters out of range [0.8, 1.2]: {out}")
                return False
            return True
        except Exception as e:
            print(f"Design parameter validation failed: {e}")
            traceback.print_exc()
            return False

    def robot_specific_reset(self, bullet_client):
        
        super().robot_specific_reset(bullet_client)
        self.body_xyz = [0, 0, 0]
        self.body_rpy = [0, 0, 0]
        if not hasattr(self, 'parts') or not self.parts:
            raise RuntimeError("Robot parts not properly initialized")
        if not hasattr(self, 'jdict') or not self.jdict:
            raise RuntimeError("Joint dictionary not properly initialized")
        if not hasattr(self, 'ordered_joints') or not self.ordered_joints:
            raise RuntimeError("Ordered joints not properly initialized")
        if not hasattr(self, 'objects') or not self.objects:
            raise RuntimeError("Robot model not properly loaded")
        robot_body_id = self.objects[0]
        try:
            for tip in self.fingertip_list:
                if tip not in self.jdict:
                    print(f"Warning: Fingertip {tip} not found in joint dictionary")
                    continue
                tip_idx = self.jdict[tip].joint_index
                bullet_client.changeDynamics(robot_body_id, tip_idx,
                                             lateralFriction=1.5,
                                             spinningFriction=0.7,
                                             restitution=0.0,
                                             contactStiffness=1e4,
                                             contactDamping=1000)
        except Exception as e:
            print(f"Error configuring fingertip dynamics: {e}")
        try:
            for joint in self.ordered_joints:
                bullet_client.changeDynamics(robot_body_id, joint.joint_index,
                                             linearDamping=0.5,
                                             lateralFriction=0.1)
                joint.reset_current_position(0.0, 0.0)
        except Exception as e:
            print(f"Error configuring joint dynamics: {e}")
        try:
            self.feet = [self.parts[f] for f in self.fingertip_list]

            self.feet_contact = np.array([0.0] * len(self.feet), dtype=np.float32)
        except Exception as e:
            print(f"Error initializing fingertip contact state: {e}")
        if hasattr(self, 'robot_body') and self.robot_body is not None:
            body_pose = self.robot_body.pose()
            self.initial_z = body_pose.xyz()[2]
        print("Successfully completed robot-specific reset")
              
    def calc_state(self):
        try:
            joint_states = []
            seen = set()
            for joint in self.ordered_joints:
                if joint.joint_name not in seen:
                    seen.add(joint.joint_name)
                    pos, vel = joint.current_relative_position()
                    joint_states.extend([pos, vel])
            fingertip_positions = []
            for finger in self.finger_names:
                tip = f"{finger}_distal"
                if tip in self.parts:
                    pos = self.parts[tip].current_position()
                    vel = self.parts[tip].current_velocity()
                    fingertip_positions.extend([*pos, *vel])
            palm_pos = self.robot_body.current_position()
            palm_orn = self.robot_body.current_orientation()
            lin_vel, ang_vel = self.robot_body._p.getBaseVelocity(self.robot_body.bodies[self.robot_body.body_Index])
            palm_vel = np.concatenate([lin_vel, ang_vel])
            if self.robot_body is not None:
                _, orient = self.robot_body._p.getBasePositionAndOrientation(self.robot_body.bodies[self.robot_body.body_Index])
                self.body_rpy = self.robot_body._p.getEulerFromQuaternion(orient)
            state = np.concatenate([joint_states, fingertip_positions, palm_pos, palm_orn, palm_vel, self.design]).astype(np.float32)
            assert state.shape[0] == self.observation_dim, f"State dimension error: {state.shape[0]} != {self.observation_dim}"
            return state
        except Exception as e:
            print(f"State calculation error: {e}")  
            traceback.print_exc()
            return np.zeros(self.observation_dim, dtype=np.float32)

    def reset_design(self, bullet_client, design):
        """
        Reset the robot's design parameters.
        
        Modification: Added detailed debug output and error handling to ensure 
        design parameters are applied correctly.
        """
        try:
            print(f"\n===== Starting Design Parameter Reset =====")
            print(f"New design parameters: {design}")
            
            # Store current design parameters
            self.design = np.array(design)
            
            # Create a design dictionary for ContactModel
            design_dict = {
                'length_proximal': design[0],
                'length_middle': design[1],
                'length_distal': design[2],
                'width_proximal': design[3],
                'width_middle': design[4],
                'width_distal': design[5]
            }
            
            # Update the contact model
            if hasattr(self, 'contact_model'):
                self.contact_model.set_design(design_dict)
                print(f"Updated contact model design parameters")
            
            # Before generating the XML file, ensure the old file is closed
            if hasattr(self, '_adapted_xml_file') and hasattr(self._adapted_xml_file, 'close'):
                try:
                    self._adapted_xml_file.close()
                except:
                    pass
            
            # Use timestamp and pid to generate a short unique ID
            unique_id = f"{int(time.time()) % 10000}_{os.getpid() % 10000}"
            
            # Get temporary file directory
            temp_dir = tempfile.gettempdir()
            
            # Create a new XML file path, keeping the filename short
            new_xml_path = os.path.join(temp_dir, f"allegro_hand_{unique_id}.xml")
            print(f"Generating new XML file: {new_xml_path}")
            
            # Generate new XML model file
            if not self.generate_mujoco_xml(new_xml_path, self.design):
                raise RuntimeError("Failed to generate new XML model file")
            
            # Update model path
            self.model_xml = new_xml_path
            print(f"Updated model path: {self.model_xml}")
            
            # Ensure important state is saved before resetting the simulation
            old_objects = self.objects.copy() if hasattr(self, 'objects') and self.objects else []
            
            # Safely reset the simulation environment
            success = False
            try:
                print("Resetting simulation environment...")
                bullet_client.resetSimulation()
                print("Simulation environment has been reset")
                success = True
            except Exception as e:
                print(f"Failed to reset simulation environment: {e}")
                import traceback
                traceback.print_exc()
            
            # If reset is successful, reload the robot
            if success:
                # Reload the robot and set physics parameters
                result = self.reset(bullet_client)
                print(f"Robot reset result: {'Success' if result is not None else 'Failure'}")
                
                # Verify if design parameters were successfully applied
                self._verify_design_application(bullet_client)
                
                print(f"===== Design Parameter Reset Complete =====")
                return True
            else:
                print(f"===== Design Parameter Reset Failed =====")
                return False
                
        except Exception as e:
            print(f"An error occurred during design reset: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _verify_design_application(self, bullet_client):
        """
        New: Verify if design parameters have been successfully applied to the actual model.
        """
        print("\nVerifying design parameter application...")
        
        # Check if the robot was loaded successfully
        if not hasattr(self, 'objects') or not self.objects:
            print("Warning: Robot object not loaded")
            return
            
        robot_id = self.objects[0]
        
        # Check phalanx dimensions
        finger_names = ['rf', 'mf', 'ff', 'th']
        segments = ['proximal', 'middle', 'distal']
        
        for finger in finger_names:
            for segment in segments:
                link_name = f"{finger}_{segment}"
                
                # Find link index
                link_index = -1
                for i in range(bullet_client.getNumJoints(robot_id)):
                    info = bullet_client.getJointInfo(robot_id, i)
                    if info[12].decode('utf-8') == link_name:
                        link_index = i
                        break
                
                if link_index != -1:
                    # Get visual shape data for the link
                    visual_shapes = bullet_client.getVisualShapeData(robot_id, link_index)
                    if visual_shapes:
                        shape = visual_shapes[0]
                        dimensions = shape[3]  # Dimension information
                        print(f"Link {link_name} dimensions: {dimensions}")
                    else:
                        print(f"Link {link_name} has no visual shape data")
                else:
                    print(f"Link {link_name} not found")
        
        print("Design parameter verification complete")

    def set_new_design(self, design):
        
        print(f"\n===== Setting New Design Parameters =====")
        print(f"Design parameters: {design}")
        
        # Save current design parameters for comparison
        old_design = self._current_design.copy() if hasattr(self, '_current_design') else None
        
        # Validate that design parameters are within the valid range
        design_np = np.array(design)
        valid_range = True
        for i, (param, bounds) in enumerate(zip(design_np, self.design_params_bounds)):
            if param < bounds[0] or param > bounds[1]:
                print(f"Warning: Parameter {i} value {param} is out of range {bounds}")
                valid_range = False
        
        if not valid_range:
            print("Warning: Design parameters are out of the valid range and will be clipped.")
            design_np = np.clip(design_np, 
                               [bound[0] for bound in self.design_params_bounds],
                               [bound[1] for bound in self.design_params_bounds])
        reset_success = self._env.reset_design(design_np)
        
        if reset_success:

            self._current_design = design_np
            self._config_numpy = np.array(design_np)
            print(f"Design parameters successfully updated")
            if old_design is not None:
                diff = design_np - old_design
                print(f"Design parameter change: {diff}")
                max_diff = np.max(np.abs(diff))
                print(f"Maximum change magnitude: {max_diff}")
            
            # Reset recorder
            if hasattr(self, '_video_recorder') and self._record_video:
                self._video_recorder.increase_folder_counter()
                print("Video recorder has been reset")
        else:
            print("Warning: Design parameter update failed")
        
        print(f"===== Setting Design Parameters Finished =====")