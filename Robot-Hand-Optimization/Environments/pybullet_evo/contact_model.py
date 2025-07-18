import numpy as np
import pybullet
import traceback

class ContactModel:
    def __init__(self, design=None):
        """
        Initialize the contact model, set the finger geometry and physical parameters

        Args:
        design: Optional design parameter dictionary
        """

        self.base_lengths = {
            'proximal': 0.0487,  
            'middle': 0.0305,    
            'distal': 0.0235    
        }
        

        self.base_widths = {
            'proximal': 0.0115,  
            'middle': 0.0117,    
            'distal': 0.0101     
        }
        
   
        self.finger_params = {
            'lengths': self.base_lengths.copy(),
            'widths': self.base_widths.copy()
        }
        
        # Set contact parameters
        self.contact_params = {
            'friction_coef': 0.8,      
            'min_force': 0.1,         
            'max_force': 10.0,         
            'normal_stiffness': 1000,  
            'tangent_stiffness': 800    
        }
        

        self.reward_params = {
            'lift_threshold': 0.1,         
            'energy_threshold': 100,       
            'time_threshold': 1000,        
            'position_tolerance': 0.05,   
            'force_balance_weight': 0.5,   
            'force_magnitude_weight': 0.5
         
        }
        
        # If design parameters are provided, update the geometry parameters
        if design is not None:
            self.set_design(design)
            
    def set_design(self, design):
        """
        Update finger geometry and physical parameters based on design parameters

        Args:
        design: Design parameter dictionary, including length and width scaling factors
        """
        try:
            print(f"Update contact model design parameters: {design}")
                      
            self.finger_params['lengths']['proximal'] = self.base_lengths['proximal'] * design.get('length_proximal', 1.0)
            self.finger_params['lengths']['middle']   = self.base_lengths['middle']   * design.get('length_middle', 1.0)
            self.finger_params['lengths']['distal']   = self.base_lengths['distal']   * design.get('length_distal', 1.0)
            self.finger_params['widths']['proximal']  = self.base_widths['proximal']  * design.get('width_proximal', 1.0)
            self.finger_params['widths']['middle']    = self.base_widths['middle']    * design.get('width_middle', 1.0)
            self.finger_params['widths']['distal']    = self.base_widths['distal']    * design.get('width_distal', 1.0)            
            length_sum = (self.finger_params['lengths']['proximal'] + 
                         self.finger_params['lengths']['middle'] + 
                         self.finger_params['lengths']['distal'])
                         
            width_sum = (self.finger_params['widths']['proximal'] + 
                        self.finger_params['widths']['middle'] + 
                        self.finger_params['widths']['distal'])
                        
            # Update physics parameters - adjust stiffness and friction based on geometry changes
            # Longer fingers require more stiffness to maintain the same damping ratio
            length_factor = length_sum / sum(self.base_lengths.values())
            self.contact_params['normal_stiffness'] = 1000 * (length_factor ** 0.5)
            self.contact_params['tangent_stiffness'] = 800 * (length_factor ** 0.5)
            
            # Width affects friction coefficient
            width_factor = width_sum / sum(self.base_widths.values())
            self.contact_params['friction_coef'] = 0.8 * (width_factor ** 0.3)          
            self.design_ratios = {
                'proximal_middle_ratio': self.finger_params['lengths']['proximal'] / self.finger_params['lengths']['middle'],
                'middle_distal_ratio': self.finger_params['lengths']['middle'] / self.finger_params['lengths']['distal'],
                'width_length_ratio': width_sum / length_sum
            }
            
            print(f"Updated finger parameters: {self.finger_params}")
            print(f"Updated contact parameters: {self.contact_params}")
            print(f"design ratios: {self.design_ratios}")
            
        except Exception as e:
            print(f"Design parameter update failed: {e}")
            traceback.print_exc()

    def compute_contact_model(self, bullet_client, robot_id, object_id, joint_angles=None, joint_torques=None):
        """
        Convert raw touchpoint data into a structured format
        """
        try:
            contact_points = bullet_client.getContactPoints(robot_id, object_id)
            if not contact_points:
                return None
                
            contact_forces = []
            for point in contact_points:

                if point[1] == robot_id:
                    link_idx = point[3]
                else:
                    link_idx = point[4]
                    
                link_name = "unknown"
                try:
                    link_info = bullet_client.getJointInfo(robot_id, link_idx)
                    if link_info and link_info[12]:
                        link_name = link_info[12].decode('utf-8')
                except:
                    pass

                force = {
                    'position': point[5],           
                    'normal': point[7],             
                    'normal_force': point[9],       
                    'lateral_friction1': point[10], 
                    'lateral_friction2': point[11], 
                    'link_idx': link_idx,          
                    'link_name': link_name         
                }
                contact_forces.append(force)
                
            return {
                'forces': contact_forces, 
                'num_contacts': len(contact_forces),
                'total_force': sum(f['normal_force'] for f in contact_forces),
                'timestamp': bullet_client.getPhysicsEngineParameters().get('numSolverIterations', 0)
            }
            
        except Exception as e:
            print(f"Contact model calculation error: {e}")
            traceback.print_exc()
            return None

    def compute_stability_reward(self, bullet_client, robot_id, object_id, joint_positions=None, joint_velocities=None, joint_torques=None, active_joint_indices=None):
  
        contact_points = bullet_client.getContactPoints(robot_id, object_id)
        if not contact_points:
            return 0.0
        if active_joint_indices is None:
            num_joints = bullet_client.getNumJoints(robot_id)
            active_joint_indices = list(range(num_joints))
        
        num_dof = len(active_joint_indices)
        if num_dof <= 0:
            return 0.0      
        if joint_positions is None or joint_velocities is None:
            joint_positions = []
            joint_velocities = []
            for j in active_joint_indices:
                pos, vel, _, _ = bullet_client.getJointState(robot_id, j)
                joint_positions.append(pos)
                joint_velocities.append(vel)
        
        # Zero acceleration assumption
        joint_accelerations = [0.0] * num_dof
        

        contact_normals = []
        contact_positions = []
        contact_forces = []
        valid_contacts = []
        valid_jacobians = []  
        
        # # First pass: collect all valid contact points and calculate Jacobian matrix
        for point in contact_points:
            contact_link = point[3] if point[1] == robot_id else point[4]
            
            if contact_link < 0 or contact_link >= bullet_client.getNumJoints(robot_id):
                continue
                
            # Get the contact point position and normal vector
            contact_pos_world = point[6] 
            contact_normal = point[7]   
            normal_force = point[9]                 
            if normal_force < 0.01:
                continue
                
            # Calculate the Jacobian matrix of the contact point
            try:
                link_state = bullet_client.getLinkState(robot_id, contact_link)
                link_pos_world = link_state[0]
                link_orient_world = link_state[1]                
                inv_pos, inv_orn = bullet_client.invertTransform(link_pos_world, link_orient_world)
                local_pos, _ = bullet_client.multiplyTransforms(
                    inv_pos, inv_orn,
                    contact_pos_world, [0, 0, 0, 1]
                )
                
 
                jac_t, jac_r = bullet_client.calculateJacobian(
                    bodyUniqueId=robot_id,
                    linkIndex=contact_link,
                    localPosition=local_pos,
                    objPositions=joint_positions,
                    objVelocities=joint_velocities,
                    objAccelerations=joint_accelerations
                )
                jac_t = np.array(jac_t)
                if jac_t.shape[1] > num_dof:
                    jac_t = jac_t[:, :num_dof]
                if jac_t.shape == (3, num_dof):
                    valid_jacobians.append(jac_t)
                    contact_normals.append(contact_normal)
                    contact_positions.append(contact_pos_world)
                    contact_forces.append(normal_force)
                    valid_contacts.append((contact_link, contact_pos_world))
                else:
                    print(f"Warning: Skipping Jacobian matrix of incorrect dimensions, shape {jac_t.shape}, expecting (3, {num_dof})")
            except Exception as e:
                print(f"Error computing Jacobian matrix: {e}")
                continue
        
        # Ensure there are enough effective contact points
        if len(valid_contacts) < 2:
            return 0.0
        obj_pos, obj_orn = bullet_client.getBasePositionAndOrientation(object_id)
        num_contacts = len(valid_contacts)
        
        # Construct the grasping matrix G and the hand Jacobian matrix H, and ensure that the dimensions are compatible
        G = np.zeros((6, num_contacts * 3))
        H = np.zeros((num_contacts * 3, num_dof))
        print(f"Construct matrix - G: {G.shape}, H: {H.shape}, valid contact points: {num_contacts}")
        for i in range(num_contacts):
            pos = contact_positions[i]
            r = np.array(pos) - np.array(obj_pos)
            
            # skew-symmetric matrix
            r_skew = np.array([
                [0, -r[2], r[1]],
                [r[2], 0, -r[0]],
                [-r[1], r[0], 0]
            ])

            G_i = np.zeros((6, 3))
            G_i[0:3, 0:3] = np.eye(3)  
            G_i[3:6, 0:3] = r_skew                 
            G[:, i*3:(i+1)*3] = G_i
            H[i*3:(i+1)*3, :] = valid_jacobians[i]

        if G.shape[1] != H.shape[0]:
            print(f"Error: incompatible final matrix dimensions, G: {G.shape}, H: {H.shape}")
 
            return 0.0
        
        # Calculate stability index
        try:
            # Use SVD to calculate the pseudo-inverse of G
            U, S, Vh = np.linalg.svd(G, full_matrices=False)          
            eps = 1e-6
            S_inv = np.array([1/s if s > eps else 0 for s in S])
            G_pinv = Vh.T @ np.diag(S_inv) @ U.T
            
            # Calculate the mapping matrix from contact force to joint torque
            GH = G @ H
            
            # Use SVD to analyze stability quality
            U2, S2, Vh2 = np.linalg.svd(GH, full_matrices=False)
            cond = float('inf')
            
            # Calculate the condition number. The smaller the condition number, the more stable it is.
            if len(S2) > 1 and S2[-1] > eps:
                cond = S2[0] / S2[-1]

                stability = 1.0 / (1.0 + np.log(1.0 + cond))
            else:
                stability = 0.0
 
            # 1. Contact point distribution
            contact_factor = min(1.0, num_contacts / 6.0)  # Consider the number of contact points
            
            # 2. Uniformity of contact force distribution
            mean_force = np.mean(contact_forces)
            std_force = np.std(contact_forces)
            force_balance = 1.0 / (1.0 + std_force/mean_force) if mean_force > 0 else 0.5
            
            # 3. Weighted combination of various stability indicators
            weighted_stability = (
                0.5 * stability +     
                0.3 * force_balance +  
                0.2 * contact_factor   
            )
            
            print(f"Stability calculation successful: matrix condition number = {cond:.2f}, stability score = {weighted_stability:.3f}")
            
            return weighted_stability
            
        except Exception as e:
            print(f"Error in calculating stability SVD: {e}")
            
            return 0.0
                   
    def compute_force_reward(self, bullet_client, robot_id, object_id):
 

        contact_points = bullet_client.getContactPoints(robot_id, object_id)
        if not contact_points:
            return 0.0
        contact_forces = []
        contact_positions = []
        
        for point in contact_points:
            force = point[9]  
            if force < 0.01: 
                continue
            contact_forces.append(force)
            contact_positions.append(point[6]) 
        
        if not contact_forces:
            return 0.0
            
        # Calculate mean force and standard deviation
        mean_force = np.mean(contact_forces)
        std_force = np.std(contact_forces)
        
        # Force Reward - Rewards for being close to the ideal force
        length_factor = sum(self.finger_params['lengths'].values()) / sum(self.base_lengths.values())
        ideal_force = 1.0 * length_factor  # Longer fingers require more force
        
        # Use a Gaussian function to reward situations close to the ideal force
        force_diff = abs(mean_force - ideal_force)
        magnitude_reward = np.exp(-0.5 * (force_diff / ideal_force)**2)
        
        # Force balance bonus - lower standard deviation means more even force distribution
        balance_factor = 1.0 / (1.0 + std_force/mean_force) if mean_force > 0 else 0
        
        # Geometric distribution of contact points
        geometry_factor = 0.0
        if len(contact_positions) >= 3:
            # Calculate the spatial distribution of contact points
            positions = np.array(contact_positions)
            centroid = np.mean(positions, axis=0)
            
            # Calculate the average distance to the centroid
            dists = np.linalg.norm(positions - centroid, axis=1)
            avg_dist = np.mean(dists)
            
            # Normalize to the range [0,1]
            geometry_factor = min(1.0, avg_dist / 0.1)  
        elif len(contact_positions) == 2:
            # Two-point case - calculating distance
            dist = np.linalg.norm(np.array(contact_positions[0]) - np.array(contact_positions[1]))
            geometry_factor = min(1.0, dist / 0.1)
        else:
            geometry_factor = 0.0
        
        force_reward = (
            0.4 * magnitude_reward + 
            0.4 * balance_factor + 
            0.2 * geometry_factor
        )
        
        return force_reward
            

    def evaluate_grasp_quality(self, contact_forces):
  
        if not contact_forces or 'forces' not in contact_forces or len(contact_forces['forces']) == 0:
            return 0.0

        # 1. Number of touchpoints score
        num_contacts = len(contact_forces['forces'])
        if num_contacts < 2:  # At least 2 contact points are required to form a stable grasp
            return 0.0
        forces = np.array([f['normal_force'] for f in contact_forces['forces']])
        positions = np.array([f['position'] for f in contact_forces['forces']])
        
        if np.mean(forces) < 0.05:  
            return 0.0

        # 2. Force balance score - considers the uniformity of force distribution
        force_mean = np.mean(forces)
        force_std = np.std(forces)
        force_balance = 1.0 - min(1.0, force_std / (force_mean + 1e-6))
        
       # 3. Spatial distribution score - considers the spatial distribution of contact points
        spatial_score = 0.0
        if num_contacts >= 3 and positions.shape[1] >= 3:  

            centroid = np.mean(positions, axis=0)
            dists = np.linalg.norm(positions - centroid, axis=1)
            avg_dist = np.mean(dists)
            spatial_score = min(1.0, avg_dist / 0.1) 
        elif num_contacts >= 2:

            dists = []
            for i in range(num_contacts):
                for j in range(i+1, num_contacts):
                    dists.append(np.linalg.norm(positions[i] - positions[j]))
            avg_dist = np.mean(dists) if dists else 0.0
            spatial_score = min(1.0, avg_dist / 0.1)
            
        # 4. Rewards based on the number of contact points
        contact_score = min(1.0, num_contacts / 4.0) 
            
       # Weighted combination of indicators
        quality = (
            0.4 * force_balance + 
            0.3 * spatial_score + 
            0.3 * contact_score
        )
        
        return quality

    def check_termination(self, bullet_client, object_id, current_step):
     
        object_pos, _ = bullet_client.getBasePositionAndOrientation(object_id)
        if object_pos[2] > self.reward_params['lift_threshold']:
            return True, "Object lifted successfully"
        if current_step > self.reward_params['time_threshold']:
            return True, "Time limit exceeded"
        return False, ""
