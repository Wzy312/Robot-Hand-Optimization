import os
import sys
import time
import hashlib
import coadapt
import experiment_configs as cfg
import json
import numpy as np
import traceback

def setup_xml_path(config):
   
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, 'Environments', 'pybullet_evo', 'allegro_hand.xml')
    
 
    if not os.path.exists(xml_path):
        print(f"Warning: XML file not found at: {xml_path}")
  
        os.makedirs(os.path.dirname(xml_path), exist_ok=True)
   
        with open(xml_path, 'w') as f:
            f.write('''<?xml version="1.0" ?>
<mujoco model="allegro_hand">
    <!-- XML content here -->
</mujoco>''')
        print(f"Created placeholder XML file at: {xml_path}")
    
    config['env']['xml_path'] = xml_path
    return config

def validate_config(config):
    """验证配置"""
    required_keys = ['data_folder', 'env']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    if 'env_name' not in config['env']:
        raise ValueError("Missing env_name in config['env']")

def ensure_directories(config):
  
 
    data_folder = os.path.join(os.path.dirname(__file__), config['data_folder'])
    os.makedirs(data_folder, exist_ok=True)
 
    env_folder = os.path.join(os.path.dirname(__file__), 'Environments')
    os.makedirs(env_folder, exist_ok=True)
    
    pybullet_folder = os.path.join(env_folder, 'pybullet_evo')
    os.makedirs(pybullet_folder, exist_ok=True)
    
    return config

def main(config):

    try:
 
        validate_config(config)       
        config = ensure_directories(config)         
        if config['env']['env_name'].lower() == 'allegrohand':
            config = setup_xml_path(config)
        folder = config['data_folder']
        rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]
        file_str = f"./{folder}/{time.ctime().replace(' ', '_')}__{rand_id}"
        config['data_folder_experiment'] = file_str
        
        if not os.path.exists(file_str):
            os.makedirs(file_str)
        with open(os.path.join(file_str, 'config.json'), 'w') as fd:
            fd.write(json.dumps(config, indent=2))
            
        print(f"Starting experiment with config: {config['name']}")
        print(f"Data will be saved to: {file_str}")
        co = coadapt.Coadaptation(config)
        co.run()
        
    except Exception as e:
        print(f"Error during experiment execution: {str(e)}")
        traceback.print_exc()
        if "bodies list is empty" in str(e):
            if 'xml_path' in config['env']:
                xml_path = config['env']['xml_path']
                print(f"\nXML file details:")
                if os.path.exists(xml_path):
                    print(f"XML file exists at: {xml_path}")
                    with open(xml_path, 'r') as f:
                        print("\nXML content:")
                        print(f.read())
                else:
                    print(f"XML file not found at: {xml_path}")
        raise

if __name__ == "__main__":
    try:

        if len(sys.argv) > 1:
            config_name = sys.argv[1]
        else:
            config_name = 'allegrohand'
        
        if config_name not in cfg.config_dict:
            valid_keys = ', '.join(cfg.config_dict.keys())
            raise ValueError(f"Invalid experiment name: '{config_name}'. Valid options are: {valid_keys}")
        
        print(f"Loading config: {config_name}")
        config = cfg.config_dict[config_name]
        main(config)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
