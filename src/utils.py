from pathlib import Path
import yaml

def load_port_config(file_path: str = "configs/general.yml") -> dict:
    config_path = Path(file_path).resolve()
    
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config