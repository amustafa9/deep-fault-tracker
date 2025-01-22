import yaml
from src.trainers import FaultEstimator3D

def load_yaml_file(file_path):
    """
    Loads a YAML file and returns its contents as a Python dictionary.

    Args:
        file_path: Path to the YAML file.

    Returns:
        A Python dictionary containing the data loaded from the YAML file.
        Returns None if the file cannot be loaded.
    """
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            return data
    except FileNotFoundError:
        print(f"Error: YAML file '{file_path}' not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error: Unable to parse YAML file: {exc}")
        return None

yaml_file = "/home/ubuntu/projects/faultseg/configs/FaultEstimator3D.yaml"
args = load_yaml_file(yaml_file)
fault_trainer = FaultEstimator3D(args)

if args['training']:
    fault_trainer.train()
else:
    pass

if args['inference']:
    fault_trainer.predict()
else:
    pass