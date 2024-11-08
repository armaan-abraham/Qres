from pathlib import Path
import torch
from qres.agent import Agent

def load_models(directory_path):
    # Convert string path to Path object if necessary
    directory = Path(directory_path)
    
    # Find all .pt files recursively
    model_files = list(directory.rglob("*.pt"))
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # List to store policy nets
    policy_nets = []

    agent_base = Agent(device=device)
    
    # Load each model
    for model_path in model_files:
        try:
            # Initialize agent
            agent = agent_base.clone()
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Load policy net state dict
            agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            
            # Add to list
            policy_nets.append(agent.policy_net)
            
            print(f"Successfully loaded model: {model_path} {agent.get_n_params()}")
            
        except Exception as e:
            print(f"Failed to load {model_path}: {str(e)}")
    
    print(f"\nTotal models loaded: {len(policy_nets)}")
    return policy_nets

def compare_parameters(policy_nets):
    if len(policy_nets) < 2:
        print("Need at least 2 models to compare")
        return
        
    reference_model = policy_nets[0]
    
    # Get named parameters from reference model
    ref_params = dict(reference_model.named_parameters())
    
    # Compare each model to the reference
    for i, model in enumerate(policy_nets[1:], 1):
        print(f"\nComparing Model {i} to Reference Model:")
        
        # Get parameters for current model
        curr_params = dict(model.named_parameters())
        
        # Compare each parameter
        for name in ref_params.keys():
            ref_tensor = ref_params[name]
            curr_tensor = curr_params[name]
            
            # Check if tensors are exactly equal
            is_equal = torch.allclose(ref_tensor, curr_tensor)
            if is_equal:
                print("equal!", name)

if __name__ == "__main__":
    # Specify the directory containing model files
    models_dir = Path(__file__).parent / "data"
    
    # Load all models
    policy_nets = load_models(models_dir)
    
    # Compare parameters
    compare_parameters(policy_nets)
