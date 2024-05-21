import torch
from rl_games.torch_runner import Runner
import os
import yaml
import torch
import gym
import numpy as np
import onnx
import onnxruntime as ort

from rl_games.algos_torch.network_builder import A2CBuilder
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from colorama import Fore, Back, Style
from emoji import emojize as em


def build_rlg_model(weights_path, params):

    weights = torch.load(weights_path, map_location=torch.device('cpu'))


    net_params = params['train']['params']['network']

    network = A2CBuilder()
    network.load(net_params)

    model_a2c = ModelA2CContinuousLogStd(network)

    build_config = {
            'actions_num' : params['task']['env']['numActions'],
            'input_shape' : (params['task']['env']['numObservations'],),
            'num_seqs' : 1,
            'value_size': 1,
            'normalize_value' : params['train']['params']['config']['normalize_value'],
            'normalize_input': params['train']['params']['config']['normalize_input']
        }
    model = model_a2c.build(build_config)
    model.to('cuda:0')

    model.load_state_dict(weights['model'])

    model.eval()

    return model

class ModelWrapper(torch.nn.Module):
    '''
    Main idea is to ignore outputs which we don't need from model
    '''
    def __init__(self, model):
        torch.nn.Module.__init__(self)
        self._model = model
        
        
    def forward(self,input_dict):
        input_dict['obs'] = self._model.norm_obs(input_dict['obs'])
        '''
        just model export doesn't work. Looks like onnx issue with torch distributions
        thats why we are exporting only neural network
        '''
        #print(input_dict)
        #output_dict = self._model.a2c_network(input_dict)
        #input_dict['is_train'] = False
        #return output_dict['logits'], output_dict['values']
        return self._model.a2c_network(input_dict)
    

def export(model, nobs, nact, name):
    import rl_games.algos_torch.flatten as flatten
    inputs = {      'is_train': False,
                    'prev_actions': None,
                    'obs': torch.zeros((1,nobs)).to('cuda:0').type(torch.float32),
                    'rnn_states': None}

    with torch.no_grad():
        adapter = flatten.TracingAdapter(ModelWrapper(model), inputs, allow_non_tensor=True)
        traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)
        flattened_outputs = traced(*adapter.flattened_inputs)
        print(flattened_outputs)
    
    print(Fore.YELLOW + em("Model traced. Exporting... :package:") + Style.RESET_ALL)
    torch.onnx.export(traced, *adapter.flattened_inputs, f"{name}.onnx", verbose=True, input_names=['obs'], output_names=['mu','log_std', 'value'])
    print(Fore.GREEN + em(":check_mark_button: Model exported") + Style.RESET_ALL)

    print(Fore.YELLOW + em(":magnifying_glass_tilted_right: Integrity check...") + Style.RESET_ALL)
    onnx_model = onnx.load(f"{name}.onnx")

    # Check that the model is well formed
    onnx.checker.check_model(onnx_model)
    print(Fore.GREEN + em(":check_mark_button: Model Integrity Ok.") + Style.RESET_ALL)


def run_inference(model, observation, det=True):
    """
    Runs inference on a model given an observation. Uses RLgames model.

    Args:
        model: A PyTorch model.
        observation: A numpy array containing the observation.

    Returns:
        A numpy array containing the action.
    """
    
    with torch.no_grad():
        obs_tensor = torch.from_numpy(observation).to('cuda:0').type(torch.float32)
        obs_dict = {'is_train': False,
                    'prev_actions': None,
                    'obs': obs_tensor,
                    'rnn_states': None}
        action_dict = model(obs_dict)
        actions = action_dict['mus'] if det else action_dict['actions']
        actions = actions.cpu().numpy()

    return actions

def check_model(rlgmodel, nobs, name):
    print(Fore.YELLOW + em(':magnifying_glass_tilted_right: Checking model inference consistency...') + Style.RESET_ALL)
    # Inference with ONNX Runtime
    ort_model = ort.InferenceSession(f"{name}.onnx")

    outputs = ort_model.run(
        None,
        {"obs": np.zeros((1, nobs)).astype(np.float32)},
    )
    print("ONNX Runtime output:  ", outputs[0][0].tolist())   
    # inference with rlgames
    rlg_outputs = run_inference(rlgmodel, np.zeros((1, nobs)))
    print("RLGames output:       ", rlg_outputs[0].tolist())
    #Check that the outputs are the same
    assert np.allclose(outputs[0], rlg_outputs, rtol=1e-03, atol=1e-05)
    print(Fore.GREEN + em(':check_mark_button: Model inference consistency is ok! :partying_face: Enjoy!') + Style.RESET_ALL)



import argparse
import yaml
import glob

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Export and check the model")

    # Add the arguments
    parser.add_argument('experiment_name', type=str, help='The name of the experiment')
    parser.add_argument('checkpoint', type=str, help='The checkpoint to use ("best" or episode number)')

    # Parse the arguments
    args = parser.parse_args()

    config_yaml = f"runs/{args.experiment_name}/config.yaml"
    if args.checkpoint == 'best':
        pth_file = f"runs/{args.experiment_name}/nn/{args.experiment_name}.pth"
    else:
        pth_files = glob.glob(f"runs/{args.experiment_name}/nn/last_ep_{args.checkpoint}_*.pth")
        print(pth_files)
        if not pth_files:
            print(f"No .pth file found for checkpoint {args.checkpoint}")
            return
        pth_file = pth_files[0]

    # Load the configuration from the yaml file
    with open(config_yaml, 'r') as f:
        config = yaml.safe_load(f)

    # Build the model
    model = build_rlg_model(pth_file, config)
    print(Fore.GREEN + 'Model loaded' + Style.RESET_ALL)


    # Get the number of observations and actions from the configuration
    nobs = config['task']['env']['numObservations']
    nact = config['task']['env']['numActions']

    # Export the model
    name = f"runs/{args.experiment_name}/nn/{args.experiment_name}export"
    export(model, nobs, nact, name)

    # Check the model
    check_model(model, nobs, name)

if __name__ == "__main__":
    main()