import argparse
import setGPU
import nanogcg
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import re
import pickle

def parse_arguments():
    parser = argparse.ArgumentParser(description="Web Shopping Agent with Nanogcg and Vicuna Model")

    # Model and Tokenizer
    parser.add_argument('--model_id', type=str, default="mistralai/Ministral-8B-Instruct-2410",
                        help='Model identifier from Hugging Face Hub')
    
    parser.add_argument('--mode', type=str, default='train', help='mode')
    
    # Task Configuration
    parser.add_argument('--task', type=int, default=3,
                        help='Task index (integer)')
    
    # Configuration Parameters for GCGConfig
    parser.add_argument('--num_steps', type=int, default=300, help='Number of steps')
    parser.add_argument('--search_width', type=int, default=128, help='Search width')
    parser.add_argument('--optim_str_init', type=str, default=''' x x x x x x x x x x x x x x x x x x x x x x x x x''',
                        help='Initial optimization string')
    parser.add_argument('--topk', type=int, default=32, help='Top-K parameter')
    parser.add_argument('--n_replace', type=int, default=1, help='Number of replacements')
    parser.add_argument('--buffer_size', type=int, default=0, help='Buffer size')
    parser.add_argument('--filter_ids', type=bool, default=True, help='Filter IDs flag')
    parser.add_argument('--early_stop', action='store_true', help='Filter IDs flag')
    parser.add_argument('--use_mellowmax', type=bool, default=False, help='Use Mellowmax flag')
    parser.add_argument('--verbosity', type=str, default='CRITICAL', help='Verbosity level')

    # Data and Output
    parser.add_argument('--data_path', type=str, default="data/webshop/train_ministral.csv", help='Path to the CSV data file')
    
    # Optional: Output directory (defaults to current directory)
    parser.add_argument('--output_dir', type=str, default="results/webshop/",
                        help='Directory to save the results. The filename will be generated automatically based on model_id and task.')
    parser.add_argument('--suffix', type=str, default="normal")
    return parser.parse_args()

def sanitize_filename(name):
    """
    Sanitize the model_id to create a filesystem-friendly filename.
    Replaces non-alphanumeric characters with underscores.
    """
    return re.sub(r'[^A-Za-z0-9]+', '_', name)

def generate_output_path(model_id, task, output_dir, args):
    """
    Generates an output file path based on model_id and task.
    """
    sanitized_model_id = sanitize_filename(model_id)
    filename = f"{sanitized_model_id}_task{task}_o{args.n_replace}_b{args.buffer_size}_gcg.pkl"
        
    path = os.path.join(output_dir, args.mode)
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.join(path, filename)

def load_model_and_tokenizer(model_id):
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer

def prepare_data(data_path, task, mode):
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    if mode == 'train':
        goals = data[f'adv_search_history_task_{task}'].tolist()
        targets = data[f'adv_target_task_{task}'].tolist()
    else:
        goals = [data[f'adv_search_history_task_{task}'].tolist()[0]]
        targets = [data[f'adv_target_task_{task}'].tolist()[0]]
    return goals, targets

def insert_optim_str(observation: str, asin: str) -> str:
    # Split on "[SEP]"
    parts = observation.split("[SEP]")
    
    # Clean up each part (remove extra whitespace)
    parts = [p.strip() for p in parts]
    
    # Find the ASIN
    for i, part in enumerate(parts):
        if part == asin:
            # The next part should be the product description where we insert optim_str
            if i + 1 < len(parts):
                parts[i + 1] = parts[i + 1] + " {optim_str}"
            break  # Stop after the first match

    # Reconstruct the modified observation
    modified_observation = " [SEP] ".join(parts)
    return modified_observation

def run_agent(model, tokenizer, goals, targets, config, output_path, args):
    results = []
    asr = 0

    for i in tqdm(range(len(goals)), desc="Processing Goals"):
        target = f'{targets[i]}'
        
        current_message = eval(goals[i])
        current_message[-1]["content"] = insert_optim_str(current_message[-1]["content"], target)
        result = nanogcg.vanilla_run(model, tokenizer, current_message, 'click[' + target + ']', config)
        
        results.append(result)
        
        # Save the results to a JSON file
        with open(output_path, 'wb') as f:  # Note the 'wb' mode for writing in binary
            pickle.dump(results, f)
    print(f"Results saved to {output_path}")

def main():
    args = parse_arguments()
    args.early_stop = True
    
    # Generate output_path based on model_id and task
    output_path = generate_output_path(args.model_id, args.task, args.output_dir, args)
    print(f"Output will be saved to: {output_path}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id)

    # Prepare data
    goals, targets = prepare_data(args.data_path, args.task, args.mode)

    # Set up GCGConfig
    config = nanogcg.VanillaGCGConfig(
        num_steps=args.num_steps,
        search_width=args.search_width,
        optim_str_init=args.optim_str_init,
        topk=args.topk,
        early_stop=args.early_stop,
        n_replace=args.n_replace,
        buffer_size=args.buffer_size,
        filter_ids=args.filter_ids,
        use_mellowmax=args.use_mellowmax,
        verbosity=args.verbosity,
        # Add other configuration parameters if needed
    )
    # Run the agent
    run_agent(model, tokenizer, goals, targets, config, output_path, args)

if __name__ == "__main__":
    main()
