import os
import datasets
import pandas as pd
import argparse

from src.prompts import INVESTIGATOR_PROMPT_TEMPLATE

def make_map_fn(split):
    """
    Constructs the mapping function for the dataset.
    verl expects exactly these fields: data_source, prompt, ability, reward_model, extra_info.
    """
    def process_fn(example, idx):
        behavior = example['behavior']
        
        # 1. Format the prompt as a standard HF chat list
        prompt_content = INVESTIGATOR_PROMPT_TEMPLATE.format(behavior=behavior)        

        prompt = [{
            "role": "user",
            "content": prompt_content
        }]
        
        # 2. Pack the metadata inside the "reward_model" ground_truth
        # verl will pass this exact dictionary to your compute_score wrapper!
        ground_truth_dict = {
            "behavior": behavior,
            "behavior_id": example['behavior_id'],
            "optimizer_target": example['optimizer_target']
        }
        
        # 3. Return the exact schema verl expects
        data = {
            "data_source": "transluce_jailbreak",
            "prompt": prompt,
            "ability": "red_teaming",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth_dict
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "category": example.get("category", "unknown")
            }
        }
        return data
    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data')
    parser.add_argument('--input_file', default='raw_behaviors.json')
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    # 1. Load your raw JSON (Hugging Face datasets handles JSONL automatically)
    dataset = datasets.load_dataset('json', data_files=args.input_file, split='train')

    train_dataset = dataset
    test_dataset = dataset

    # 3. Apply the mapping function
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # 4. Save to Parquet
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))

    print(f"✅ Saved {len(train_dataset)} train and {len(test_dataset)} test examples to {args.local_dir}")