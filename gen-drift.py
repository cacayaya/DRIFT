import os
import json
import random
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator


def batchify_data(data, batch_size):
    """
    Split the full list of data entries into chunks of size batch_size.
    Yields lists of entries (batches).
    """
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def collate_fn(batch, tokenizer, device):
    """
    Given a list of `batch` entries and a tokenizer, concatenate each entry's
    conversation into one prompt string and tokenize the entire batch together.
    
    Returns:
      - input_ids: tensor of token IDs, shape (batch_size, seq_len)
      - attention_mask: tensor of attention masks, same shape
      - batch: original list of entries (so we can map outputs back)
    """
    formatted_prompts = []
    for entry in batch:
        prompt_messages = entry["prompt"]
        rejected = entry["rejected"]
        user_pref = entry["user_preference"]

        # Build the prompt string exactly as in the original code
        formatted = ""
        # 1. Add the first "system" message (if any)
        for msg in prompt_messages:
            if msg["role"] == "system":
                formatted += msg["content"] + "\n\n"
                break
        # 2. Add conversation history (user/assistant turns)
        for msg in prompt_messages:
            if msg["role"] == "system":
                continue
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"

        # 3. Append the single rejected response
        rej_txt = rejected[0]["content"]
        formatted += f"Assistant: {rej_txt}\n"
        # 4. Append the user's preference feedback
        pref_txt = user_pref["content"]
        formatted += f"User: {pref_txt}\n"
        # 5. Add our instruction to produce an improved response
        formatted += (
            "System: Based on the user's latest feedback, please provide "
            "an improved response that addresses their concerns.\n"
        )
        formatted += "Assistant: "

        formatted_prompts.append(formatted)

    # Tokenize the entire batch at once. Pad sequences to the longest in this batch.
    encodings = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048  # adjust if prompts can exceed this
    )

    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    return input_ids, attention_mask, batch


def generate_batch(model, tokenizer, input_ids, attention_mask, max_new_tokens=512):
    """
    Run model.generate on the provided batched inputs.
    Returns a list of generated text strings (one per batch entry).
    """
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
        do_sample=True,
        return_dict_in_generate=True
    )
    sequences = outputs.sequences  # shape: (batch_size, seq_len + new_tokens)

    # Decode each generated sequence to a string
    decoded_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)

    chosen_list = []
    marker = "Assistant: "
    for text in decoded_texts:
        # Find the last occurrence of "Assistant: " and take everything after it
        idx = text.rfind(marker)
        if idx != -1:
            chosen_txt = text[idx + len(marker) :].strip()
        else:
            # Fallback if marker is not found
            chosen_txt = text.strip()
        chosen_list.append(chosen_txt)

    return chosen_list


def main():
    # ------------ Argument Parsing ------------
    parser = argparse.ArgumentParser(description="Generate DRIFT preference pairs")
    parser.add_argument("--test_size", type=int, default=100, 
                        help="Number of pairs for test set (default: 100)")
    parser.add_argument("--train_size", type=int, default=None,
                        help="Number of pairs for train set (default: all remaining after test)")
    parser.add_argument("--model_name", type=str, 
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name to use for generation")
    parser.add_argument("--input_file", type=str,
                        default="./data/dsat_data.jsonl",
                        help="Input JSONL file path")
    parser.add_argument("--train_output_file", type=str,
                        default="./data/DRIFT/qwen2.5-7b/iter1/train.jsonl",
                        help="Output file for training data")
    parser.add_argument("--test_output_file", type=str,
                        default="./data/DRIFT/qwen2.5-7b/iter1/test.jsonl",
                        help="Output file for test data")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for generation (default: 16)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens to generate (default: 512)")
    
    args = parser.parse_args()
    
    # ------------ Configuration ------------
    model_name = args.model_name
    input_file = args.input_file
    train_output_file = args.train_output_file
    test_output_file = args.test_output_file
    test_size = args.test_size
    train_size = args.train_size
    
    os.makedirs(os.path.dirname(train_output_file), exist_ok=True)

    # Load the entire dataset into memory
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} entries from {input_file}")
    
    # Determine actual train/test sizes based on available data
    total_data = len(data)
    actual_test_size = min(test_size, total_data)
    
    if train_size is None:
        actual_train_size = total_data - actual_test_size
    else:
        actual_train_size = min(train_size, total_data - actual_test_size)
    
    print(f"Will generate {actual_test_size} test pairs and {actual_train_size} train pairs")

    # Initialize Accelerator for multi‐GPU inference
    accelerator = Accelerator(mixed_precision="bf16", cpu=False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left"   # For causal models, pad on the left
    )
    # Ensure a pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model sharded across all GPUs, using BF16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    tokenizer = accelerator.prepare(tokenizer)

    batch_size = args.batch_size   # Increase until you hit memory limit
    max_new_tokens = args.max_new_tokens

    f_test = open(test_output_file,  "w", encoding="utf-8")
    f_train = open(train_output_file, "w", encoding="utf-8")

    # Counter of how many preference pairs have been written so far
    written_count = 0
    max_pairs_to_generate = actual_test_size + actual_train_size

    for batch_entries in tqdm(batchify_data(data, batch_size),
                              total=(len(data) + batch_size - 1) // batch_size):
        
        # Stop if we've generated enough pairs
        if written_count >= max_pairs_to_generate:
            break
            
        input_ids, attention_mask, this_batch = collate_fn(
            batch_entries, tokenizer, accelerator.device
        )

        with torch.no_grad():
            chosen_texts = generate_batch(
                model, tokenizer, input_ids, attention_mask, max_new_tokens=max_new_tokens
            )

        # For each generated response in this batch:
        for entry, chosen_txt in zip(this_batch, chosen_texts):
            # Stop if we've generated enough pairs
            if written_count >= max_pairs_to_generate:
                break
                
            chosen = [{"role": "assistant", "content": chosen_txt}]
            preference_pair = {
                "prompt": entry["prompt"],
                "chosen": chosen,
                "rejected": entry["rejected"]
            }

            # If we have written fewer than actual_test_size so far, send to test.jsonl
            if written_count < actual_test_size:
                f_test.write(json.dumps(preference_pair, ensure_ascii=False) + "\n")
                f_test.flush()
            else:
                # After test_size, send everything to train.jsonl
                f_train.write(json.dumps(preference_pair, ensure_ascii=False) + "\n")
                f_train.flush()

            written_count += 1

    # Close file handles
    f_test.close()
    f_train.close()

    test_pairs = min(actual_test_size, written_count)
    train_pairs = max(0, written_count - actual_test_size)
    
    print(f"Finished writing {written_count} pairs: "
          f"{test_pairs} → {test_output_file}, "
          f"{train_pairs} → {train_output_file}")

if __name__ == "__main__":
    main()
