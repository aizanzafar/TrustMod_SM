import os
import argparse
import json
import logging
import asyncio
import sys
from ollama import AsyncClient
import ollama
from tqdm import tqdm
from config_loader import config
import pprint

# Silence HTTP Noise
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def checkLogprobs(client: AsyncClient, model: str) -> bool:
    """Checks if the model/server version supports logprobs."""
    try:
        response = await client.chat(
            model=model,
            messages=[{'role': 'user', 'content': 'Hi'}],
            options={'logprobs': True, 'num_logprobs': 1}
        )
        # Check if the key exists and isn't None
        return response.get('logprobs') is not None
    except Exception:
        return False

async def evaluate_post(client: AsyncClient, model: str, sample: dict, use_logprobs: bool) -> dict:
    prompt = f"Evaluate this post:\n\n{sample.get('obfuscated_text', '')}"
    system_prompt = config.moderation.get("system_prompt", "You are a moderator.")
    
    # Request logprobs in options only if the model supports it
    chat_options = {'temperature': config.moderation.get("temperature", 0.0)}
    if use_logprobs:
        chat_options['logprobs'] = True
        chat_options['num_logprobs'] = 5

    try:
        response = await client.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            format='json',
            options=chat_options
        )
        
        result_content = json.loads(response['message']['content'])
        sample['model_verdict'] = result_content.get('verdict')
        sample['model_rationale'] = result_content.get('rationale')
        
        # --- Logic Hierarchy ---
        
        # 1. Try Logprobs first
        sample['model_confidence'] = None
        if use_logprobs and 'logprobs' in response and response['logprobs']:
            verdict_str = str(sample['model_verdict']).lower().strip()
            # Find the token in the logprobs list that matches the verdict
            for entry in response['logprobs']:
                token_text = entry.get('token', '').lower().strip().replace('"', '')
                if token_text == verdict_str:
                    sample['verdict_logprob'] = entry.get('logprob')
                    break
        
        # 2. Fallback to JSON confidence if logprobs failed/unavailable
        if (sample['model_confidence'] == None):
            sample['model_confidence'] = result_content.get('confidence')
        
        # 3. If neither exists, they remain None (already handled by .get())

    except Exception as e:
        logging.error(f"Error evaluating post: {e}")
        return None
        
    return sample

async def run_benchmark(model: str, input_file: str, output_file: str):
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}", file=sys.stderr)
        return

    logging.info(f"Evaluating {model}...")
    client = AsyncClient()
    
    # Check capability once before starting the loop
    has_logprobs = await checkLogprobs(client, model)
    logging.info(f"Model {model} supports logprobs: {has_logprobs}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f]

        batch_size = config.moderation.get("batch_size", 10)
        results = []

        # Setup output file
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        with tqdm(total=len(samples), file=sys.stdout, desc=f"Benchmark {model}") as pbar:
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                # Pass the 'has_logprobs' boolean to the function
                batch_tasks = [evaluate_post(client, model, sample, has_logprobs) for sample in batch]
                batch_results = await asyncio.gather(*batch_tasks)
                
                # Filter out None results from crashes
                valid_results = [r for r in batch_results if r is not None]
                results.extend(valid_results)
                
                # Append to file incrementally
                with open(output_file, 'a', encoding='utf-8') as f:
                    for res in valid_results:
                        f.write(json.dumps(res) + "\n")
                
                pbar.update(len(batch))

        logging.info(f"{len(results)} have been processed!")
                
    except Exception as e:
        logging.error(f"FATAL BENCHMARK ERROR for model {model}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrustMod-SM Benchmark Runner")
    parser.add_argument("--model", type=str, required=True, help="Ollama model name")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save results")
    parser.add_argument("--input_file", type=str, default=config.paths.get("input_file", "data/sbic_obfuscated.jsonl"))
    parser.add_argument("--test", action="store_true") # Changed to action="store_true" for easier CLI use
    
    args = parser.parse_args()
    
    print("Pulling model")
    ollama.pull(args.model)
    print("Finished pulling model")
    
    if args.test:
        asyncio.run(checkLogprobs(AsyncClient(), args.model))
    else:
        asyncio.run(run_benchmark(args.model, args.input_file, args.output_file))