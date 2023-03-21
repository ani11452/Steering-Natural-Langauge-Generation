#from lib2to3.pgen2 import token
import torch
import statistics
from transformers import AutoTokenizer, GPT2LMHeadModel
import numpy as np

from testharness import *
from generator import *
from scoring import *

NUM_TOK_TO_GEN = 25
NUM_BEAMS = 1

def main():

    # Initialize base GPT2
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")


    # Initialize Word Bank
    word_bank = WordBank(num_clusters=5)
    word_bank.load_word_vecs()
    word_bank.load_word_bank()
    word_bank.create_clusters()
    word_bank.visualize_wbank()


    th = TestHarness()
    th.load_prompts()

    # Ensure that Generator works
    prompt = 'In the garden, I usually'
    gen = Generator(word_bank) # set default params
    ret = gen.beam_search(prompt, num_beams=1, tokens_to_generate=20)[0][0]


    DEBUG = 50
    debug_ctr = 0
    # first, load the jsonl file of the toxicity prompts


    custom_output_scores = []
    gpt_output_scores = []

    for result in th.challenging_prompts:
        #result = json.loads(json_str)
        prompt = result['prompt']['text']
        
        prompt_len = len(prompt)
        custom_res = gen.beam_search(prompt, num_beams=NUM_BEAMS, tokens_to_generate=NUM_TOK_TO_GEN)[0][0]

        inputs = tokenizer(prompt, return_tensors="pt")
        gpt2_output = tokenizer.batch_decode(model.generate(**inputs, num_beams=NUM_BEAMS, do_sample=True, max_new_tokens=NUM_TOK_TO_GEN, pad_token_id=50256))[0]
        
        print(f'PROMPT: {prompt}')
        print(f'CUSTOM_OUTPUT: {custom_res[prompt_len:]}')
        print(f'GPT2_OUTPUT: {gpt2_output[prompt_len:]}')
        
        
        print('\n------------------')
        base_score = th.get_perspective_score(gpt2_output[prompt_len:])
        custom_score = th.get_perspective_score(custom_res[prompt_len:])
        
        custom_output_scores.append(custom_score)
        gpt_output_scores.append(base_score)
        
        debug_ctr += 1
        if debug_ctr == DEBUG:
            break

        print(f'CUSTOM MEAN: {statistics.mean(custom_output_scores)}')
        print(f'GPT2 MEAN: {statistics.mean(gpt_output_scores)}')


    print(ret)


if __name__ == "__main__":
    main()