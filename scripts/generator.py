import torch
import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel
from WordBank import WordBank
from scoring import *
import os

class Generator:
    def __init__(self,
                 wb,
                 score_mode='dist',
                 target='far',
                 weight=0.33,
                 specificity=3,
                 top_p_val=0.73,
                 top_k_val=100,
                 search_space_size=2):
        # Initialize model and tokenizer
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.WordBank = wb

        self.SCORE_MODE = score_mode
        self.TARGET = target
        self.WEIGHT = weight
        self.SPECIFICITY = specificity
        self.SEARCH_SPACE_NUM = search_space_size
        
        self.top_p_val = top_p_val
        self.top_k_val = top_k_val

        self.scoring_function = None
        if score_mode == 'dot':
            self.scoring_function = dot_score
        elif score_mode == 'dist':
            self.scoring_function = distance_score
        elif score_mode == 'stat':
            self.scoring_function = statistics
    """
    def sample_idx(sorted_vals):
        #softmax_scores = sorted_vals.softmax(dim=-1).detach().numpy()

        #ret = np.random.choice(softmax_scores, p=softmax_scores)
        #print(ret)
        #return np.where(softmax_scores==ret)[0][0]

        normalized_scores = torch.div(sorted_vals, torch.sum(sorted_vals))
        # normalized_scores = sorted_vals.numpy() / np.sum(sorted_vals.numpy())

        ret = torch.multinomial(normalized_scores, 1)
        # ret = np.random.choice(normalized_scores, p=normalized_scores)

        return ret
        # return np.where(normalized_scores==ret)[0][0]
    """ 
    
    """
    samples from search space
    """
    def sample_idx(self, sorted_vals):

        # Uncomment below to softmax before sampling
        #softmax_scores = sorted_vals.softmax(dim=-1).detach().numpy()
        
        #ret = np.random.choice(softmax_scores, p=softmax_scores)
        #print(ret)
        #return np.where(softmax_scores==ret)[0][0]
        normalized_scores = sorted_vals.numpy() / np.sum(sorted_vals.numpy())
        
        ret = np.random.choice(normalized_scores, p=normalized_scores)
        return np.where(normalized_scores==ret)[0][0]

    def top_p(self, tup, p):
        sorted_vals, indices = tup
        trunc_sorted_vals = []
        sum_so_far = 0
        
        for val in sorted_vals:
            sum_so_far += val
            trunc_sorted_vals.append(val)
            if sum_so_far >= p:
                break

        sorted_vals = torch.FloatTensor(trunc_sorted_vals)
        indices = indices[:len(trunc_sorted_vals)]
        # print("Sorted_vals top_p: ", sorted_vals)
        # print("Indices indices: ", indices)
        return sorted_vals, indices

    def get_embeddings(self, sorted_vals, indices, top_embeddings):
        for word_idx in range(len(indices)):
            word = self.tokenizer.decode(indices[word_idx])
            if word.strip().lower() not in self.WordBank.GloVe.keys():
                sorted_vals[word_idx] = 0  # disregard this token
                top_embeddings.append(self.WordBank.GloVe['failure']) # TOFIX
            else:
                if word[1:].isalpha() or word.isalpha():
                    top_embeddings.append(self.WordBank.GloVe[word.strip().lower()])
                else:
                    top_embeddings.append(self.WordBank.GloVe[word.strip()])
    

    def rerank(self, sorted_vals, indices, dist_score):
        eps = 0.00000000000001
        # re-rank the weightings, factor in dist_score
        
        dist_score = torch.FloatTensor(dist_score)
        
        if self.TARGET == 'close':
            # a smaller value is better
            dist_score = (1 / (dist_score + eps)) * self.SPECIFICITY
            # sorted_vals = torch.log(sorted_vals) + WEIGHT * torch.log(dist_score.softmax(dim=-1))
            
            sorted_vals = (1 - self.WEIGHT) * sorted_vals + self.WEIGHT * dist_score.softmax(dim=-1)
            #sorted_vals += WEIGHT * dist_score
            
            # sorted_vals += dist_score.softmax(dim=-1)
            # sorted_vals += (((1 / (dist_score + eps)) ** exponent) * hyper_weight)
        elif self.TARGET == 'far':
            # a larger value is better
            dist_score = dist_score * self.SPECIFICITY
                        
            sorted_vals = (1 - self.WEIGHT) * sorted_vals + self.WEIGHT * dist_score.softmax(dim=-1)    
            #sorted_vals += WEIGHT * dist_score 
            
            # sorted_vals += (((dist_score / 100) ** exponent) * hyper_weight)
        else:
            print('MODE error')
        
        # sorted_vals = sorted_vals.softmax(dim=-1)
        sort_indices = torch.argsort(sorted_vals)
        sorted_vals = sorted_vals[sort_indices]
        final_ranked_indices = indices[sort_indices]
        #final_ranked_indices = [indices[s] for s in sort_indices]
        
        return final_ranked_indices, sorted_vals

    # generate one word given a prompt_beam


    def generate_one(self, prompts, done, index_tracker):
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        next_token_scores = logits[:, -1, :].softmax(dim=-1)
        sorted_vals, indices = torch.sort(next_token_scores, descending=True)

        if self.top_p_val > 0:
            x = zip(sorted_vals, indices)
            res = [self.top_p(tup, p=self.top_p_val) for tup in x]
        else:
            sorted_vals = sorted_vals[:, -self.top_k_val:].cpu().detach()
            indices = indices[:, -self.top_k_val:].detach()
            res = list(zip(sorted_vals, indices))

        top_embeddings = []

        for tup in res:
            new = []
            self.get_embeddings(tup[0], tup[1], new)
            top_embeddings.append(new)

        all_scores = []
        for prompt in top_embeddings:
            dist_score = [self.scoring_function(embed, self.WordBank.wb_embeddings, self.WordBank.clusters, self.WordBank.n_clusters) for embed in prompt]
            all_scores.append(dist_score)

        final_ranked_indices = []
        sorted_vals = []

        for i, tup in enumerate(res):
            temp = self.rerank(tup[0], tup[1], all_scores[i])
            final_ranked_indices.append(temp[0][-self.SEARCH_SPACE_NUM:])
            sorted_vals.append(temp[1][-self.SEARCH_SPACE_NUM:])

        for i, sort in enumerate(sorted_vals):
            idx = self.sample_idx(sort[:])
            decode = int(final_ranked_indices[i][-idx])
            tok = self.tokenizer.decode(decode)

            if tok == self.tokenizer.eos_token:
                done.append(prompts[i])
                index_tracker.append(i)
            else:
                prompts[i] += tok

        return prompts, done, index_tracker

    def beam_search(self, prompts, tokens_to_generate=25):
        res = []
        done = []
        index_tracker = []
        for i in range(tokens_to_generate):
            prompts, done, index_tracker = self.generate_one(prompts, done, index_tracker)
            res += done
            new_prompts = []
            for idx in range(len(prompts)):
                if idx not in index_tracker:
                    new_prompts.append(prompts[idx])
            prompts = new_prompts
            done = []
            index_tracker = []
        res += prompts
        return res
