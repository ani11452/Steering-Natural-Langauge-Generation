import torch
import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel
from WordBank import WordBank
from scoring import *
import os

class Generator:
    def __init__(   self, 
                    WordBank,
                    score_mode='dist',
                    target='far',
                    weight=0.5,
                    specificity=5, 
                    top_p_val=0.5, 
                    top_k_val=10, 
                    search_space_size=5):
        # Initialize model and tokenizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.WordBank = WordBank
    
        self.SCORE_MODE = score_mode
        self.TARGET = target
        self.WEIGHT = weight
        self.SPECIFICITY = specificity
        self.SEARCH_SPACE_NUM = search_space_size

        self.top_p_val = top_p_val
        self.top_k_val = top_k_val
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
        return np.where(normalized_scores==ret)[0][0], normalized_scores

    
    def top_p(self, sorted_vals, indices, p):
        trunc_sorted_vals = []
        sum_so_far = 0
        # reversed?
        for val in reversed(sorted_vals):
            sum_so_far += val
            trunc_sorted_vals.append(val)
            if sum_so_far > p:
                break
        sorted_vals = torch.FloatTensor(trunc_sorted_vals)
        indices = indices[-len(sorted_vals):]
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
        
        if (self.SCORE_MODE != 'dot' and self.TARGET == 'close') or (self.SCORE_MODE == 'dot' and self.TARGET == 'far'):
            # a smaller value is better
            dist_score = (1 / (dist_score + eps)) * self.SPECIFICITY
            # sorted_vals = torch.log(sorted_vals) + WEIGHT * torch.log(dist_score.softmax(dim=-1))
            
            sorted_vals = (1 - self.WEIGHT) * sorted_vals + self.WEIGHT * dist_score.softmax(dim=-1)
            #sorted_vals += WEIGHT * dist_score
            
            # sorted_vals += dist_score.softmax(dim=-1)
            # sorted_vals += (((1 / (dist_score + eps)) ** exponent) * hyper_weight)
        elif (self.SCORE_MODE != 'dot' and self.TARGET == 'far') or (self.SCORE_MODE == 'dot' and self.TARGET == 'close'):
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

    def generate_one(self, prompt_beam, idx):
        prompt = prompt_beam[0]
        score = prompt_beam[1]
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model(**inputs, labels=inputs["input_ids"], use_cache=False)
        #loss = outputs.loss
        logits = outputs.logits
        next_token_scores = logits[:, -1, :].softmax(dim=-1)

        sorted_vals, indices = torch.sort(next_token_scores[0])
        
        # Calculate Top-P
        if self.top_p_val > 0:
            sorted_vals, indices = self.top_p(sorted_vals[:], indices[:], self.top_p_val)
        else:
            # else, we just do top-k
            sorted_vals = sorted_vals[-self.top_k_val:]
            indices = indices[-self.top_k_val:]

        #print([tokenizer.decode(word) for word in indices])

        top_embeddings = [] 
        self.get_embeddings(sorted_vals, indices, top_embeddings)

        #log = open("log.txt", "a")
        #log.write('PRE-RERANK:\n')
        #print_words(reversed(sorted_vals), reversed(indices), log)

        #top_embeddings = [GloVe[tokenizer.decode(word).strip().lower()] for word in indices]

        # calculate distance to cluster
        if self.SCORE_MODE == 'dist':
            dist_score = [  distance_score(embed, 
                            self.WordBank.wb_embeddings, 
                            self.WordBank.clusters, 
                            self.WordBank.n_clusters) for embed in top_embeddings]
        
        # all other modes are deprecated for now
       # elif self.SCORE_MODE == 'dot':


        # sorted_vals are softmaxed logits
        final_ranked_indices, sorted_vals = self.rerank(sorted_vals, indices, dist_score)

        # replace -1 with -idx for true beam search
        # add variability instead for true decoding (TODO)
        # TODO normalization
        
        #log.write('POST-RERANK:\n')
        #print_words(sorted_vals, final_ranked_indices, log)
        
        # must sample index if we use top_p

        ###
        # TOP-K Search Space
        sorted_vals = sorted_vals[-self.SEARCH_SPACE_NUM:]
        final_ranked_indices = final_ranked_indices[-self.SEARCH_SPACE_NUM:]
        ###
        
        ###
        # TOP-P Search Space
        #sorted_vals, final_ranked_indices = top_p(sorted_vals[:], final_ranked_indices[:], top_p_val)
        #sorted_vals = torch.flip(sorted_vals, [-1])
        ###
        

        # COMMENT IN TO ENABLE LOGGING
        if self.top_p_val > 0:
            #log.write('RERANK SPACE:\n')
            #print_words(sorted_vals.softmax(dim=-1), final_ranked_indices, log)
            #print_words(sort)
            idx, norm_scores = self.sample_idx(sorted_vals[:])
            #print_words(norm_scores, final_ranked_indices, log)
        
        best_word = self.tokenizer.decode(final_ranked_indices[-idx])
        prompt += best_word

        # add normalization by length


        #return [prompt, score + s_vals[-idx].detach().numpy()]
        #log.write('--------------------------\n')
        #log.close()
        #(1/len(prompt)+1) *
        # adjusted to ensure that we keep generating more words.
        # otherwise, we stop almost immediately since the probability of the
        # second word is 20%, the probability of the first guessed word was ~80%
        #print (sorted_vals[-idx].detach().numpy())
        #print (len(prompt) + sorted_vals[-idx].detach().numpy())

        # add score here! TODO
        return [prompt, (len(prompt)*4) + sorted_vals[-idx].detach().numpy()] # subject to change

    def beam_search(self, prompt, num_beams=1, tokens_to_generate=25):
        beams = [[prompt, 0]]

        #if os.path.exists("log.txt"):
            # delete the file
        #    os.remove("log.txt")
        
        for token_num in range(tokens_to_generate):
            #print(token_num)
            num_to_investigate = len(beams)
            for beam_idx in range(num_to_investigate):
                prompt_beam = beams[beam_idx]
                for position in range(num_beams):
                    ret = self.generate_one(prompt_beam, position)
                    beams.append(ret)

            # or normalize scores by length here
            beams = sorted(beams, key=lambda x: -x[1])
            
            #FORCE MAX LENGTH GENERATION: beams = sorted(beams, key=lambda x: -len(x[0]))
            #print(beams)
            #print('-------------')
            beams = beams[:num_beams]
        return beams