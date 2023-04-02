# Steering Natural Language Generation by Optimizing Vector-to-Cluster Distance

This repository includes the submission code for the Stanford CS224n Custom Project for Winter 2023. 

## Abstract
Language models (LMs) are often trained on large corpora of web text, which often contain toxic or harmful text. This behavior can thus be extracted from LM outputs through prompting, hindering their safe and widespread usage. We propose and evaluate a novel weighted-decoding approach to steering Natural Language Generation (NLG), and address the issue of toxic text generation to evaluate the effectiveness of our approach. This model requires no additional training, can be set to steer towards or away from any topic or sentiment in a matter of seconds, requires no additional training, explicit blacklist, and is computationally efficient compared to other decoder-based models. Given a set of ~55 words to form a representation of a target or goal, our model automatically creates sub-clusters and influences generation. We develop an interface for our model which sits on top of HuggingFace's GPT2LMHeadModel and provide a testing suite to evaluate model toxicity. To quantitatively explore the effectiveness of this proposed method, we attempt to steer prompted generation away from as well as towards toxic text, and compare the results against GPT2's performance on the same prompts. We find that our model does reduce toxic generation, but its success at this target is not as significant as other more invasive or computationally expensive methods. However, it does successfully allow for outputs to lean much farther into their predisposed tendencies for a given set of prompts, which is particularly relevant in settings where certain types of prompts and model behaviors are expected.

See SteeringNLGReport.pdf in repo for full write-up
