'''
Get CoT explanations from GPTs for our dataset - with and without WP
'''
from openai import OpenAI
import ipdb
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import time
import csv

random.seed(1)
np.random.seed(1)

if __name__=="__main__":
    with open("/Users/venkat/Research/jessy-openai-key.txt", 'r') as f:
        client = OpenAI(api_key=f.read().strip())
    
    prompt = ""
    with open('prompts/explanations+wp.txt', 'r') as f:
        prompt = f.read()
    
    comments = []
    df = pd.read_csv('../data/test_data.tsv', sep='\t',  quoting=csv.QUOTE_NONE, escapechar="\\", engine='python')
    ['train']
    # df = df[df['split']=='test']
    # Only generate explanations for comments that need some explanation
    df = df[~(df.tagged_comment==df.tokenized_comment)].reset_index(drop=True).sample(frac=1, random_state=1)
    df = df[df['comment_id']=='hdhhhne']
    comments = df.loc[:, ['comment_id', 'tokenized_comment', 'tagged_comment', 'ref_expressions', 'team', 'opp', 'win_prob']].values.tolist()
    for (comment_id, comment, tagged_comment, ref_expressions, team, opp, win_prob) in tqdm(comments):
        full_prompt = (
            prompt + "\n" +
            "COMMENT: " + str(comment) + "\n" + 
            "IN-GROUP: " + str(team).title() + "\n" + 
            "OUT-GROUP: " + str(opp).title() + "\n" + 
            "WIN PROBABILITY: " + str(np.round(win_prob*100, 1)) + "%\n" +
            "TARGET: " + str(tagged_comment) + "\n" + 
            "REF_EXPRESSIONS: " + str(ref_expressions) + "\n" +
            "EXPLANATION: "
        )
        
        response = client.chat.completions.create(
          model="gpt-4o",
          messages=[{"role": "user", "content": full_prompt}],
          max_tokens=256,
          seed=1
        )

        reply = response.choices[0].message.content
        to_write = str(comment_id) + "," + str(reply) + "\n"
        print(reply)
        # with open('model-outputs/gpt/explanations.txt', 'a') as f:
        #     f.write(to_write)
        time.sleep(0.5)
