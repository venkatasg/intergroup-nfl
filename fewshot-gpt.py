'''
Get CoT explanations with predictions from GPTs for examples in our dataset
'''
from openai import OpenAI
import ipdb
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import time
import csv
from math import sin,pi
import argparse

random.seed(1)
np.random.seed(1)
rng = np.random.default_rng()

def wp_to_description(row):
    if ((row.win_prob>=0) and (row.win_prob<0.1)):
        return row.opp + " are extremely likely to win."
    elif ((row.win_prob>=0.1) and (row.win_prob<0.25)):
        return row.opp + " are likely to win."
    elif ((row.win_prob>=0.25) and (row.win_prob<0.48)):
        return row.opp + " are slightly likely to win."
    elif ((row.win_prob>=0.52) and (row.win_prob<0.75)):
        return row.team + " are slightly likely to win." 
    elif ((row.win_prob>=0.75) and (row.win_prob<0.9)):
        return row.team + " are likely to win."
    elif ((row.win_prob>=0.9) and (row.win_prob<=1)):
        return row.team + " are extremely likely to win."   
    else:
        return "Both teams are equally likely to win." 
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--model', choices=['gpt-4o', 'gpt-3.5-turbo'], required=True)
    parser.add_argument('--temp', action='store_true')
    parser.add_argument('--folder', required=True)
    parser.add_argument('--seed', required=True)
    
    args = parser.parse_args()
    
    
    with open("/Users/venkat/Research/jessy-openai-key.txt", 'r') as f:
        client = OpenAI(api_key=f.read().strip())
    
    prompt = ""
    with open(args.prompt, 'r') as f:
        prompt = f.read()
    
    comments = []
    df = pd.read_csv('../data/test_data.tsv', sep='\t',  quoting=csv.QUOTE_NONE, escapechar="\\", engine='python')
    ['train']
    df = df[df['split']=='test']
    df['game_state'] = df.apply(lambda x: wp_to_description(x), axis=1)
    comments = df.loc[:, ['comment_id', 'tokenized_comment', 'team', 'opp', 'game_state', 'win_prob']].values.tolist()
    for (comment_id, comment, team, opp, game_state, win_prob) in tqdm(comments):
        full_prompt = (
            prompt +
            "COMMENT: " + str(comment) + "\n" + 
            "IN-GROUP: " + str(team).title() + "\n" + 
            "OUT-GROUP: " + str(opp).title() + "\n"
        )
        if 'ling' in args.prompt:
            full_prompt += (
                "GAME STATE: " + game_state + "\n"
                "REF_EXPRESSIONS: "
            )
        elif 'wp' in args.folder:
            full_prompt += (
                "WIN PROBABILITY: " + str(np.round(win_prob*100, 1)) + "%\n" 
                "REF_EXPRESSIONS: "
            )
        else:
            full_prompt += (
                "REF_EXPRESSIONS: "
            )

        response = client.chat.completions.create(
          model=args.model,
          messages=[{"role": "user", "content": full_prompt}],
          max_tokens=512,
          seed=int(args.seed),
          temperature=(sin(pi*win_prob) if args.temp else 1)
        )
        
        reply = response.choices[0].message.content
        to_write = str(comment_id) + "," + str(reply).replace("\n", " ") + "\n"
        
        with open('model-outputs/' + args.model +'/' + args.folder + '/seed' + args.seed + '_output.txt', 'a') as f:
            f.write(to_write)
        time.sleep(0.5)
