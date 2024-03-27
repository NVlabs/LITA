# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import argparse
import json
import os

import openai
import time

NUM_SECONDS_TO_SLEEP = 0.5


def request_openai(gpt_prompt, max_tokens):

    message=[]
    message.append({"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."})
    message.append({"role": "user", "content": gpt_prompt})
    response = openai.ChatCompletion.create(
        model='gpt-4-0613',
        messages = message,
        max_tokens=max_tokens,
        temperature=0.2,
        request_timeout=60,
    )
    response = response['choices'][0]['message']['content']
    return response


def get_eval(content: str, max_tokens: int):
    while True:
        try:
            response = request_openai(content, max_tokens)
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer')
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    # rule
    with open(args.rule, 'r') as f:
        prompt = f.read()
    role = "Assistant"

    # answer and question
    answers = json.load(open(args.answer, 'r'))

    # context
    contexts = json.load(open(args.context, 'r'))
    def get_context(qid):
        vid = qid[:11]
        captions = contexts['v_' + vid]
        context = ""
        for i, (start, end) in enumerate(captions['timestamps']):
            sentence = captions['sentences'][i].strip()
            context += f"<{start}> <{end}> {sentence}\n"
        return context.strip()

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')
    idx = 0
    for answer in answers:
        ques = answer['prompt']
        ans1 = answer['text_gnd']
        ans2 = answer['text_out']

        cap_str = get_context(answer['question_id'])
        content = (f'[Context]\n{cap_str}\n\n'
                   f'[Question]\n{ques}\n\n'
                   f'[{role} 1]\n{ans1}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        cur_js = {
            'id': idx+1,
            'question_id': answer['question_id'],
            'answer_id': answer['answer_id'],
        }
        if idx >= len(cur_reviews):
            review = get_eval(content, args.max_tokens)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        idx += 1
        print(idx)
    review_file.close()
