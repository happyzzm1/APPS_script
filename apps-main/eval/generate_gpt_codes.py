"""
Run a trained model to generate Python code.
"""
import io
import json
import random
import os
import pprint
import time


from reindent import run as run_reindent
from tqdm import tqdm
# for timing and debugging
from datetime import datetime, date

from generate_util import load_model_tokenizer, generate_code


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    GitHub dataset was indented
    只在生成prompt时有用，且必须peeking>0
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr,
        ret,
        config={
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    return ret.getvalue()


def generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None):
    """
    生成模型的prompt，用来让模型生成代码
    prompt结构：
    "\\nQUESTION:\\n" + <question> + <starter_code> + "\\nUse Standard Input format"("\\nUse Call-Based format") +
    "\\nANSWER:\\n"
    """
    _input = "\nQUESTION:\n"
    with open(prompt_path, "r") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    # 如果存在starter代码，则附加到prompt后面
    if starter_path is not None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data  # + "\n"
        _input += data
    else:
        # _input += "\n\n"
        pass

    with open(test_case_path, "r") as f:
        data = json.load(f)
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"  # \n"
    else:
        _input += "\nUse Call-Based format"  # \n"

    _input += "\nANSWER:\n"

    if args.peeking > 0.0:
        # Need to do some peeking.
        # Read one example solution
        with open(solutions_path, 'r') as f:
            sols = json.load(f)

        # Choose the shortest solution for the model to use.
        # This is so we can conserve tokens (1024 max)
        # sample_sol = min(sols, key=len)

        # # Add args.peeking% of that solution to the prompt
        # sample_sol_token_ids = tokenizer.encode(sample_sol, verbose=False)
        # num_to_keep = int(len(sample_sol_token_ids) * args.peeking)
        # sample_sol_token_ids = sample_sol_token_ids[:num_to_keep]
        # _input += tokenizer.decode(sample_sol_token_ids)

        # Alternatively take a random solution
        sample_sol = random.choice(sols)
        rand_sol = reindent_code(sample_sol)
        rand_sol = tokenizer.encode(rand_sol, verbose=False)
        tokens_taken = int(args.peek_frac * len(rand_sol))
        rand_sol = rand_sol[:tokens_taken]
        _input += tokenizer.decode(rand_sol)
    else:
        sample_sol = None

    return _input, sample_sol


def main(args):
    args_dict = vars(args)
    print(pprint.pformat(args_dict))

    with open(args.test_loc, "r") as f:
        problems = json.load(f)  # 一个列表，里面的元素为问题所在的路径
    problems = sorted(problems)  # 固定序列顺序

    generated_codes = {}  # 保存生成的代码, key为问题序号, value为一个list, 保存所有生成的代码结果
    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    if not args.end:
        codes_loc = os.path.join(args.save, f"all_codes.json")
    else:
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")

    # Only do the problems that are specified.
    if args.index:
        problems = [problems[args.index]]
    else:
        if args.start > len(problems) or args.start < 0:
            print(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = problems[start:end]
    # print(problems)

    # todo：初始化自定义模型和Tokenizer
    print("start to load")
    model, tokenizer, model_args = load_model_tokenizer(args.model)
    model_args.batch_size = args.batch_size
    model_args.max_length = args.max_length
    model_args.temp = args.temp

    # main eval loop
    print("start to generate code")
    for index, problem in enumerate(tqdm(problems)):
        prob_path = os.path.join(args.root, problem)
        if args.debug:
            print(f"problem path = {prob_path}")

        test_case_path = os.path.join(prob_path, "input_output.json")  # 问题的输入输出所在路径
        # print(test_case_path)
        prompt_path = os.path.join(prob_path, "question.txt")  # 问题文本所在路径
        starter_path = os.path.join(prob_path, "starter_code.py")
        solutions_path = os.path.join(prob_path, "solutions.json")
        if not os.path.exists(starter_path):
            starter_path = None
        if not os.path.exists(test_case_path) or not os.path.exists(prompt_path):
            print("not exist")
            continue

        # 获取输入文本；sample_sol默认为None
        if args.qfn is None:
            prompt_text, sample_sol = generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer,
                                                      starter_path)
        # 指定question的文件名
        else:
            prompt_text, sample_sol = generate_prompt(args, test_case_path, os.path.join(prob_path, "question_en.txt"),
                                                      solutions_path, tokenizer, starter_path)
        if args.debug:
            print("PROMPT_TEXT:")
            print(prompt_text)

        # Feed this into the model.
        start = time.time()
        try:
            res_list = generate_code(raw_text=prompt_text, model=model, tokenizer=tokenizer, args=model_args)
            output_str = res_list[0]
        except Exception as e:
            print("Unexpected exception in generating solution")
            print(e)
            # Default to empty string on errors
            res_list = []
            output_str = ""
        end = time.time()

        for i in range(len(res_list)):
            if args.peeking == 1.0:
                res_list[i] = sample_sol
            # elif len(res_list[i]):
            #     # 选择ANSWER后面的部分，并去除无用字符
            #     split_res = res_list[i].split("ANSWER:\n")
            #     if len(split_res) > 1:
            #         res_list[i] = split_res[1].replace("<|endoftext|>", "")
            #     else:
            #         res_list[i] = res_list[i].replace("<|endoftext|>", "")

        # Save the generated sol
        generated_codes[index + args.start] = res_list  # 存储为list，长度为总sample数量

        if args.debug:
            print(f"Generation time: {end - start}")
            print(f"Generated output string(show one case):")
            print(output_str)
            print("------------------------------------------------------------")

    with open(codes_loc, "w") as f:
        json.dump(generated_codes, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a trained model to generate Python code.")
    parser.add_argument("-r", "--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-t", "--test_loc", default="~/apps/data_split/test.json", type=str)
    parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("-s", "--start", default=0, type=int)
    parser.add_argument("-e", "--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results")
    parser.add_argument("--qfn", type=str, default=None)  # question_file_name
    # parser.add_argument("--arch", default="gpt2", choices=transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST)
    # parser.add_argument("--num-beams", default=5, type=int)
    # 用于模型的参数，非必须
    parser.add_argument('--model', type=str, default='codegen-350M-mono')
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.2)

    test_args = parser.parse_args()

    main(test_args)
