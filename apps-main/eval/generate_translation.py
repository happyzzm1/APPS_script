import json
import os
import pprint
import time

from WebOTS import get_result
from tqdm import tqdm
# for timing and debugging
from datetime import datetime, date


def main(args):
    args_dict = vars(args)
    print(pprint.pformat(args_dict))

    with open(args.test_loc, "r") as f:
        problems = json.load(f)  # 一个列表，里面的元素为问题所在的路径
    problems = sorted(problems)  # 固定序列顺序

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

    # main eval loop
    host = "ntrans.xfyun.cn"
    print("start to generate translation")
    for index, problem in enumerate(tqdm(problems)):
        start = time.time()
        prob_path = os.path.join(args.root, problem)
        if args.debug:
            print(f"problem path = {prob_path}")

        prompt_path = os.path.join(prob_path, "question.txt")  # 源问题文本所在路径

        for i in range(3):
            try:
                with open(prompt_path, "r") as f:
                    data = f.readlines()
                    data = "".join(data)
                    raw_prompt = data  # 源语言为en

                    gClass = get_result(host)
                    gClass.Text = raw_prompt
                    gClass.set_from_language('en')
                    gClass.set_to_language('cn')
                    cn_translation = gClass.call_url()
            except Exception as e:
                print("Unexpected exception in generating cn translation")
                print(e)
                cn_translation = None
            if cn_translation is not None:
                break

        if cn_translation is not None:
            with open(os.path.join(prob_path, "question_cn.txt"), mode='w', encoding='utf-8') as f:
                f.write(cn_translation)
            print(f"generate cn translation successfully on: {problem}")
            if args.debug:
                print("cn_translation:")
                print(cn_translation)
        else:
            print(f"generate cn translation failed on: {problem}")
            continue

        en_translation = None
        for i in range(3):
            try:
                raw_prompt = cn_translation  # 源语言为cn

                gClass = get_result(host)
                gClass.Text = raw_prompt
                gClass.set_from_language('cn')
                gClass.set_to_language('en')
                en_translation = gClass.call_url()
            except Exception as e:
                print("Unexpected exception in generating en translation")
                print(e)
                en_translation = None
            if en_translation is not None:
                break

        if en_translation is not None:
            with open(os.path.join(prob_path, "question_en.txt"), mode='w', encoding='utf-8') as f:
                f.write(en_translation)
            print(f"generate en translation successfully on: {problem}")
            if args.debug:
                print("en_translation:")
                print(en_translation)
        else:
            print(f"generate en translation failed on: {problem}")
            continue

        end = time.time()
        if args.debug:
            print(f"Generation time: {end - start}")
            print("------------------------------------------------------------")

    print(f"generate translation successfully on {args.test_loc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a trained model to generate Python code.")
    parser.add_argument("-r", "--root", default="./", type=str, help="where the data is stored.")
    parser.add_argument("-t", "--test_loc", default="~/apps/data_split/test.json", type=str)
    parser.add_argument("-s", "--start", default=0, type=int)
    parser.add_argument("-e", "--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")

    test_args = parser.parse_args()

    main(test_args)
