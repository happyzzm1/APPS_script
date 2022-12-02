"""
需要用户自己实现，来提供一些函数接口。
比如load_model，用来加载模型
generate_code，用来生成代码
"""
import argparse

# sample.py包含用户自己的模型和相关函数，此处结构可以自行修改
from sample import load, gen_code


def load_model_tokenizer(model_name):
    """
    由generate_gpt_codes.py调用，用来加载模型和Tokenizer。
    函数参数与返回值由用户自行定义。
    """
    model_args = create_model_args(model_name)
    model, tokenizer, new_model_args = load(model_args)
    return model, tokenizer, new_model_args


def generate_code(raw_text, model, tokenizer, args):
    """
    由generate_gpt_codes.py调用，用来生成代码。
    函数参数与返回值由用户自行定义。
    """
    res = gen_code(raw_text, model, tokenizer, args)
    return res


def create_model_args(model_name):
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-t", "--test_loc", default="~/apps/data_split/test.json", type=str)
    parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("-s", "--start", default=0, type=int)
    parser.add_argument("-e", "--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results")

    parser.add_argument("--qfn", type=str, default=None)  # question_file_name
    parser.add_argument('--model', type=str, default='codegen-350M-mono')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--rng-deterministic', type=bool, default=True)
    parser.add_argument('--p', type=float, default=0.95)
    parser.add_argument('--temp', type=float, default=0.2)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--no-fp16', action="store_false")  # 默认为True
    parser.add_argument('--pad', type=int, default=50256)
    parser.add_argument('--context', type=str, default='def helloworld():')
    parser.add_argument('--times', type=int, default=4)
    args = parser.parse_args()
    args.model = model_name
    return args
