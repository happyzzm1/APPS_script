"""
Run solutions from one problem.
"""
import argparse
import json
import numpy as np
import os
import pprint
import multiprocessing
import testing_util as test_util

# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from types import SimpleNamespace
from typing import Dict

EXAMPLE_RESULTS = {"0": [[-2]], "1": [[False, False, False]], "2": [[True, True]],
                   "3": [[False, True, False, True, False, False, False, True, False, True, False, True, True, True,
                          False, True]],
                   "4": [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]}
EXAMPLE_ARGS = SimpleNamespace(debug=True)
TIMEOUT = 10


def print_results(results: Dict, args: argparse.Namespace = None):
    """
    Given the results evaluated against the testcases we output some statistics.

    # >>> print_results(EXAMPLE_RESULTS, EXAMPLE_ARGS)
    number of compile errors = 1 avg = 0.2
    number of runtime errors = 1 avg = 0.2
    number of test cases run = 5
    Test Case Average (average accuracy over problems) = 0.3
    Strict Accuracy (all test cases passed / total problems) = 0.2
    """
    res = []
    per_prob_res = []
    all_correct = []
    for index in results:
        res_list = results[index]
        for i in range(len(res_list)):
            sample_res = np.asarray(res_list[i])  # [-2]; [False, True, False]
            problem_results = np.asarray(res_list[i])
            res.extend(problem_results)
            per_prob_res.append(np.mean(problem_results > 0))
            all_correct.append(np.all(problem_results > 0))

    # We count both compile errors and runtime errors for multiple tests as one error.
    compile_errors = len([e for e in res if -2 in e])
    runtime_errors = len([e for e in res if -1 in e])
    total_testcases = len(res)
    if args and args.debug:
        print(f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases}")
        print(f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}")
        print(f"number of test cases run = {total_testcases}")

    print(f"Test Case Average (average accuracy over problems) = {np.mean(per_prob_res)}")
    print(f"Strict Accuracy (all test cases passed / total problems) = {np.mean(all_correct)}")


def print_pass_k(results: Dict, args: argparse.Namespace = None):
    """
    Given the results evaluated against the testcases we output pass@k.
    default k is 1, 10
    todo:可修改的k值

    # >>> print_results(EXAMPLE_RESULTS, EXAMPLE_ARGS)
    number of compile errors = 1 avg = 0.2
    number of runtime errors = 1 avg = 0.2
    number of test cases run = 5
    Test Case Average (average accuracy over problems) = 0.3
    Strict Accuracy (all test cases passed / total problems) = 0.2
    """
    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    new_results = {}
    estimator_data = {}
    res = []
    per_prob_res = []
    all_correct = []
    for index in results:
        new_res_list = []
        res_list = results[index]  # 一个列表，每个元素都是一个列表，保存test case的结果
        for i in range(len(res_list)):
            sample_res = np.asarray(res_list[i])  # [-2]; [False, True, False]
            if -2 in sample_res or -1 in sample_res or False in sample_res:
                new_res_list.append(0)
            else:
                new_res_list.append(1)
        new_results[index] = new_res_list  # 一个列表，每个元素都是0/1值，代表函数功能正确性，长度为采样数量
        estimator_data[index] = {"correct_num": sum(new_res_list), "sample_num": len(new_res_list)}
    k_list = [1, 10]
    pass_at_k = {f"pass@{k}": np.array([estimator(estimator_data[index]['sample_num'],
                                                  estimator_data[index]['correct_num'], k)
                                        for index in estimator_data]).mean() for k in k_list}

    print(f"pass@k = {pass_at_k}")


def check_correctness(prob_path, generation, timeout, debug):
    """Check correctness of one code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    def _temp_run(prob_path, generation, debug, result):
        result.append(test_util.run_test(prob_path=prob_path, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(prob_path, generation, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        # Remark: ideally we would consider that all tests failed, but we can't access number of tests here easily,
        # so we use 21=the average number of tests for a sample in the test split instead
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]  # [[-1, -1, -1...]]
        if debug:
            print(f"global timeout")
    return result[0]  # 返回一个一维列表，表明test cases的结果


def eval_and_save_problems(args):
    with open(args.test_loc, "r") as f:
        problems = sorted(json.load(f))
    print(len(problems))

    results = {}
    codes_loc = os.path.join(args.save, f"all_codes.json")
    if not os.path.exists(codes_loc):
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes.json")

    if os.path.exists(codes_loc):
        results_loc = os.path.join(args.save, f"all_results.json")
    else:
        results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.json")
    print(codes_loc, results_loc)

    with open(codes_loc, "r") as f:
        generated_codes = json.load(f)  # 读入生成的代码

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

    if args.stop_early:
        problems = problems[:args.stop_early]

    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        try:
            if args.debug:
                print(f"\n\nproblem path = {problem}")
            sample_list = generated_codes[str(index + args.start)]  # 读取与问题对应的代码列表
            # output_str = generated_codes[str(index + args.start)]  # 读取与问题对应的代码
        except:
            print("CANNOT FIND OUTPUT_STR FOR", problem)
            continue
        prob_path = os.path.join(args.root, problem)

        # with open(os.path.join(prob_path, "solutions.json"), "r") as f:
        #     sols = json.load(f)

        if not os.path.exists(args.save):
            os.makedirs(args.save)

        res = []
        # 可以保存多个sample答案
        # for o_idx, o in enumerate(output_str):
        for o_idx, o in enumerate(sample_list):
            if args.debug:
                print(f"\nTesting solution {o_idx}")
            curr_res = [-2]
            try:
                # 检查正确性
                curr_res = check_correctness(prob_path=prob_path, generation=o, timeout=TIMEOUT, debug=args.debug)
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                        e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                if not np.all(curr_res):
                    print(f"Results were not all True: {curr_res}")
            except Exception as e:
                print(f"test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_res, list)
                res.append(curr_res)

        if args.debug:
            print(f"\nHow to read results [-2] = compile error, [-1] = runtime error "
                  f"[False] = failed test case [True] = passed test case")
            # print(f"results = {res}")

        results[index + args.start + args.index] = res  # [[],[],...,[]]

        with open(results_loc, "w") as f:
            try:
                f.write(json.dumps(results))
            except Exception as e:
                import pdb
                pdb.set_trace()
                print(f"didn't save problem due to {e}")

    return results


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if args.print_results:
        results = {}
        codes_loc = os.path.join(args.save, f"all_codes.json")
        if os.path.exists(codes_loc):
            results_loc = os.path.join(args.save, f"all_results.json")
        else:
            results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.json")
        with open(results_loc, "r") as f:
            results = json.load(f)
    else:
        results = eval_and_save_problems(args)

    # print_results(results, args)
    print_pass_k(results, args)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    parser.add_argument("-t", "--test_loc", default="../data_split/test.json", type=str,
                        help="path to the json containing problem paths to be evaluated.")
    parser.add_argument("-r", "--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-s", "--start", default=0, type=int)
    parser.add_argument("-e", "--end", default=None, type=int,
                        help="If you want to evaluate a subset of problems specify start and ending index. "
                             "File with start and ending prefix must exist typically used with batch evaluation.")
    parser.add_argument("-i", "--index", default=0, type=int)
    parser.add_argument("-p", "--print_results", action="store_true",
                        help="If you have already evaluated the results and only want to print them.")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--save", type=str, default="./results",
                        help="Where the evaluated data is loaded from and results saved to.")
    parser.add_argument("--stop-early", default=None, type=int)

    args = parser.parse_args()

    main(args)
