import re

def extract_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    print(f"Solution: {solution_str}, Ground Truth: {ground_truth}")
    answer = extract_answer(solution_str)
    if str(answer) == str(ground_truth):
        return 1
    else:
        return -1