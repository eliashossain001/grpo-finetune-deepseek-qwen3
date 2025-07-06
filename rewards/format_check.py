import re

reasoning_start = "<think>"
reasoning_end = "</think>"
solution_end_regex = rf"{reasoning_end}(.*)"
match_format = re.compile(solution_end_regex, re.DOTALL)

def match_format_exactly(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response): score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        score = 0
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        scores.append(score)
    return scores
