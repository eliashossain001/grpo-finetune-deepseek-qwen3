import re

match_format = re.compile(r"</think>(.*)", re.DOTALL)
match_numbers = re.compile(r".*?[\s]{0,}([-]?[\d\.\,]{1,})", flags=re.MULTILINE | re.DOTALL)

def check_answer(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    guesses = [match_format.search(r).group(1) if match_format.search(r) else None for r in responses]

    scores = []
    for g, a in zip(guesses, answer):
        score = 0
        if g is None: scores.append(-2.0); continue
        if g == a: score += 5.0
        elif g.strip() == a.strip(): score += 3.5
        else:
            try:
                ratio = float(g) / float(a)
                score += 2.0 if 0.9 <= ratio <= 1.1 else 1.5 if 0.8 <= ratio <= 1.2 else -2.5
            except: score -= 4.5
        scores.append(score)
    return scores

def check_numbers(prompts, completions, answer, **kwargs):
    responses = [c[0]["content"] for c in completions]
    guesses = [match_numbers.search(r).group(1) if match_numbers.search(r) else None for r in responses]
    scores = []

    for g, a in zip(guesses, answer):
        try:
            a, g = float(a.strip()), float(g.strip().replace(",", ""))
            scores.append(3.5 if g == a else -1.5)
        except:
            scores.append(-2.5 if g is None else 0)
    return scores
