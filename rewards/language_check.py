from utils.langid_utils import get_lang

def format_and_language_reward_func(completions, **kwargs):
    scores = []
    for c in completions:
        content = c[0]["content"]
        lang = get_lang(content)
        scores.append(5.0 if lang == 'id' else -3.0 if lang in ['en', 'zh'] else -5.0)
    return scores
