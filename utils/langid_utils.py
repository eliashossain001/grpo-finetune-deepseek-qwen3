import langid

def get_lang(text: str) -> str:
    if not text: return "und"
    lang, _ = langid.classify(text)
    return lang
