def float_pe(v: float, gnum: int = 3) -> str:
    texts = [
        f'{v:.{gnum}g}',
        f'{v / 1e3:.{gnum}g}k',
        f'{v / 1e6:.{gnum}g}M',
        f'{v / 1e9:.{gnum}g}G',
    ]

    best_text = None
    for text in texts:
        if best_text is None or len(text) < len(best_text):
            best_text = text

    return best_text
