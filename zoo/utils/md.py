import pandas as pd


def markdown_to_df(markdown_text: str) -> pd.DataFrame:
    lines = markdown_text.strip().split('\n')
    header = lines[0]
    data = lines[2:]
    header = header.strip('|').split('|')
    header = [col.strip() for col in header]
    data = [row.strip('|').split('|') for row in data]
    data = [[cell.strip() for cell in row] for row in data]

    df = pd.DataFrame(
        columns=header,
        data=data
    )
    return df
