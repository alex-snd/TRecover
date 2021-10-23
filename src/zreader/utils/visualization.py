from typing import List, Union

from torch import Tensor

from config import vars


def visualize_columns(grid: Union[Tensor, List[str]], delimiter: str = '', as_rows=False) -> Union[str, List[str]]:
    if len(grid) == 0:
        return ''

    columns = list()
    max_depth = 0
    rows = list()

    if isinstance(grid, Tensor):
        for c in range(grid.shape[0]):
            columns.append([vars.NUM2ALPHABET[pos] for pos in range(grid.shape[1]) if grid[c, pos]])
            max_depth = len(columns[c]) if len(columns[c]) > max_depth else max_depth
    else:
        columns = [list(column) for column in grid]
        max_depth = max([len(column) for column in grid])

    for d in range(max_depth):

        row = str()
        for c in range(len(columns)):
            row += f'{delimiter}{columns[c][d]}' if d < len(columns[c]) else f'{delimiter} '

        rows.append(f'{row}{delimiter}')

    return rows if as_rows else '\n'.join(rows)


def visualize_target(tgt: Tensor, delimiter: str = '') -> str:
    tgt = [vars.NUM2ALPHABET[ch_id] for ch_id in tgt.tolist()]

    return f'{delimiter}{delimiter.join(tgt)}{delimiter}'
