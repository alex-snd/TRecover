from typing import List, Union


def visualize_columns(columns: List[str], delimiter: str = '', as_rows=False) -> Union[str, List[str]]:
    """
    Get the columns string representation.

    Parameters
    ----------
    columns : List[str]
        Columns for keyless reading.
    delimiter : str
        Delimiter for columns visualization.
    as_rows : bool
        Return visualization as a list of strings.

    Returns
    -------
    Union[str, List[str]]:
        Visualization as a string or list of strings.

    """

    if len(columns) == 0:
        return ''

    rows = list()
    columns = [list(column) for column in columns]
    max_depth = max([len(column) for column in columns])

    for d in range(max_depth):

        row = str()
        for c in range(len(columns)):
            row += f'{delimiter}{columns[c][d]}' if d < len(columns[c]) else f'{delimiter} '

        rows.append(f'{row}{delimiter}')

    return rows if as_rows else '\n'.join(rows)


def visualize_target(target: List[str], delimiter: str = '') -> str:
    """
    Get the target string representation.

    Parameters
    ----------
    target : List[str]
        List of correct symbols for each column.
    delimiter : str
        Delimiter for target visualization.

    Returns
    -------
    str:
        Visualization as a string.

    """

    return f'{delimiter}{delimiter.join(target)}{delimiter}'
