import pandas as pd

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data


@data(1)
def partial_basic_stats(df: pd.DataFrame):
    """Compute the basic stats partial

    The data argument contains a pandas-dataframe containing the local
    data from the node.
    """
    # count rows
    num_rows = df.shape[0]

    # verify it's an integer
    assert isinstance(num_rows, int)

    # return the values as a dict
    return {
        "count": num_rows
    }
