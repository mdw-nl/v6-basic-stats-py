import pandas as pd

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data


@data(1)
def partial_basic_stats(df: pd.DataFrame, column_name: str):
    """Compute the basic stats partial

    The data argument contains a pandas-dataframe containing the local
    data from the node.
    """
    # extract the column_name from the dataframe.
    info(f"Extracting column {column_name}")
    values = df[column_name]

    # count number of rows
    # TODO: this is just the uber-basic example, the skeleton.
    info("Computing partials")
    local_count = len(values)

    # return the values as a dict
    return {
        "count": local_count
    }
