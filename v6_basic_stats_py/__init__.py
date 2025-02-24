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

@data(1)
def flexible_stats(df: pd.DataFrame, **kwargs):
    """
    Compute flexible statistics on a dataframe.
    
    Accepts additional keyword arguments to specify which statistics to compute:
    
    - numeric_stats: List of statistic names for numeric columns.
                     Default: ["count", "mean", "std", "median", "min", "max", "iqr", "kurtosis", "skewness"]
    - categorical_stats: List of statistic names for non-numeric (categorical) columns.
                         Default: ["count", "unique", "value_counts"]
    - group_by: Optional column name to group by before computing stats.
    
    Returns a dictionary with computed statistics per column (or per group if group_by is provided).
    """
    # Use task-supplied options or fall back to defaults.
    numeric_stats = kwargs.get("numeric_stats", 
                                 ["count", "mean", "std", "median", "min", "max", "iqr", "kurtosis", "skewness"])
    categorical_stats = kwargs.get("categorical_stats", 
                                   ["count", "unique", "value_counts"])
    group_by = kwargs.get("group_by", None)

    info(f"Selected numeric stats: {numeric_stats}.")
    info(f"Selected categorical stats: {categorical_stats}.")
    info(f"Goupping by {group_by}" if group_by else "No groupping will be performed!")
    
    # Mapping for numeric statistics functions.
    numeric_funcs = {
        "count": lambda x: x.count(),
        "mean": lambda x: x.mean(),
        "std": lambda x: x.std(),
        "median": lambda x: x.median(),
        "min": lambda x: x.min(),
        "max": lambda x: x.max(),
        "iqr": lambda x: x.quantile(0.75) - x.quantile(0.25),
        "kurtosis": lambda x: x.kurtosis(),
        "skewness": lambda x: x.skew()
    }
    
    # Mapping for categorical statistics functions.
    categorical_funcs = {
        "count": lambda x: x.count(),
        "unique": lambda x: x.nunique(),
        "value_counts": lambda x: x.value_counts().to_dict()
    }
    
    # Helper function to compute stats for a series given a mapping and a list of desired stats.
    def compute_stats(series, funcs, stats_list):
        return {stat: funcs[stat](series) for stat in stats_list if stat in funcs}
    
    results = {}
    
    # Optionally compute stats per group if group_by column is provided.
    if group_by:
        groups = df.groupby(group_by)
        for name, group in groups:
            group_result = {}
            for col in group.columns:
                if pd.api.types.is_numeric_dtype(group[col]):
                    group_result[col] = compute_stats(group[col], numeric_funcs, numeric_stats)
                else:
                    group_result[col] = compute_stats(group[col], categorical_funcs, categorical_stats)
            results[name] = group_result
    else:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                results[col] = compute_stats(df[col], numeric_funcs, numeric_stats)
            else:
                results[col] = compute_stats(df[col], categorical_funcs, categorical_stats)
    
    return results
