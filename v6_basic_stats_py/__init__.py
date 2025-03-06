import pandas as pd
import numpy as np

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
    
    If group_by is provided, raw group results are not returned. Instead, for each column
    the group-level computed statistics are aggregated using second-order stats (e.g., overall mean, std,
    median, and quantiles) to prevent disclosure of patient-level information.
    
    Returns a dictionary with computed statistics per column.
    """
    # Use task-supplied options or fall back to defaults.
    numeric_stats = kwargs.get("numeric_stats", 
                                 ["count", "mean", "std", "median", "min", "max", "iqr", "kurtosis", "skewness"])
    categorical_stats = kwargs.get("categorical_stats", 
                                   ["count", "unique", "value_counts"])
    group_by = kwargs.get("group_by", None)

    info(f"Selected numeric stats: {numeric_stats}.")
    info(f"Selected categorical stats: {categorical_stats}.")
    info(f"Grouping by {group_by}" if group_by else "No grouping will be performed!")
    
    # Mapping for numeric statistics functions.
    # Note: 'min' and 'max' are commented out as they could reveal sensitive patient-level info.
    numeric_funcs = {
        "count": lambda x: int(x.count()),
        "mean": lambda x: float(x.mean()),
        "std": lambda x: float(x.std()),
        "median": lambda x: float(x.median()),
        # "min": lambda x: x.min(),   # sensitive, replaced by aggregated quantiles below
        # "max": lambda x: x.max(),   # sensitive, replaced by aggregated quantiles below
        "iqr": lambda x: float(x.quantile(0.75) - x.quantile(0.25)),
        "kurtosis": lambda x: float(x.kurtosis()),
        "skewness": lambda x: float(x.skew())
    }
    
    # Mapping for categorical statistics functions.
    # Note: 'value_counts' is omitted when grouping, as we do not want to expose keys.
    categorical_funcs = {
        "count": lambda x: int(x.count()),
        "unique": lambda x: int(x.nunique()),
        # "value_counts": lambda x: x.value_counts().to_dict()  # not safe when grouping
    }
    
    # Helper function to compute stats for a series given a mapping and a list of desired stats.
    def compute_stats(series, funcs, stats_list):
        return {stat: funcs[stat](series) for stat in stats_list if stat in funcs}
    
    # Helper function to aggregate a list of numeric values into second-order statistics.
    def second_order_stats(values):
        arr = np.array(values)
        if arr.size == 0:
            return {}
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else None,
            "median": float(np.median(arr)),
            "25%": float(np.percentile(arr, 25)),
            "75%": float(np.percentile(arr, 75))
        }
    
    results = {}
    
    if group_by:
        # When grouping, do not return raw results per group.
        # Instead, compute the desired stats per group and aggregate these values.
        groups = df.groupby(group_by)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Filter out potentially dangerous operations.
                safe_numeric_stats = [stat for stat in numeric_stats if stat in numeric_funcs]
                # Dictionary to collect each stat's group-level values.
                agg_dict = {stat: [] for stat in safe_numeric_stats}
                for _, group in groups:
                    computed = compute_stats(group[col], numeric_funcs, safe_numeric_stats)
                    for stat in safe_numeric_stats:
                        if stat in computed and computed[stat] is not None:
                            agg_dict[stat].append(computed[stat])
                # Aggregate second-level stats over groups.
                results[col] = { stat: second_order_stats(agg_dict[stat]) for stat in safe_numeric_stats }
            else:
                safe_categorical_stats = [stat for stat in categorical_stats if stat in categorical_funcs]
                agg_dict = {stat: [] for stat in safe_categorical_stats}
                for _, group in groups:
                    computed = compute_stats(group[col], categorical_funcs, safe_categorical_stats)
                    for stat in safe_categorical_stats:
                        if stat in computed and computed[stat] is not None:
                            agg_dict[stat].append(computed[stat])
                results[col] = { stat: second_order_stats(agg_dict[stat]) for stat in safe_categorical_stats }
    else:
        # Without grouping, compute stats directly for each column.
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                results[col] = compute_stats(df[col], numeric_funcs, numeric_stats)
            else:
                results[col] = compute_stats(df[col], categorical_funcs, categorical_stats)
    info(results)
    return results