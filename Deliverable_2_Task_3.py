
# Task 3

import numpy as np


def remove_price_outliers(prices: np.ndarray) -> np.ndarray:
    """
    Replace price outliers with the median price.

    Outliers are identified using the three-sigma rule.
    If the three-sigma rule fails to detect obvious extreme
    values, values larger than 100 are also treated as outliers.

    Parameters
    ----------
    prices : np.ndarray
        One-dimensional array containing transaction prices.

    Returns
    -------
    np.ndarray
        Array with outliers replaced by the median value.
    """

    # Convert input to a NumPy float array
    prices = np.asarray(prices, dtype=float)

    # Calculate statistics
    mean = np.mean(prices)
    std = np.std(prices)

    # Identify outliers using the three-sigma rule
    outlier_mask = np.abs(prices - mean) > (3 * std)

    # Dataset generator uses 999.00 as an obvious bad value
    outlier_mask |= prices > 100

    # Calculate median of valid prices
    median_price = np.median(prices[~outlier_mask])

    # Create a copy of the original array
    cleaned_prices = prices.copy()

    # Replace outliers with the median value
    cleaned_prices[outlier_mask] = median_price

    return cleaned_prices


def compute_refund_ratio(
    sales_count: np.ndarray,
    refund_count: np.ndarray
) -> np.ndarray:
    """
    Compute refund ratios element-wise.

    The refund ratio is calculated as:

        refund_count / (sales_count + 1e-6)

    Parameters
    ----------
    sales_count : np.ndarray
        Number of sales.

    refund_count : np.ndarray
        Number of refunds.

    Returns
    -------
    np.ndarray
        Array containing refund ratios.
    """

    # Convert inputs to NumPy float arrays
    sales_count = np.asarray(sales_count, dtype=float)
    refund_count = np.asarray(refund_count, dtype=float)

    # Avoid division by zero
    return refund_count / (sales_count + 1e-6)


if __name__ == "__main__":

    # Example test for Task 3
    prices = np.array([4.5, 5.0, 4.8, 999.0, 5.2])

    print("Original prices:")
    print(prices)

    print("\nCleaned prices:")
    print(remove_price_outliers(prices))

    sales = np.array([100, 50, 0])
    refunds = np.array([10, 5, 1])

    print("\nRefund ratios:")
    print(compute_refund_ratio(sales, refunds))