import sys

from  numbers_holder import *
from series_analyzer import *


def print_result(series_info: SeriesInfo):
    np.set_printoptions(linewidth=sys.maxsize)
    print(f"\nResult:")
    print(f"\nSource series: {series_info.source_series}")
    print(f"\nVariational series: {series_info.variational}")
    print(f"\nExtremal values: min {series_info.extremal_values[0]} max {series_info.extremal_values[1]}")
    print(f"\nRange: {series_info.range}")
    print(f"\nSample mean: {series_info.sample_mean}")
    print(f"\nSample variance: {series_info.sample_variance}")
    print(f"\nSample deviation: {series_info.sample_deviation}")
    series_info.empirical_distribution_function.fig.show()
    series_info.distributed_frequencies_histogram.fig.show()
    series_info.distributed_frequencies_polygon.fig.show()


def run():
    try:
        numbers = get_numbers()
        series_info = analyze_series(numbers)
        print_result(series_info)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    run()
