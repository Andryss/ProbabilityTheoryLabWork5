import math

import matplotlib.pyplot as plt
import numpy as np


class Function:
    fig: plt.Figure
    ax: plt.Axes

    def __init__(self, f, a):
        self.fig = f
        self.ax = a


class SeriesInfo:
    source_series: np.ndarray
    length: int
    value_counts: dict
    variational: np.ndarray                         # вариационный ряд
    extremal_values: (float, float)                 # экстремальные значения
    range: float                                    # размах
    sample_mean: float                              # выборочное математическое ожидание
    sample_variance: float                          # выборочная дисперсия
    sample_deviation: float                         # среднеквадратическое отклонение
    empirical_distribution_function: Function       # эмпирическая функция распределения
    distributed_frequencies_histogram: Function     # гистограмма распределенных частот
    distributed_frequencies_polygon: Function       # полигон распределенных частот


def calculate_meta_info(info: SeriesInfo):
    info.length = len(info.source_series)
    info.value_counts = dict(zip(*np.unique(np.sort(info.source_series), return_counts=True)))


def calculate_variational(info: SeriesInfo):
    info.variational = np.sort(info.source_series)


def calculate_extremal_values(info: SeriesInfo):
    info.extremal_values = (info.variational[0], info.variational[-1])


def calculate_range(info: SeriesInfo):
    info.range = info.extremal_values[1] - info.extremal_values[0]


def calculate_sample_mean(info: SeriesInfo):
    info.sample_mean = info.variational.sum() / info.length


def calculate_sample_variance(info: SeriesInfo):
    squared = info.variational * info.variational
    squared_mean = squared.sum() / len(squared)
    info.sample_variance = squared_mean - info.sample_mean ** 2


def calculate_sample_deviation(info: SeriesInfo):
    info.sample_deviation = math.sqrt(info.sample_variance)


def calculate_empirical_distribution_function(info: SeriesInfo):

    def count_proba(x):
        proba = 0
        for count in info.value_counts.keys():
            if count < x:
                proba += info.value_counts[count]
        return proba / info.length

    x_vals = np.linspace(info.extremal_values[0] - info.range * 0.1, info.extremal_values[1] + info.range * 0.1, 10_000)
    y_func = np.vectorize(count_proba)
    y_vals = y_func(x_vals)

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals)
    ax.set_title("Empirical distribution function")
    info.empirical_distribution_function = Function(fig, ax)


def calculate_distributed_frequencies_histogram(info: SeriesInfo):
    intervals_count = math.floor(1 + 3.322 * math.log(info.length, 10))

    fig, ax = plt.subplots()
    ax.hist(info.variational, bins=intervals_count)
    ax.set_title("Distributed frequencies histogram")
    info.distributed_frequencies_histogram = Function(fig, ax)


def calculate_distributed_frequencies_polygon(info: SeriesInfo):
    x_vals = info.value_counts.keys()
    y_vals = info.value_counts.values()

    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals)
    ax.set_title("Distributed frequencies polygon")
    info.distributed_frequencies_polygon = Function(fig, ax)


def analyze_series(series: np.ndarray) -> SeriesInfo:
    info = SeriesInfo()
    info.source_series = series.reshape(-1)
    calculate_meta_info(info)
    
    calculate_variational(info)
    calculate_extremal_values(info)
    calculate_range(info)
    calculate_sample_mean(info)
    calculate_sample_variance(info)
    calculate_sample_deviation(info)
    calculate_empirical_distribution_function(info)
    calculate_distributed_frequencies_histogram(info)
    calculate_distributed_frequencies_polygon(info)
    
    return info
