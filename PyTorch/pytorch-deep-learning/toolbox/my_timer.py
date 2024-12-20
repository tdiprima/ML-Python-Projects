"""
Calculates and prints elapsed training time on a specified device (default is CPU).
"""
from timeit import default_timer as timer

import torch


def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):
    # Prints difference between start and end time.
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


def testing():
    start_time = timer()
    # some code...
    end_time = timer()
    print_train_time(start=start_time, end=end_time, device="cpu")


# testing()
