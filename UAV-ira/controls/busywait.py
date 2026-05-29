# This module provides a utility function for simple busy-wait delays that can be used to create precise timing delays without relying on sleep functions.
# which may not be accurate for very short durations, due to OS context switches.
import time

def delay_busywait(duration_seconds=0.001):
    current_time = time.perf_counter
    end_time = current_time() + duration_seconds
    while current_time() < end_time:
        pass  # Busy-wait loop