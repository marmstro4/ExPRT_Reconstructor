#Blame: Michael Armstrong
#Contact: marmstro@lbl.gov

#Internal dependencies
import utilities
import proton_reco
import analysis

#External dependencies
import math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from itertools import combinations
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
import csv
import sys
import multiprocessing as mp
import time


def reconstruct_events(
    event_indices,
    numbers,
    target_thickness,
    beam_spot,
    progress_queue,
    simresx,
    simresy,
    simresz
):

    positions_local = []
    proton_count = 0

    total = len(event_indices)

    for i, line in enumerate(event_indices):

        fitcent = proton_reco.vertex_reco(
            line,
            numbers,
            target_thickness,
            beam_spot,
            simresx,
            simresy,
            simresz
        )

        if fitcent is not None:
            positions_local.append([
                numbers[line][0],
                fitcent[0],
                fitcent[1],
                fitcent[2]
            ])

            proton_count += 1

        # Send progress every 100 events
        if (i + 1) % 100 == 0:
            progress_queue.put(100)

    # Send remaining events if the chunk isn't divisible by 100
    remainder = total % 100

    if remainder != 0:
        progress_queue.put(remainder)

    return positions_local, proton_count


def format_time(seconds):

    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes = seconds / 60

    if minutes < 60:
        return f"{minutes:.1f}m"

    hours = int(minutes // 60)
    mins = int(minutes % 60)

    return f"{hours}h {mins}m"


def run_application():

    evt_num = utilities.line_count(ExPRT_hits)

    numbers = utilities.extract_numbers(ExPRT_hits)

    indices = list(range(min(evt_num, len(numbers))))

    nthreads = 20

    chunks = np.array_split(indices, nthreads)

    manager = mp.Manager()
    progress_queue = manager.Queue()

    args = [
        (
            chunk,
            numbers,
            target_thickness,
            beam_spot,
            progress_queue,
            simresx,
            simresy,
            simresz
        )
        for chunk in chunks
    ]

    positions = []
    proton_count = 0

    total_events = len(indices)

    # Start timer
    start_time = time.time()

    with mp.Pool(processes=nthreads) as pool:

        result_async = pool.starmap_async(
            reconstruct_events,
            args
        )

        completed = 0

        while not result_async.ready():

            while not progress_queue.empty():
                completed += progress_queue.get()

            # Don't allow completed to exceed total
            completed = min(completed, total_events)

            # Calculate elapsed time
            elapsed = time.time() - start_time

            # Calculate percentage
            percent = (
                completed / total_events * 100
                if total_events > 0
                else 100
            )

            # Calculate processing rate
            if elapsed > 0 and completed > 0:
                rate = completed / elapsed

                # Estimate remaining time
                remaining_events = total_events - completed
                eta = remaining_events / rate

                eta_string = format_time(eta)
                elapsed_string = format_time(elapsed)

                print(
                    f"\rProgress: {percent:6.2f}% | "
                    f"{completed:,}/{total_events:,} events | "
                    f"Rate: {rate:,.1f} evt/s | "
                    f"Elapsed: {elapsed_string} | "
                    f"Remaining: {eta_string}",
                    end="",
                    flush=True
                )

            else:

                print(
                    f"\rProgress: {percent:6.2f}% | "
                    f"{completed:,}/{total_events:,} events | "
                    f"Calculating ETA...",
                    end="",
                    flush=True
                )

            time.sleep(0.5)

        # Get results
        results = result_async.get()

    # Final timing
    elapsed = time.time() - start_time

    print("\n")
    print("Finished")
    print("Total time:", format_time(elapsed))

    if elapsed > 0:
        print(
            "Average rate:",
            f"{total_events / elapsed:,.1f}",
            "events/s"
        )

    for local_positions, count in results:

        filtered_positions = []

        for pos in local_positions:

            if np.any(np.abs(pos[3]) > target_thickness / 2):
                continue

            filtered_positions.append(pos)

        positions.extend(filtered_positions)

        proton_count += len(filtered_positions)

    print(
        "Reconstructed:",
        proton_count,
        "/",
        evt_num
    )

    return positions


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 main.py <file_prefix>")
        sys.exit(1)

    file_prefix = sys.argv[1]

    ExPRT_hits = f"{file_prefix}_hits.csv"

    print("Using hit file:", ExPRT_hits)

    print("verticies are obtained in 'vertex' object:")
    print("formatting: evt,x,y,z")

    target_thickness = 50
    beam_spot = 1

    # Add Gaussian smear to position resolutions (sigma)
    simresx = 0.0
    simresy = 0.0
    simresz = 0.0

    vertex = run_application()
