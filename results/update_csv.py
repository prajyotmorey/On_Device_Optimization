import csv
import os

def update_results_csv(results):
    results_file = "results/results.csv"

    file_exists = os.path.exists(results_file)

    with open(
        results_file,
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(results)
results_file = "results/results.csv"

file_exists = os.path.exists(results_file)

with open(
    results_file,
    "a",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results.keys()
    )

    if not file_exists:
        writer.writeheader()

    writer.writerow(results)