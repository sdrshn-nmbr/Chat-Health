import pandas as pd
import argparse
import os
from datetime import datetime


def deduplicate_csv(input_file, output_dir=None):
    print(f"Reading file: {input_file}")

    df = pd.read_csv(input_file)
    initial_count = len(df)

    print(f"Initial number of rows: {initial_count}")

    df_no_duplicates = df.drop_duplicates(subset=["Context", "Response"])
    final_count = len(df_no_duplicates)

    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_deduplicated_{timestamp}.csv")
    report_file = os.path.join(output_dir, f"deduplication_report_{timestamp}.txt")

    df_no_duplicates.to_csv(output_file, index=False)

    total_dupes = initial_count - final_count

    report = f"""
    Deduplication Report
    Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Input File: {input_file}
    Output File: {output_file}
    Initial row count: {initial_count}
    Exact duplicates removed: {total_dupes}
    Final row count: {final_count}
    Reduction percentage: {(total_dupes/initial_count)*100:.2f}%
    """

    with open(report_file, "w") as f:
        f.write(report)

    print("\nDeduplication complete!")
    print(report)
    print(f"\nDeduplicated file saved as: {output_file}")
    print(f"Report saved as: {report_file}")

    return output_file, report_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove exact duplicates from a CSV file."
    )
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument(
        "--output_dir", help="Directory to save output files", default=None
    )

    args = parser.parse_args()
    deduplicate_csv(args.input_file, args.output_dir)

# use like this:
# python dedupe.py input_file.csv --output_dir /path/to/output/directory

# eg:
# python deduplicate_csv.py mental_health_conversations.csv --output_dir ./deduplicated_files

# if no output directory is specified, the deduplicated file and report will be saved in the current directory
