import pandas as pd
import argparse
import os
from datetime import datetime


def deduplicate_csv(input_file, output_dir=None):
    """
    Remove only exact duplicates from a CSV file containing mental health conversations.
    Preserves entries with same context but different responses.
    Saves both the deduplicated file and a report of the deduplication process.
    """
    print(f"Reading file: {input_file}")

    # Read the CSV file
    df = pd.read_csv(input_file)
    initial_count = len(df)

    print(f"Initial number of rows: {initial_count}")

    # Remove only exact duplicates (identical Context AND Response)
    df_no_duplicates = df.drop_duplicates(subset=["Context", "Response"])
    final_count = len(df_no_duplicates)

    # Handle output directory
    if output_dir is None:
        output_dir = "."  # Use current directory if none specified

    os.makedirs(output_dir, exist_ok=True)

    # Generate output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_deduplicated_{timestamp}.csv")
    report_file = os.path.join(output_dir, f"deduplication_report_{timestamp}.txt")

    # Save deduplicated dataset
    df_no_duplicates.to_csv(output_file, index=False)

    # Generate report
    total_dupes = initial_count - final_count

    report = f"""Deduplication Report
    
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Input File: {input_file}
Output File: {output_file}

Initial row count: {initial_count}
Exact duplicates removed: {total_dupes}
Final row count: {final_count}

Reduction percentage: {(total_dupes/initial_count)*100:.2f}%

Note: Only exact duplicates (identical Context AND Response pairs) were removed.
Entries with the same context but different responses were preserved.
    """

    # Save report
    with open(report_file, "w") as f:
        f.write(report)

    print("\nDeduplication complete!")
    print(report)
    print(f"\nDeduplicated file saved as: {output_file}")
    print(f"Report saved as: {report_file}")

    return output_file, report_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove exact duplicates from a CSV file containing mental health conversations."
    )
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument(
        "--output_dir", help="Directory to save output files (optional)", default=None
    )

    args = parser.parse_args()
    deduplicate_csv(args.input_file, args.output_dir)
