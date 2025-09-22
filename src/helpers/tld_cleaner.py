import argparse
import pandas as pd

def write_url_domain_and_category_csv(input_csv_path: str, output_csv_path: str) -> None:
    """
    Read the suspicious TLDs CSV, select 'url_domain' and 'metadata_category', and write to a new CSV.
    """
    df_raw = pd.read_csv(input_csv_path)
    required_columns = ["metadata_tld", "metadata_severity"]
    missing_columns = [c for c in required_columns if c not in df_raw.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in input CSV: {missing_columns}")
    
    df_filtered = df_raw[required_columns].copy()
    df_filtered.to_csv(output_csv_path, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Suspicious TLDs CSV to url_domain and metadata_category")
    parser.add_argument("--input", dest="input_csv", required=False, help="Path to input CSV")
    parser.add_argument("--output", dest="output_csv", required=False,help="Path to output CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    write_url_domain_and_category_csv(args.input_csv, args.output_csv)