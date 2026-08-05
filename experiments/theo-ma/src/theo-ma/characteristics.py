import argparse
import pandas as pd

def profile_dataset(df, name):
    print(f"\n===== {name} =====")

    print(f"Rows                : {len(df)}")
    print(f"Columns             : {len(df.columns)}")

    missing = df.isna().sum().sum()
    total = df.size

    print(f"Missing values      : {missing}")
    print(f"Missing ratio       : {missing/total:.3f}")

    duplicate_rows = df.duplicated().sum()
    print(f"Duplicate rows      : {duplicate_rows}")

    print(f"\nColumn statistics")

    for col in df.columns:

        dtype = str(df[col].dtype)

        unique = df[col].nunique(dropna=True)

        avg_length = None

        if dtype == "object":
            avg_length = (
                df[col]
                .dropna()
                .astype(str)
                .str.len()
                .mean()
            )

        print(f"{col:25s}")
        print(f"    type            : {dtype}")
        print(f"    unique values   : {unique}")

        if avg_length is not None:
            print(f"    avg str length  : {avg_length:.2f}")


def compare(source, target):

    print("\n===== Dataset comparison =====")

    common = set(source.columns) & set(target.columns)

    print(f"Common attributes   : {len(common)}")
    print(f"Shared names        : {sorted(common)}")

    print(f"Only source         : {sorted(set(source.columns)-common)}")
    print(f"Only target         : {sorted(set(target.columns)-common)}")


def gt_statistics(gt):

    print("\n===== Ground Truth =====")

    print(f"Pairs               : {len(gt)}")

    d1_unique = gt.iloc[:,0].nunique()
    d2_unique = gt.iloc[:,1].nunique()

    print(f"Unique source ids   : {d1_unique}")
    print(f"Unique target ids   : {d2_unique}")

    print(f"1:1 ratio           : {d1_unique == len(gt) and d2_unique == len(gt)}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--gt")

    args = parser.parse_args()

    source = pd.read_csv(args.source, sep=",")
    target = pd.read_csv(args.target, sep=",")

    profile_dataset(source, "SOURCE")
    profile_dataset(target, "TARGET")

    compare(source, target)

    if args.gt:
        gt = pd.read_csv(args.gt, sep="|")
        gt_statistics(gt)


if __name__ == "__main__":
    main()