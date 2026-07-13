"""Data preprocessing pipeline for the H&M recommendation dataset.

This script loads raw CSV files, standardizes fields, removes invalid rows,
processes customer demographics, prevents data leakage via strict mapping,
and writes model-ready parquet outputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_INTERACTION_COLUMNS = [
    "customer_id",
    "article_id",
    "t_dat",
    "price",
    "sales_channel_id",
]

REQUIRED_ITEM_COLUMNS = [
    "article_id",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "department_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
]

REQUIRED_CUSTOMER_COLUMNS = [
    "customer_id",
    "FN",
    "Active",
    "club_member_status",
    "fashion_news_frequency",
    "age",
]

CATEGORY_COLUMNS = [
    "prod_name",
    "product_type_name",
    "product_group_name",
    "department_name",
    "section_name",
    "garment_group_name",
    "detail_desc",
]


@dataclass
class PipelineStats:
    """Track pipeline row counts for observability."""

    interactions_raw_rows: int = 0
    interactions_train_rows: int = 0
    interactions_val_rows: int = 0
    interactions_test_rows: int = 0
    interactions_clean_rows: int = 0
    interactions_dropped_missing: int = 0
    interactions_dropped_invalid_price: int = 0
    interactions_deduplicated: int = 0
    interactions_dropped_orphan_items: int = 0
    items_raw_rows: int = 0
    items_clean_rows: int = 0
    items_with_images: int = 0
    customers_raw_rows: int = 0
    customers_clean_rows: int = 0


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        msg = f"Missing columns in {name}: {', '.join(missing)}"
        raise ValueError(msg)


def _normalize_text(series: pd.Series) -> pd.Series:
    out = series.fillna("").astype(str).str.strip().str.lower()
    # Collapse repeated spaces to keep category text consistent.
    return out.str.replace(r"\s+", " ", regex=True)


def _resolve_image_path(images_dir: Path, article_id: str) -> str | None:
    article_clean = article_id.strip()
    candidates = [article_clean, article_clean.zfill(10)]
    for candidate in candidates:
        folder = candidate[:3]
        jpg = images_dir / folder / f"{candidate}.jpg"
        png = images_dir / folder / f"{candidate}.png"
        if jpg.exists():
            return str(jpg)
        if png.exists():
            return str(png)
    return None


def _build_item_text(frame: pd.DataFrame) -> pd.Series:
    parts = [
        frame[column].fillna("").astype(str)
        for column in CATEGORY_COLUMNS
    ]
    merged = parts[0]
    for section in parts[1:]:
        merged = merged + " " + section
    return merged.str.replace(r"\s+", " ", regex=True).str.strip()


def load_interactions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, REQUIRED_INTERACTION_COLUMNS, "transactions")
    return frame[REQUIRED_INTERACTION_COLUMNS].copy()


def load_items(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, REQUIRED_ITEM_COLUMNS, "articles")
    return frame[REQUIRED_ITEM_COLUMNS].copy()


def load_customers(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, REQUIRED_CUSTOMER_COLUMNS, "customers")
    return frame[REQUIRED_CUSTOMER_COLUMNS].copy()


def clean_interactions(
    frame: pd.DataFrame,
    valid_article_ids: set[str],
    stats: PipelineStats,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = frame.copy()

    out["customer_id"] = _normalize_text(out["customer_id"])
    out["article_id"] = out["article_id"].astype("string").str.strip()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["sales_channel_id"] = pd.to_numeric(
        out["sales_channel_id"], errors="coerce"
    ).astype("Int64")

    parsed_dates = pd.to_datetime(out["t_dat"], errors="coerce")
    out["event_date"] = parsed_dates.dt.date.astype("string")
    out["event_ts"] = parsed_dates

    max_event_ts = out["event_ts"].max()
    if pd.isna(max_event_ts):
        empty = out.iloc[0:0].copy()
        stats.interactions_dropped_missing = len(out)
        return empty, empty, empty

    # Calculate month difference using standard math to avoid offset casting errors
    max_year = max_event_ts.year
    max_month = max_event_ts.month

    # month_diff is 0 for the latest month, 1 for the previous, etc.
    month_diff = (max_year - out["event_ts"].dt.year) * 12 + (max_month - out["event_ts"].dt.month)
    
    # Map the latest month to 6, previous to 5, down to 1
    month_number = 6 - month_diff

    day_of_month = out["event_ts"].dt.day
    week_of_month = ((day_of_month - 1) // 7) + 1

    in_six_month_window = month_number.between(1, 6)
    month6_mask = month_number == 6

    train_mask = in_six_month_window & (
        (month_number <= 5) | (month6_mask & (week_of_month <= 2))
    )
    val_mask = month6_mask & (week_of_month == 3)
    test_mask = month6_mask & (week_of_month == 4)

    train_df = out[train_mask].copy()
    val_df = out[val_mask].copy()
    test_df = out[test_mask].copy()

    def _clean_split(split_df: pd.DataFrame) -> pd.DataFrame:
        before_missing = len(split_df)
        split_df = split_df.dropna(
            subset=["customer_id", "article_id", "event_ts", "price"]
        )
        split_df = split_df[split_df["customer_id"] != ""]
        split_df = split_df[split_df["article_id"] != ""]
        stats.interactions_dropped_missing += before_missing - len(split_df)

        before_price = len(split_df)
        split_df = split_df[split_df["price"] > 0]
        stats.interactions_dropped_invalid_price += (
            before_price - len(split_df)
        )

        before_orphans = len(split_df)
        split_df = split_df[split_df["article_id"].isin(valid_article_ids)]
        stats.interactions_dropped_orphan_items += (
            before_orphans - len(split_df)
        )

        # Keep duplicate rows because repeated purchases are meaningful events.
        split_df = split_df.sort_values(
            ["customer_id", "event_ts", "article_id"]
        ).reset_index(drop=True)
        return split_df

    train_df = _clean_split(train_df)
    val_df = _clean_split(val_df)
    test_df = _clean_split(test_df)

    stats.interactions_deduplicated = 0
    return train_df, val_df, test_df


def clean_items(
    frame: pd.DataFrame,
    stats: PipelineStats,
    images_dir: Path | None = None,
) -> pd.DataFrame:
    out = frame.copy()
    out["article_id"] = out["article_id"].astype("string").str.strip()
    out = out[out["article_id"] != ""]

    for column in CATEGORY_COLUMNS:
        out[column] = _normalize_text(out[column])

    if images_dir is not None and images_dir.exists():
        out["image_path"] = out["article_id"].apply(
            lambda article_id: _resolve_image_path(images_dir, article_id)
        )
    else:
        out["image_path"] = None

    out["has_image"] = out["image_path"].notna()
    out["item_text"] = _build_item_text(out)

    out = out.reset_index(drop=True)
    stats.items_with_images = int(out["has_image"].sum())
    stats.items_clean_rows = len(out)
    return out


def clean_customers(frame: pd.DataFrame, stats: PipelineStats) -> pd.DataFrame:
    out = frame.copy()
    out["customer_id"] = _normalize_text(out["customer_id"])
    
    # Fill missing ages with median for demographic cold-start
    out["age"] = out["age"].fillna(out["age"].median())
    
    # Clean membership fields
    out["club_member_status"] = _normalize_text(out["club_member_status"])
    out["club_member_status"] = out["club_member_status"].replace("", "none")
    
    out = out.reset_index(drop=True)
    stats.customers_clean_rows = len(out)
    return out


def build_id_maps(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = pd.DataFrame(
        {"customer_id": sorted(interactions["customer_id"].unique())}
    )
    users["user_idx"] = range(len(users))

    items = pd.DataFrame(
        {"article_id": sorted(interactions["article_id"].unique())}
    )
    items["item_idx"] = range(len(items))
    return users, items


def apply_id_maps(
    interactions: pd.DataFrame,
    user_map: pd.DataFrame,
    item_map: pd.DataFrame,
) -> pd.DataFrame:
    out = interactions.merge(user_map, on="customer_id", how="left")
    out = out.merge(item_map, on="article_id", how="left")
    # Cast safely; new users/items in val/test sets will become NaN
    out["user_idx"] = out["user_idx"].astype("Int64")
    out["item_idx"] = out["item_idx"].astype("Int64")
    return out


def write_outputs(
    processed_dir: Path,
    interactions_train: pd.DataFrame,
    interactions_val: pd.DataFrame,
    interactions_test: pd.DataFrame,
    items: pd.DataFrame,
    customers: pd.DataFrame,
    user_map: pd.DataFrame,
    item_map: pd.DataFrame,
    stats: PipelineStats,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)

    interactions_train_out = processed_dir / "interactions_train.parquet"
    interactions_val_out = processed_dir / "interactions_val.parquet"
    interactions_test_out = processed_dir / "interactions_test.parquet"
    interactions_clean_out = processed_dir / "interactions_clean.parquet"
    items_out = processed_dir / "items_clean.parquet"
    customers_out = processed_dir / "customers_clean.parquet"
    user_map_out = processed_dir / "customer_map.parquet"
    item_map_out = processed_dir / "item_map.parquet"
    summary_out = processed_dir / "preprocess_summary.json"

    interactions_train.to_parquet(interactions_train_out, index=False)
    interactions_val.to_parquet(interactions_val_out, index=False)
    interactions_test.to_parquet(interactions_test_out, index=False)
    
    # Save a combined version for easy global EDA if needed
    pd.concat(
        [interactions_train, interactions_val, interactions_test],
        ignore_index=True,
    ).to_parquet(interactions_clean_out, index=False)
    
    items.to_parquet(items_out, index=False)
    customers.to_parquet(customers_out, index=False)
    user_map.to_parquet(user_map_out, index=False)
    item_map.to_parquet(item_map_out, index=False)

    payload = {
        "interactions_raw_rows": stats.interactions_raw_rows,
        "interactions_train_rows": len(interactions_train),
        "interactions_val_rows": len(interactions_val),
        "interactions_test_rows": len(interactions_test),
        "interactions_clean_rows": (
            len(interactions_train)
            + len(interactions_val)
            + len(interactions_test)
        ),
        "interactions_dropped_missing": stats.interactions_dropped_missing,
        "interactions_dropped_invalid_price": stats.interactions_dropped_invalid_price,
        "interactions_deduplicated": stats.interactions_deduplicated,
        "interactions_dropped_orphan_items": stats.interactions_dropped_orphan_items,
        "items_raw_rows": stats.items_raw_rows,
        "items_clean_rows": stats.items_clean_rows,
        "items_with_images": stats.items_with_images,
        "customers_raw_rows": stats.customers_raw_rows,
        "customers_clean_rows": stats.customers_clean_rows,
    }
    summary_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_pipeline(raw_dir: Path, processed_dir: Path, force: bool = False) -> None:
    expected_outputs = [
        "interactions_train.parquet",
        "interactions_val.parquet",
        "interactions_test.parquet",
        "interactions_clean.parquet",
        "items_clean.parquet",
        "customers_clean.parquet",
        "customer_map.parquet",
        "item_map.parquet",
        "preprocess_summary.json",
    ]

    # Check if all files exist and we are not forcing a run
    if not force and all((processed_dir / f).exists() for f in expected_outputs):
        print(f"Skipping preprocessing: All output files already exist in {processed_dir}")
        print("   (Run with --force if you need to overwrite them)")
        return

    transactions_path = raw_dir / "transactions_train.csv"
    articles_path = raw_dir / "articles.csv"
    customers_path = raw_dir / "customers.csv"

    for path in [transactions_path, articles_path, customers_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")

    stats = PipelineStats()

    print("Loading raw datasets...")
    interactions_raw = load_interactions(transactions_path)
    items_raw = load_items(articles_path)
    customers_raw = load_customers(customers_path)

    stats.interactions_raw_rows = len(interactions_raw)
    stats.items_raw_rows = len(items_raw)
    stats.customers_raw_rows = len(customers_raw)

    print("Cleaning metadata...")
    images_dir = raw_dir / "images"
    items_clean = clean_items(items_raw, stats, images_dir=images_dir)
    customers_clean = clean_customers(customers_raw, stats)
    valid_ids = set(items_clean["article_id"].unique())
    
    print("Splitting and cleaning interactions...")
    (
        interactions_train,
        interactions_val,
        interactions_test,
    ) = clean_interactions(interactions_raw, valid_ids, stats)

    print("Building ID maps...")
    user_map, item_map = build_id_maps(interactions_train)
    
    print("Applying ID maps...")
    interactions_train_mapped = apply_id_maps(interactions_train, user_map, item_map)
    interactions_val_mapped = apply_id_maps(interactions_val, user_map, item_map)
    interactions_test_mapped = apply_id_maps(interactions_test, user_map, item_map)

    items_mapped = items_clean.merge(item_map, on="article_id", how="left")
    customers_mapped = customers_clean.merge(user_map, on="customer_id", how="left")
    
    items_mapped["item_idx"] = items_mapped["item_idx"].astype("Int64")
    customers_mapped["user_idx"] = customers_mapped["user_idx"].astype("Int64")

    print("Saving parquet outputs...")
    write_outputs(
        processed_dir=processed_dir,
        interactions_train=interactions_train_mapped,
        interactions_val=interactions_val_mapped,
        interactions_test=interactions_test_mapped,
        items=items_mapped,
        customers=customers_mapped,
        user_map=user_map,
        item_map=item_map,
        stats=stats,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess H&M recommendation dataset"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing transactions_train.csv, articles.csv, and customers.csv",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where cleaned parquet files will be saved",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force preprocessing to run even if the output files already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        raw_dir=args.raw_dir, 
        processed_dir=args.processed_dir, 
        force=args.force
    )
    print(f"Preprocessing step finished.")


if __name__ == "__main__":
    main()