"""Calculate plant-level drought indices from three-class PNG masks.

Expected grayscale label values:
    0   = background
    128 = healthy-green plant tissue
    255 = visibly drought-stressed plant tissue

The script processes every PNG in INPUT_FOLDER and writes per-image results,
dataset summaries, validation information, and publication-quality plots.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
INPUT_FOLDER = Path(
    r"C:/WVSU_Documents/Research/Drought_Stress_Evan_Allen/"
    r"Data_segments/three_class_processed_091626/"
)

# Results will be saved inside the mask folder.
OUTPUT_FOLDER = INPUT_FOLDER / "dsi_hf_results"

# Show each figure on screen in addition to saving it as a 300-dpi PNG.
# Close the current plot window to allow the script to continue.
SHOW_PLOTS = True

# Use a consistent, publication-readable font size in every generated figure.
FIGURE_FONT_SIZE = 15

plt.rcParams.update(
    {
        "font.size": FIGURE_FONT_SIZE,
        "axes.labelsize": FIGURE_FONT_SIZE,
        "xtick.labelsize": FIGURE_FONT_SIZE,
        "ytick.labelsize": FIGURE_FONT_SIZE,
        "legend.fontsize": FIGURE_FONT_SIZE,
    }
)

BACKGROUND_VALUE = 0
HEALTHY_VALUE = 128
STRESSED_VALUE = 255

# PNG is lossless, so exact values are expected. Set to 1 or 2 only if your
# image-export workflow introduces small deviations around 0, 128, or 255.
VALUE_TOLERANCE = 0

# Optional metadata file with at least these columns:
#     filename,treatment
#
# Example:
#     plant01.png,Control
#     plant31.png,Drought
#
# Leave as None when no metadata file is available.
METADATA_CSV = None

# Example:
# METADATA_CSV = INPUT_FOLDER / "plant_metadata.csv"


LABEL_VALUES = np.array(
    [BACKGROUND_VALUE, HEALTHY_VALUE, STRESSED_VALUE],
    dtype=np.int16,
)


def load_grayscale_labels(path: Path) -> np.ndarray:
    """Load a grayscale or palette mask and return a 2D integer array."""
    with Image.open(path) as image:
        array = np.asarray(image)

    if array.ndim == 2:
        return array.astype(np.int16)

    if array.ndim == 3:
        rgb = array[:, :, :3]

        if (
            np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
            and np.array_equal(rgb[:, :, 0], rgb[:, :, 2])
        ):
            return rgb[:, :, 0].astype(np.int16)

    raise ValueError(
        "Mask is not grayscale. Export the ilastik labels as a grayscale PNG "
        "with pixel values 0, 128, and 255."
    )


def validate_or_remap_labels(
    mask: np.ndarray,
) -> tuple[np.ndarray, int, str]:
    """Validate labels and optionally remap nearby values within tolerance."""
    original = mask.copy()

    distances = np.abs(mask[..., None] - LABEL_VALUES)
    nearest_indices = np.argmin(distances, axis=-1)
    nearest_distances = np.min(distances, axis=-1)

    valid = nearest_distances <= VALUE_TOLERANCE

    unexpected_count = int(np.count_nonzero(~valid))
    unexpected_values = np.unique(original[~valid])

    unexpected_text = ";".join(
        str(int(value)) for value in unexpected_values[:20]
    )

    if len(unexpected_values) > 20:
        unexpected_text += ";..."

    remapped = LABEL_VALUES[nearest_indices]

    # Keep unexpected values unchanged so they are excluded from all classes.
    remapped[~valid] = original[~valid]

    return remapped, unexpected_count, unexpected_text


def safe_ratio(numerator: int, denominator: int) -> float:
    """Return a ratio or NaN when the denominator is zero."""
    return float(numerator / denominator) if denominator else math.nan


def analyze_mask(path: Path) -> dict:
    """Calculate pixel counts, healthy fraction, and DSI for one mask."""
    mask = load_grayscale_labels(path)

    mask, unexpected_count, unexpected_values = validate_or_remap_labels(
        mask
    )

    background = int(
        np.count_nonzero(mask == BACKGROUND_VALUE)
    )
    healthy = int(
        np.count_nonzero(mask == HEALTHY_VALUE)
    )
    stressed = int(
        np.count_nonzero(mask == STRESSED_VALUE)
    )

    canopy = healthy + stressed
    total = int(mask.size)

    healthy_fraction = safe_ratio(healthy, canopy)
    drought_severity_index = safe_ratio(stressed, canopy)

    return {
        "filename": path.name,
        "width_px": int(mask.shape[1]),
        "height_px": int(mask.shape[0]),
        "total_pixels": total,
        "background_pixels": background,
        "healthy_pixels": healthy,
        "stressed_pixels": stressed,
        "canopy_pixels": canopy,
        "canopy_cover_fraction": safe_ratio(canopy, total),
        "canopy_cover_percent": 100.0 * safe_ratio(canopy, total),
        "healthy_fraction_HF": healthy_fraction,
        "healthy_percent": 100.0 * healthy_fraction,
        "drought_severity_index_DSI": drought_severity_index,
        "stressed_percent": 100.0 * drought_severity_index,
        "stressed_to_healthy_ratio": safe_ratio(stressed, healthy),
        "unexpected_pixel_count": unexpected_count,
        "unexpected_pixel_values": unexpected_values,
        "valid_label_fraction": safe_ratio(
            total - unexpected_count,
            total,
        ),
    }


def make_dataset_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Create descriptive statistics for the plant-level variables."""
    variables = [
        "healthy_fraction_HF",
        "drought_severity_index_DSI",
        "healthy_percent",
        "stressed_percent",
        "canopy_cover_fraction",
        "canopy_cover_percent",
        "canopy_pixels",
    ]

    rows = []

    for variable in variables:
        values = pd.to_numeric(
            results[variable],
            errors="coerce",
        ).dropna()

        rows.append(
            {
                "variable": variable,
                "n": int(values.size),
                "mean": values.mean(),
                "standard_deviation": values.std(ddof=1),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "minimum": values.min(),
                "maximum": values.max(),
            }
        )

    return pd.DataFrame(rows)


def save_composition_plot(
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a stacked plot of healthy and visibly stressed canopy pixels."""
    plotting = (
        results.dropna(
            subset=["healthy_percent", "stressed_percent"]
        )
        .sort_values("drought_severity_index_DSI")
        .reset_index(drop=True)
    )

    if plotting.empty:
        return

    # P1 has the lowest DSI, and the final plant has the highest DSI.
    plotting["plot_label"] = [
        f"P{i}" for i in range(1, len(plotting) + 1)
    ]

    plotting[
        [
            "plot_label",
            "filename",
            "healthy_fraction_HF",
            "drought_severity_index_DSI",
            "healthy_percent",
            "stressed_percent",
        ]
    ].to_csv(
        output_path.parent / "plant_plot_label_mapping.csv",
        index=False,
    )

    x = np.arange(len(plotting))

    # Allow adequate space for the larger 15-point plant labels.
    width = max(12.0, 0.35 * len(plotting))

    fig, ax = plt.subplots(figsize=(width, 7.0))

    ax.bar(
        x,
        plotting["healthy_percent"],
        color="#2E8B57",
        label="Healthy-green",
    )

    ax.bar(
        x,
        plotting["stressed_percent"],
        bottom=plotting["healthy_percent"],
        color="#D95F02",
        label="Visibly stressed",
    )

    ax.set_ylabel(
        "Fraction of canopy pixels (%)",
        fontsize=FIGURE_FONT_SIZE,
    )

    ax.set_xlabel(
        "Plant images ranked by increasing DSI",
        fontsize=FIGURE_FONT_SIZE,
    )

    ax.set_ylim(0, 100)

    ax.set_xticks(x)

    ax.set_xticklabels(
        plotting["plot_label"],
        rotation=90,
        fontsize=FIGURE_FONT_SIZE,
    )

    ax.tick_params(
        axis="both",
        labelsize=FIGURE_FONT_SIZE,
    )

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0, 1.14),
        fontsize=FIGURE_FONT_SIZE,
    )

    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def save_distribution_plot(
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a histogram of plant-level RGB-derived DSI values."""
    dsi = results[
        "drought_severity_index_DSI"
    ].dropna()

    if dsi.empty:
        return

    bins = min(
        12,
        max(5, int(np.sqrt(len(dsi)))),
    )

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    ax.hist(
        dsi,
        bins=bins,
        color="#D95F02",
        edgecolor="white",
        alpha=0.85,
    )

    ax.axvline(
        dsi.mean(),
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"Mean = {dsi.mean():.3f}",
    )

    ax.axvline(
        dsi.median(),
        color="#2166AC",
        linestyle=":",
        linewidth=2.0,
        label=f"Median = {dsi.median():.3f}",
    )

    ax.set_xlabel(
        "RGB-derived drought severity index (DSI)",
        fontsize=FIGURE_FONT_SIZE,
    )

    ax.set_ylabel(
        "Number of plant images",
        fontsize=FIGURE_FONT_SIZE,
    )

    ax.set_xlim(0, 1)

    ax.tick_params(
        axis="both",
        labelsize=FIGURE_FONT_SIZE,
    )

    ax.legend(
        frameon=False,
        fontsize=FIGURE_FONT_SIZE,
    )

    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)


def add_metadata_and_group_outputs(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Optionally merge treatment labels and produce group results."""
    if METADATA_CSV is None:
        return results

    metadata_path = Path(METADATA_CSV)
    metadata = pd.read_csv(metadata_path)

    required = {"filename", "treatment"}
    missing = required.difference(metadata.columns)

    if missing:
        raise ValueError(
            "Metadata file is missing columns: "
            + ", ".join(sorted(missing))
        )

    merged = results.merge(
        metadata[["filename", "treatment"]],
        on="filename",
        how="left",
        validate="1:1",
    )

    if merged["treatment"].isna().any():
        missing_names = merged.loc[
            merged["treatment"].isna(),
            "filename",
        ].tolist()

        raise ValueError(
            f"No treatment label for: {missing_names}"
        )

    group_summary = (
        merged.groupby(
            "treatment",
            observed=True,
        )["drought_severity_index_DSI"]
        .agg(
            n="count",
            mean="mean",
            standard_deviation="std",
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    group_summary.to_csv(
        OUTPUT_FOLDER / "dsi_summary_by_treatment.csv",
        index=False,
    )

    groups = list(
        merged["treatment"].drop_duplicates()
    )

    data = [
        merged.loc[
            merged["treatment"] == group,
            "drought_severity_index_DSI",
        ]
        .dropna()
        .to_numpy()
        for group in groups
    ]

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    ax.boxplot(
        data,
        tick_labels=groups,
        patch_artist=True,
        boxprops={
            "facecolor": "#F4A261",
            "alpha": 0.55,
        },
        medianprops={
            "color": "black",
            "linewidth": 1.8,
        },
    )

    rng = np.random.default_rng(20260916)

    for position, values in enumerate(
        data,
        start=1,
    ):
        jitter = rng.normal(
            position,
            0.045,
            size=len(values),
        )

        ax.scatter(
            jitter,
            values,
            s=40,
            alpha=0.75,
            color="#264653",
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_ylabel(
        "RGB-derived drought severity index (DSI)",
        fontsize=FIGURE_FONT_SIZE,
    )

    ax.set_ylim(0, 1)

    ax.tick_params(
        axis="both",
        labelsize=FIGURE_FONT_SIZE,
    )

    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FOLDER / "dsi_by_treatment.png",
        dpi=300,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close(fig)

    # Run a Mann-Whitney test only when exactly two treatment groups exist.
    if (
        len(groups) == 2
        and all(len(values) > 0 for values in data)
    ):
        try:
            from scipy.stats import mannwhitneyu

            statistic, p_value = mannwhitneyu(
                data[0],
                data[1],
                alternative="two-sided",
            )

            rank_biserial = (
                (2.0 * statistic)
                / (len(data[0]) * len(data[1]))
                - 1.0
            )

            median_difference = float(
                np.median(data[0])
                - np.median(data[1])
            )

            pd.DataFrame(
                [
                    {
                        "test": "two-sided Mann-Whitney U",
                        "group_1": groups[0],
                        "group_2": groups[1],
                        "U_statistic": statistic,
                        "p_value": p_value,
                        "rank_biserial_effect_size": rank_biserial,
                        "median_difference_group1_minus_group2": (
                            median_difference
                        ),
                        "note": (
                            "Plant images, not pixels, "
                            "are the independent observations."
                        ),
                    }
                ]
            ).to_csv(
                OUTPUT_FOLDER / "dsi_treatment_test.csv",
                index=False,
            )

        except ImportError:
            print(
                "SciPy is not installed; "
                "the treatment test was skipped."
            )

    return merged


def main() -> None:
    """Run the complete plant-level DSI analysis."""
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {INPUT_FOLDER}"
        )

    image_paths = sorted(
        path
        for path in INPUT_FOLDER.glob("*.png")
        if path.is_file()
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No PNG masks found in: {INPUT_FOLDER}"
        )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []
    failed = []

    for index, path in enumerate(
        image_paths,
        start=1,
    ):
        try:
            record = analyze_mask(path)
            records.append(record)

            print(
                f"[{index:03d}/{len(image_paths):03d}] "
                f"{path.name}: "
                f"healthy={record['healthy_pixels']:,} px "
                f"({record['healthy_percent']:.2f}%), "
                f"stressed={record['stressed_pixels']:,} px "
                f"({record['stressed_percent']:.2f}%), "
                f"canopy={record['canopy_pixels']:,} px, "
                f"HF={record['healthy_fraction_HF']:.4f}, "
                f"DSI={record['drought_severity_index_DSI']:.4f}"
            )

        except Exception as error:
            # Continue so one bad file does not hide all results.
            failed.append(
                {
                    "filename": path.name,
                    "error": str(error),
                }
            )

            print(
                f"[{index:03d}/{len(image_paths):03d}] "
                f"FAILED {path.name}: {error}"
            )

    if not records:
        raise RuntimeError(
            "No masks could be analyzed. "
            "Review the errors printed above."
        )

    results = pd.DataFrame(records)

    results = add_metadata_and_group_outputs(
        results
    )

    results.to_csv(
        OUTPUT_FOLDER / "per_image_dsi_hf.csv",
        index=False,
    )

    summary = make_dataset_summary(results)

    summary.to_csv(
        OUTPUT_FOLDER / "dataset_summary.csv",
        index=False,
    )

    if failed:
        pd.DataFrame(failed).to_csv(
            OUTPUT_FOLDER / "failed_images.csv",
            index=False,
        )

    save_composition_plot(
        results,
        OUTPUT_FOLDER / "canopy_pixel_composition.png",
    )

    save_distribution_plot(
        results,
        OUTPUT_FOLDER / "dsi_distribution.png",
    )

    display_columns = [
        "filename",
        "healthy_pixels",
        "stressed_pixels",
        "canopy_pixels",
        "healthy_percent",
        "stressed_percent",
        "healthy_fraction_HF",
        "drought_severity_index_DSI",
        "canopy_cover_percent",
    ]

    if "treatment" in results.columns:
        display_columns.insert(
            1,
            "treatment",
        )

    print(
        "\n================ PER-PLANT RESULTS ================"
    )

    print(
        results[display_columns].to_string(
            index=False,
            formatters={
                "healthy_percent": "{:.2f}".format,
                "stressed_percent": "{:.2f}".format,
                "healthy_fraction_HF": "{:.4f}".format,
                "drought_severity_index_DSI": "{:.4f}".format,
                "canopy_cover_percent": "{:.2f}".format,
            },
        )
    )

    print(
        "\n================ DATASET SUMMARY ================"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    unexpected_total = int(
        results["unexpected_pixel_count"].sum()
    )

    zero_canopy = int(
        results["canopy_pixels"].eq(0).sum()
    )

    print("\nAnalysis complete")
    print(
        f"  Successfully analyzed: {len(results)} masks"
    )
    print(
        f"  Failed:                {len(failed)} masks"
    )
    print(
        f"  Unexpected pixels:     {unexpected_total}"
    )
    print(
        f"  Zero-canopy masks:     {zero_canopy}"
    )
    print(
        f"  Results folder:        {OUTPUT_FOLDER}"
    )


if __name__ == "__main__":
    main()