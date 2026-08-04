from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


Number = int | float
Series = Sequence[Number] | Mapping[str, Sequence[Number]]


def generate_paper_pdf_plot(
    x_values: Sequence[Number],
    y_values: Series,
    output_path: str | Path,
    *,
    x_label: str = "",
    y_label: str = "",
    title: str | None = None,
    figure_size: tuple[float, float] = (3.5, 2.6),
    marker: str = "o",
    line_width: float = 1.5,
    dpi: int = 300,
    grid: bool = True,
) -> Path:
    """Generate a publication-ready PDF or PNG line plot.

    Args:
        x_values: Values for the x-axis.
        y_values: Values for the y-axis. Pass a sequence for one line or a
            mapping like {"method_a": [...], "method_b": [...]} for multiple lines.
        output_path: Figure path where the plot will be written. Supported
            extensions are .pdf and .png. If no extension is provided, .pdf is used.
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        title: Optional figure title.
        figure_size: Figure size in inches. Defaults to a single-column width.
        marker: Marker used in each line.
        line_width: Width of each plotted line.
        dpi: Resolution used when saving the figure.
        grid: Whether to include a light grid.

    Returns:
        Path to the generated figure.
    """
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".pdf")
    output_format = output_path.suffix.lower().lstrip(".")
    if output_format not in {"pdf", "png"}:
        raise ValueError(
            f"Unsupported output format '.{output_format}'. Use '.pdf' or '.png'."
        )

    if len(x_values) == 0:
        raise ValueError("x_values cannot be empty")

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, ax = plt.subplots(figsize=figure_size)

        if isinstance(y_values, Mapping):
            for label, series_values in y_values.items():
                _validate_same_length(x_values, series_values, label)
                ax.plot(
                    x_values,
                    series_values,
                    marker=marker,
                    linewidth=line_width,
                    label=label,
                )
            ax.legend(frameon=False)
        else:
            _validate_same_length(x_values, y_values, "y_values")
            ax.plot(x_values, y_values, marker=marker, linewidth=line_width)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        if grid:
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format=output_format, bbox_inches="tight", dpi=dpi)
        plt.close(fig)

    return output_path


def _validate_same_length(
    x_values: Sequence[Number],
    y_values: Sequence[Number],
    label: str,
) -> None:
    if len(x_values) != len(y_values):
        raise ValueError(
            f"x_values and {label} must have the same length: "
            f"{len(x_values)} != {len(y_values)}"
        )
