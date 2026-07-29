from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PDF_DIR = OUT_DIR / "pdf"
PDF_PATH = PDF_DIR / "censored_gp_summary_since_last_meeting.pdf"

PAGE_SIZE = landscape(letter)
PAGE_W, PAGE_H = PAGE_SIZE
MARGIN_X = 0.45 * inch
MARGIN_Y = 0.38 * inch
CONTENT_W = PAGE_W - 2 * MARGIN_X
CONTENT_H = PAGE_H - 2 * MARGIN_Y

BLUE = colors.HexColor("#1F4E79")
TEAL = colors.HexColor("#087E8B")
DARK = colors.HexColor("#202A36")
MUTED = colors.HexColor("#5F6B7A")
LIGHT = colors.HexColor("#EEF4F8")
GRID = colors.HexColor("#D7DEE7")


def load_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            textColor=BLUE,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=BLUE,
            spaceBefore=0,
            spaceAfter=7,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=TEAL,
            spaceBefore=4,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.3,
            leading=12.1,
            textColor=DARK,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=9.5,
            textColor=DARK,
        ),
        "table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.4,
            leading=8.8,
            textColor=DARK,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.4,
            leading=8.8,
            textColor=colors.white,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.7,
            leading=9.4,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=5,
        ),
    }


STYLES = load_styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def esc(text: object) -> str:
    return escape(str(text))


def fmt(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def fmt0(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.0f}"


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{100.0 * float(x):.0f}%"


def scaled_image(path: Path, max_w: float, max_h: float) -> Image:
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    img = Image(str(path), width=w * scale, height=h * scale)
    img.hAlign = "CENTER"
    return img


def figure(path: str, caption: str, max_w: float = CONTENT_W, max_h: float = 5.2 * inch) -> list:
    img_path = OUT_DIR / path
    return [
        scaled_image(img_path, max_w, max_h),
        p(caption, "caption"),
    ]


def two_figures(
    left_path: str,
    left_caption: str,
    right_path: str,
    right_caption: str,
    max_h: float = 3.55 * inch,
) -> Table:
    gap = 0.18 * inch
    col_w = (CONTENT_W - gap) / 2
    left = figure(left_path, left_caption, col_w, max_h)
    right = figure(right_path, right_caption, col_w, max_h)
    table = Table([[left, right]], colWidths=[col_w, col_w])
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return table


def three_figures(paths_and_captions: list[tuple[str, str]], max_h: float = 2.2 * inch) -> Table:
    gap = 0.12 * inch
    col_w = (CONTENT_W - 2 * gap) / 3
    row = [figure(path, caption, col_w, max_h) for path, caption in paths_and_captions]
    table = Table([row], colWidths=[col_w, col_w, col_w])
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return table


def make_table(rows: list[list[object]], col_widths: list[float]) -> Table:
    table_rows = []
    for i, row in enumerate(rows):
        style = "table_header" if i == 0 else "table"
        table_rows.append([p(esc(cell), style) for cell in row])
    table = Table(table_rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    return table


def bullet_items(items: list[str]) -> list:
    flowables = []
    for item in items:
        flowables.append(p("- " + item, "body"))
    return flowables


def row(df: pd.DataFrame, **filters: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        mask &= df[key].eq(value)
    sub = df[mask]
    if len(sub) != 1:
        raise ValueError(f"Expected one row for {filters}, got {len(sub)}")
    return sub.iloc[0]


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN_X, 0.20 * inch, "Censored GP reconstruction update")
    canvas.drawRightString(PAGE_W - MARGIN_X, 0.20 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    gp1d_single = pd.read_csv(OUT_DIR / "gp1d_censored_results.csv")
    gp1d_multi = pd.read_csv(OUT_DIR / "gp1d_multiseed_summary.csv")
    gp2d_multi = pd.read_csv(OUT_DIR / "gp2d_multiseed_fixed_vs_tuned_summary.csv")
    kernel_summary = pd.read_csv(OUT_DIR / "gp2d_kernel_comparison" / "summary.csv")
    path_summary = pd.read_csv(OUT_DIR / "gp2d_path_kernel" / "summary.csv")

    sampled_1d = row(gp1d_single, method="censored sampled")
    clipped_1d = row(gp1d_single, method="exact clipped")
    fixed_1d_sampled = row(gp1d_multi, scenario="fixed_ell_0.55", method="censored sampled")
    tuned_1d_sampled = row(gp1d_multi, scenario="unsat_mll_tuned", method="censored sampled")
    fixed_2d_axis = row(gp2d_multi, scenario="fixed_ell_0.70", shape="axis_gaussian", method="censored sampled")
    tuned_2d_axis = row(gp2d_multi, scenario="unsat_mll_tuned", shape="axis_gaussian", method="censored sampled")
    fixed_2d_laser = row(gp2d_multi, scenario="fixed_ell_0.70", shape="laser_path", method="censored sampled")
    tuned_2d_laser = row(gp2d_multi, scenario="unsat_mll_tuned", shape="laser_path", method="censored sampled")
    rotated_iso = row(kernel_summary, kernel="isotropic_fixed", shape="rotated_wake", method="censored sampled")
    rotated_inf = row(kernel_summary, kernel="anisotropic_informed", shape="rotated_wake", method="censored sampled")
    laser_iso_kernel = row(kernel_summary, kernel="isotropic_fixed", shape="laser_path", method="censored sampled")
    laser_global_aniso = row(kernel_summary, kernel="anisotropic_informed", shape="laser_path", method="censored sampled")
    path_fixed = row(path_summary, kernel="path_aligned_fixed", method="censored sampled")
    path_tuned = row(path_summary, kernel="path_aligned_tuned", method="censored sampled")

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=PAGE_SIZE,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_Y,
        bottomMargin=MARGIN_Y + 0.12 * inch,
    )
    story = []

    story.append(Spacer(1, 0.35 * inch))
    story.append(p("Censored GP Reconstruction Experiments", "title"))
    story.append(p("Summary since last meeting: 1D uncertainty, lengthscale tuning, 2D shapes, observation density, and kernel priors", "subtitle"))
    story.extend(
        bullet_items(
            [
                "The censored likelihood helps most when paired with a useful prior. Saturated pixels provide inequality information, but the GP prior decides how that information propagates into the hidden hot region.",
                f"In the 1D fixed-lengthscale test, sampled censored GP reduced peak error from {fmt0(clipped_1d['peak_abs_error'])} K for clipped regression to {fmt0(sampled_1d['peak_abs_error'])} K and covered the true peak.",
                f"Unsaturated-only lengthscale tuning was not uniformly beneficial: in 1D, sampled censored coverage fell from {fmt_pct(fixed_1d_sampled['true_peak_coverage'])} with fixed ell to {fmt_pct(tuned_1d_sampled['true_peak_coverage'])} with tuned ell.",
                f"In 2D, isotropic RBF worked reasonably for compact fields but struggled on the moving-laser path; sampled censored peak error was about {fmt0(fixed_2d_laser['peak_abs_error_mean'])} K with fixed isotropic ell.",
                f"The path-aligned fixed kernel was the clearest improvement for the moving-laser case: sampled censored peak error dropped to {fmt0(path_fixed['peak_abs_error_mean'])} K with {fmt_pct(path_fixed['true_peak_coverage'])} peak coverage.",
            ]
        )
    )
    story.append(Spacer(1, 0.10 * inch))
    story.append(p("Experiment map", "subsection"))
    story.append(
        make_table(
            [
                ["Experiment", "Kernel/prior used", "Main comparison", "Main lesson"],
                ["1D uncertainty bands", "1D RBF, fixed ell = 0.55", "exact, discard, censored Laplace, censored sampled, oracle", "Sampling from the inequality-conditioned posterior gives a much more plausible peak interval."],
                ["1D lengthscale sweep/tuning", "1D RBF, fixed sweep and unsaturated MLL tuning", "ell in {0.3, 0.5, 0.8, 1.2, 1.8}; then 30 seeds", "Lengthscale is not a nuisance parameter; it controls whether the hidden peak can rise."],
                ["2D shapes and density", "Isotropic 2D RBF, ell selected from unsaturated MLL", "axis Gaussian, rotated wake, moving-laser path; 13x13, 17x17, 21x21 observations", "More observations help, but geometry mismatch remains visible."],
                ["2D fixed vs tuned ell", "Isotropic 2D RBF, fixed ell = 0.70 vs unsaturated MLL tuned", "20 seeds per shape", "Unsaturated tuning often selects smoother priors and can worsen hidden-peak recovery."],
                ["Kernel misspecification", "Isotropic, anisotropic tuned, anisotropic informed, path-aligned", "10 seeds, same GP methods", "Geometry-aware kernels are essential for wake/path-like fields."],
            ],
            [1.35 * inch, 2.25 * inch, 2.55 * inch, 3.25 * inch],
        )
    )
    story.append(PageBreak())

    story.append(p("1D Uncertainty Bands", "section"))
    story.append(
        p(
            "Setup: a 1D Gaussian-like temperature profile was censored by y_i = min(T_i + eps_i, c). "
            "The GP used a squared-exponential/RBF kernel k(x,x') = sigma_f^2 exp(-(x-x')^2/(2 ell^2)) with fixed ell = 0.55, sigma_f = 500, and noise sd = 20. "
            "The saturated likelihood contribution is P(y_i = c | f_i) = Phi((f_i - c)/sigma), and the sampled censored GP approximates the posterior conditioned on f_S >= c.",
            "body",
        )
    )
    story.extend(
        figure(
            "gp1d_censored_one_sd_bands.png",
            f"One-standard-deviation posterior bands. Exact clipped is sharply biased downward; sampled censored GP gives a broad but plausible hidden-peak range and lowers peak error to {fmt0(sampled_1d['peak_abs_error'])} K.",
            CONTENT_W,
            5.55 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("1D Lengthscale Sweep", "section"))
    story.append(
        p(
            "Only the RBF lengthscale ell was varied. This tests Adrienne's point that recovering a clipped peak requires a smoothness assumption: if the prior is too local, the interior of the saturated region can stay close to the threshold; if it is too smooth, the model can again underfit the peak.",
            "body",
        )
    )
    story.append(
        two_figures(
            "gp1d_lengthscale_sweep_field_error.png",
            "Relative field error across fixed lengthscales.",
            "gp1d_lengthscale_sweep_peak_intervals.png",
            "Peak interval behavior across fixed lengthscales.",
            5.05 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("1D Multi-seed Lengthscale Tuning", "section"))
    story.append(
        p(
            "I repeated the 1D experiment over 30 random observation/noise seeds and compared fixed ell = 0.55 with ell selected by maximizing the marginal likelihood of only the unsaturated observations.",
            "body",
        )
    )
    rows = [["Scenario", "Method", "Field L2", "Hot-region L2", "Peak error", "Peak coverage", "Mean ell"]]
    for scenario, label in [("fixed_ell_0.55", "fixed ell = 0.55"), ("unsat_mll_tuned", "unsat MLL tuned")]:
        for method in ["exact clipped", "discard saturated", "censored Laplace", "censored sampled"]:
            r = row(gp1d_multi, scenario=scenario, method=method)
            rows.append(
                [
                    label,
                    method,
                    fmt(r["field_rel_l2_mean"]),
                    fmt(r["hot_region_rel_l2_mean"]),
                    fmt0(r["peak_abs_error_mean"]),
                    fmt_pct(r["true_peak_coverage"]),
                    fmt(r["selected_lengthscale_mean"], 2),
                ]
            )
    story.append(make_table(rows, [1.35 * inch, 1.35 * inch, 0.82 * inch, 0.95 * inch, 0.85 * inch, 0.95 * inch, 0.70 * inch]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        three_figures(
            [
                ("gp1d_multiseed_peak_error.png", "Peak error over seeds."),
                ("gp1d_multiseed_peak_coverage.png", "True peak inside the 95% interval."),
                ("gp1d_selected_lengthscales.png", "Selected ell from unsaturated MLL."),
            ],
            2.35 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("2D Shape Study: Axis Gaussian", "section"))
    story.append(
        p(
            "Setup: the first 2D extension used an isotropic 2D RBF kernel k(z,z') = sigma_f^2 exp(-||z-z'||^2/(2 ell^2)). "
            "For the shape/density sweep, ell was selected from unsaturated observations. The axis Gaussian is the closest 2D analogue of the original toy example.",
            "body",
        )
    )
    story.extend(
        figure(
            "gp2d_shape_density_reconstructions_axis_gaussian.png",
            "Axis Gaussian reconstruction examples across observation density. The isotropic RBF prior is well matched to this geometry.",
            CONTENT_W,
            5.35 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("2D Shape Study: Rotated Wake", "section"))
    story.append(
        p(
            "The rotated wake introduces anisotropy and orientation. This is a first test of model misspecification: the isotropic RBF does not encode the long, tilted correlation structure of the field.",
            "body",
        )
    )
    story.extend(
        figure(
            "gp2d_shape_density_reconstructions_rotated_wake.png",
            "Rotated wake reconstruction examples. Censored methods still use the inequality information, but the isotropic kernel has to explain a directionally elongated field.",
            CONTENT_W,
            5.35 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("2D Shape Study: Moving-laser Path", "section"))
    story.append(
        p(
            "The moving-laser path is the hardest case for a plain isotropic RBF because the high-temperature structure is curved and localized along a path, not radially smooth around one center.",
            "body",
        )
    )
    story.extend(
        figure(
            "gp2d_shape_density_reconstructions_laser_path.png",
            "Moving-laser reconstruction examples. The isotropic kernel tends to smear or under-estimate the hidden path peak.",
            CONTENT_W,
            5.35 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("Observation Density", "section"))
    story.append(
        p(
            "I varied the observation grid size from 13x13 to 21x21 while keeping the same three shapes and the same isotropic RBF setup with unsaturated-data lengthscale selection. This tests whether denser camera sampling alone fixes saturation-induced peak loss.",
            "body",
        )
    )
    story.append(
        two_figures(
            "gp2d_shape_density_field_error.png",
            "Relative field error versus observation density.",
            "gp2d_shape_density_hot_region_error.png",
            "Hot-region error versus observation density.",
            3.10 * inch,
        )
    )
    story.append(Spacer(1, 0.06 * inch))
    story.append(
        two_figures(
            "gp2d_shape_density_peak_error.png",
            "Peak error versus observation density.",
            "gp2d_shape_density_peak_coverage.png",
            "True peak coverage versus observation density.",
            3.10 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("2D Fixed vs Tuned Isotropic Lengthscale", "section"))
    story.append(
        p(
            "To separate density effects from hyperparameter effects, I repeated the 2D shape experiment over 20 seeds and compared fixed ell = 0.70 with ell selected from unsaturated marginal likelihood.",
            "body",
        )
    )
    rows = [["Shape", "Scenario", "Sampled field L2", "Sampled peak error", "Sampled coverage", "Mean ell"]]
    for shape, label in [("axis_gaussian", "axis Gaussian"), ("rotated_wake", "rotated wake"), ("laser_path", "moving-laser path")]:
        for scenario, s_label in [("fixed_ell_0.70", "fixed ell = 0.70"), ("unsat_mll_tuned", "unsat MLL tuned")]:
            r = row(gp2d_multi, scenario=scenario, shape=shape, method="censored sampled")
            rows.append(
                [
                    label,
                    s_label,
                    fmt(r["field_rel_l2_mean"]),
                    fmt0(r["peak_abs_error_mean"]),
                    fmt_pct(r["true_peak_coverage"]),
                    fmt(r["selected_lengthscale_mean"], 2),
                ]
            )
    story.append(make_table(rows, [1.35 * inch, 1.40 * inch, 1.05 * inch, 1.05 * inch, 1.00 * inch, 0.85 * inch]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        three_figures(
            [
                ("gp2d_multiseed_fixed_vs_tuned_peak_error.png", "Peak error over 20 seeds."),
                ("gp2d_multiseed_fixed_vs_tuned_peak_coverage.png", "Peak coverage over 20 seeds."),
                ("gp2d_multiseed_fixed_vs_tuned_lengthscales.png", "Selected isotropic ell values."),
            ],
            2.25 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("Kernel Misspecification: Isotropic and Global Anisotropic Kernels", "section"))
    story.append(
        p(
            "I then changed the kernel rather than the observation model. Kernels tested: isotropic fixed ell = 0.70; isotropic tuned by unsaturated MLL; anisotropic tuned over ell_x, ell_y, and angle; and informed anisotropic kernels chosen from the known shape geometry.",
            "body",
        )
    )
    story.append(
        p(
            "The anisotropic kernel used rotated coordinates (u, v): k(z,z') = sigma_f^2 exp(-(u-u')^2/(2 ell_x^2) - (v-v')^2/(2 ell_y^2)). For the rotated wake, the informed kernel used angle 34 deg, ell_x = 1.25, ell_y = 0.28.",
            "body",
        )
    )
    rows = [["Case", "Isotropic fixed sampled peak", "Informed anisotropic sampled peak", "Coverage change"]]
    rows.append(["rotated wake", fmt0(rotated_iso["peak_abs_error_mean"]), fmt0(rotated_inf["peak_abs_error_mean"]), f"{fmt_pct(rotated_iso['true_peak_coverage'])} to {fmt_pct(rotated_inf['true_peak_coverage'])}"])
    rows.append(["moving-laser path", fmt0(laser_iso_kernel["peak_abs_error_mean"]), fmt0(laser_global_aniso["peak_abs_error_mean"]), f"{fmt_pct(laser_iso_kernel['true_peak_coverage'])} to {fmt_pct(laser_global_aniso['true_peak_coverage'])}"])
    story.append(make_table(rows, [1.35 * inch, 1.70 * inch, 1.85 * inch, 1.25 * inch]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        two_figures(
            "gp2d_kernel_comparison/sampled_peak_error.png",
            "Sampled censored GP peak error under global kernels.",
            "gp2d_kernel_comparison/sampled_field_error.png",
            "Sampled censored GP field error under global kernels.",
            3.05 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("Kernel Misspecification: Reconstruction Example", "section"))
    story.append(
        p(
            "For the rotated wake, the reconstruction panel makes the kernel effect more visible than the error bars alone. The informed anisotropic kernel uses the wake direction and different along-wake/across-wake lengthscales, so the recovered hot region follows the elongated tilted structure more naturally than the isotropic alternatives.",
            "body",
        )
    )
    story.extend(
        figure(
            "gp2d_kernel_comparison/sampled_reconstructions_rotated_wake_seed00.png",
            "Rotated-wake censored-sampled reconstructions for seed 0 under isotropic and anisotropic global kernels.",
            CONTENT_W,
            4.85 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("Kernel Misspecification: Coverage and Selected Parameters", "section"))
    story.append(
        p(
            "The global anisotropic kernel improved straight or nearly straight anisotropic structure, especially the rotated wake, but it was not enough for the curved moving-laser path. This motivated a kernel whose coordinates follow the laser trajectory.",
            "body",
        )
    )
    story.append(
        two_figures(
            "gp2d_kernel_comparison/peak_coverage.png",
            "Peak coverage for all GP methods and global kernels.",
            "gp2d_kernel_comparison/anisotropic_selected_parameters.png",
            "Parameters selected by unsaturated MLL for anisotropic tuned kernels.",
            4.05 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("Path-aligned Kernel for the Moving-laser Field", "section"))
    story.append(
        p(
            "For the moving-laser field, I added a path-aligned kernel. Each point (x, y) is mapped approximately to path coordinates (s, r), where s is position along the known laser path and r is signed distance away from it. The kernel is anisotropic in these coordinates: k = sigma_f^2 exp(-(s-s')^2/(2 ell_s^2) - (r-r')^2/(2 ell_r^2)).",
            "body",
        )
    )
    story.append(
        p(
            "Two versions were tested: path-aligned fixed with ell_s = 1.10 and ell_r = 0.20, and path-aligned tuned with ell_s and ell_r selected from unsaturated MLL. The tuned version usually chose ell_s near 2.05 and ell_r = 0.45, which over-smoothed the hidden peak.",
            "body",
        )
    )
    story.extend(
        figure(
            "gp2d_path_kernel/sampled_reconstructions_seed00.png",
            "Moving-laser sampled reconstructions for seed 0. The path-aligned fixed kernel is the first prior that reconstructs the hidden peak shape convincingly.",
            CONTENT_W,
            4.65 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("Path-aligned Kernel: Axis Gaussian Check", "section"))
    story.append(
        p(
            "As a sanity check, I also applied the path-kernel comparison panel to the axis Gaussian. This field is not generated by the laser path, so the global anisotropic kernel is the more natural geometry-aware prior here, while the path-aligned panels show what the laser-path prior does when its geometry is not needed.",
            "body",
        )
    )
    story.extend(
        figure(
            "gp2d_path_kernel/sampled_reconstructions_axis_gaussian_seed00.png",
            "Axis-Gaussian censored-sampled reconstructions for seed 0 under isotropic fixed, global anisotropic, path-aligned fixed, and path-aligned tuned kernels.",
            CONTENT_W,
            4.65 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("Path-aligned Kernel: Rotated Wake Check", "section"))
    story.append(
        p(
            "The rotated wake is useful as a misspecification check: it has a preferred direction, but not the curved laser-path geometry. This panel helps separate the value of generic anisotropy from the value of a path-specific kernel.",
            "body",
        )
    )
    story.extend(
        figure(
            "gp2d_path_kernel/sampled_reconstructions_rotated_wake_seed00.png",
            "Rotated-wake censored-sampled reconstructions for seed 0 under isotropic fixed, global anisotropic, path-aligned fixed, and path-aligned tuned kernels.",
            CONTENT_W,
            4.65 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("Path-aligned Kernel Results", "section"))
    story.append(
        p(
            f"For censored sampled GP, the path-aligned fixed kernel reduced mean peak error from {fmt0(path_summary[(path_summary['kernel'] == 'isotropic_fixed') & (path_summary['method'] == 'censored sampled')].iloc[0]['peak_abs_error_mean'])} K under isotropic fixed to {fmt0(path_fixed['peak_abs_error_mean'])} K, and peak coverage improved from 0% to {fmt_pct(path_fixed['true_peak_coverage'])}. The tuned path-aligned kernel had lower global field error but missed the peak because the unsaturated data favored an overly smooth path prior.",
            "body",
        )
    )
    rows = [["Kernel", "Field L2", "Hot-region L2", "Peak error", "Peak coverage", "ell_s", "ell_r"]]
    for kernel in ["isotropic_fixed", "anisotropic_informed", "path_aligned_fixed", "path_aligned_tuned"]:
        r = row(path_summary, kernel=kernel, method="censored sampled")
        rows.append(
            [
                r["kernel_label"],
                fmt(r["field_rel_l2_mean"]),
                fmt(r["hot_region_rel_l2_mean"]),
                fmt0(r["peak_abs_error_mean"]),
                fmt_pct(r["true_peak_coverage"]),
                fmt(r["lengthscale_s_mean"], 2),
                fmt(r["lengthscale_r_mean"], 2),
            ]
        )
    story.append(make_table(rows, [1.55 * inch, 0.80 * inch, 0.92 * inch, 0.85 * inch, 0.95 * inch, 0.55 * inch, 0.55 * inch]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        three_figures(
            [
                ("gp2d_path_kernel/sampled_peak_error.png", "Sampled censored GP peak error."),
                ("gp2d_path_kernel/peak_coverage.png", "Peak coverage for all methods."),
                ("gp2d_path_kernel/path_selected_parameters.png", "Path-kernel parameters selected from unsaturated MLL."),
            ],
            2.35 * inch,
        )
    )
    story.append(PageBreak())

    story.append(p("Interpretation and Next Numerical Step", "section"))
    story.extend(
        bullet_items(
            [
                "The important variable is now the prior/kernel, not just the censoring likelihood. Censoring tells the model that hidden values exceeded the camera ceiling; the kernel decides how high and where that hidden mass can plausibly be.",
                "Exact clipped GP remains a useful negative control: it is consistently biased downward and overconfident near the saturated peak.",
                "Discarding saturated pixels can be competitive when the prior is already strong, but it wastes direct inequality information and can create overly broad or shape-dependent uncertainty.",
                "Laplace censored GP is useful as a cheap approximation, but sampled censored GP is more informative for peak uncertainty because the inequality-conditioned posterior can be skewed.",
                "Unsaturated-only MLL is not enough for hyperparameter selection in this problem. It often sees the smooth cool region well but cannot see the hidden hot peak.",
                "The most promising direction is geometry-aware censored GP: use kernels informed by the heat source path, wake direction, material anisotropy, or a simple physics model, then select hyperparameters using the censored likelihood or validation on synthetic fields.",
            ]
        )
    )
    story.append(Spacer(1, 0.12 * inch))
    story.append(p("Candidate next experiment", "subsection"))
    story.append(
        p(
            "Use the path-aligned kernel as the main moving-laser baseline, then compare fixed geometry-informed hyperparameters against hyperparameters learned with the full censored likelihood rather than only unsaturated observations. This directly tests whether the saturation-aware likelihood can improve both reconstruction and calibration when the kernel family is appropriate.",
            "body",
        )
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build_pdf()
    print(PDF_PATH)
