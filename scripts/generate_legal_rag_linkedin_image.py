"""Generate the mobile-friendly Legal RAG Bench results image for LinkedIn."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1080
HEIGHT = 1350
OUTPUT = Path("docs/images/legal-rag-bench-linkedin-results.png")

BACKGROUND = "#F6F8FB"
CARD = "#FFFFFF"
TEXT = "#111827"
MUTED = "#667085"
BORDER = "#D9E0E8"
TRACK = "#E9EDF2"
BLUE = "#2F78D4"
BLUE_PALE = "#EAF2FC"
GREEN = "#19A974"
GREEN_PALE = "#E8F7F1"

FONT_CANDIDATES = {
    "regular": [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ],
    "bold": [
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ],
}


def font(style: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a common system font without bundling a font file in the repository."""
    for candidate in FONT_CANDIDATES[style]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    raise FileNotFoundError(f"No supported {style} font found")


FONTS = {
    "eyebrow": font("bold", 23),
    "title": font("bold", 57),
    "subtitle": font("regular", 25),
    "legend": font("regular", 24),
    "metric_label": font("bold", 20),
    "metric_text": font("regular", 23),
    "card_title": font("bold", 34),
    "card_subtitle": font("regular", 23),
    "badge": font("bold", 18),
    "bar_label": font("regular", 23),
    "percent": font("bold", 29),
    "note": font("regular", 21),
    "brand": font("bold", 20),
}


def rounded_bar(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    value: int,
    color: str,
) -> None:
    """Draw a percentage bar with a neutral 100% track."""
    height = 22
    draw.rounded_rectangle((x, y, x + width, y + height), radius=11, fill=TRACK)
    fill_width = max(height, round(width * value / 100))
    draw.rounded_rectangle(
        (x, y, x + fill_width, y + height), radius=11, fill=color
    )


def badge(
    draw: ImageDraw.ImageDraw,
    *,
    right: int,
    y: int,
    label: str,
    fill: str,
    color: str,
) -> None:
    """Draw a right-aligned metric badge."""
    box = draw.textbbox((0, 0), label, font=FONTS["badge"])
    width = box[2] - box[0] + 30
    draw.rounded_rectangle(
        (right - width, y, right, y + 36), radius=18, fill=fill
    )
    draw.text(
        (right - width / 2, y + 17),
        label,
        font=FONTS["badge"],
        fill=color,
        anchor="mm",
    )


def result_card(
    draw: ImageDraw.ImageDraw,
    *,
    y: int,
    title: str,
    subtitle: str,
    metric: str,
    retrieval_label: str,
    retrieval: int,
    correct: int,
) -> None:
    """Draw one architecture result card."""
    left = 64
    right = WIDTH - 64
    draw.rounded_rectangle(
        (left, y, right, y + 205),
        radius=22,
        fill=CARD,
        outline=BORDER,
        width=2,
    )

    draw.text((92, y + 24), title, font=FONTS["card_title"], fill=TEXT)
    title_width = draw.textlength(title, font=FONTS["card_title"])
    draw.text(
        (100 + title_width, y + 31),
        subtitle,
        font=FONTS["card_subtitle"],
        fill=MUTED,
    )
    badge(
        draw,
        right=right - 27,
        y=y + 25,
        label=metric,
        fill=BLUE_PALE,
        color=BLUE,
    )

    label_x = 92
    bar_x = 265
    bar_width = 590
    percent_x = 968

    draw.text(
        (label_x, y + 93),
        retrieval_label,
        font=FONTS["bar_label"],
        fill=TEXT,
        anchor="lm",
    )
    rounded_bar(
        draw,
        x=bar_x,
        y=y + 82,
        width=bar_width,
        value=retrieval,
        color=BLUE,
    )
    draw.text(
        (percent_x, y + 93),
        f"{retrieval}%",
        font=FONTS["percent"],
        fill=TEXT,
        anchor="rm",
    )

    draw.text(
        (label_x, y + 154),
        "Correct",
        font=FONTS["bar_label"],
        fill=TEXT,
        anchor="lm",
    )
    rounded_bar(
        draw,
        x=bar_x,
        y=y + 143,
        width=bar_width,
        value=correct,
        color=GREEN,
    )
    draw.text(
        (percent_x, y + 154),
        f"{correct}%",
        font=FONTS["percent"],
        fill=TEXT,
        anchor="rm",
    )


def main() -> None:
    """Render and save the LinkedIn image."""
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text(
        (64, 55),
        "LEGAL RAG BENCH  ·  PHASE 1",
        font=FONTS["eyebrow"],
        fill=BLUE,
    )
    draw.text(
        (WIDTH - 64, 68),
        "RAG Evaluator",
        font=FONTS["brand"],
        fill=TEXT,
        anchor="rm",
    )
    draw.text(
        (64, 96),
        "Dense vs hybrid vs",
        font=FONTS["title"],
        fill=TEXT,
    )
    draw.text(
        (64, 158),
        "agentic retrieval",
        font=FONTS["title"],
        fill=TEXT,
    )
    draw.text(
        (64, 236),
        "100 expert questions · same corpus, generator, and judge",
        font=FONTS["subtitle"],
        fill=MUTED,
    )

    draw.rounded_rectangle((64, 292, 86, 314), radius=6, fill=BLUE)
    draw.text(
        (100, 303),
        "Gold-passage retrieval",
        font=FONTS["legend"],
        fill=TEXT,
        anchor="lm",
    )
    draw.rounded_rectangle((393, 292, 415, 314), radius=6, fill=GREEN)
    draw.text(
        (429, 303),
        "Correct answer (judged)",
        font=FONTS["legend"],
        fill=TEXT,
        anchor="lm",
    )

    draw.rounded_rectangle(
        (64, 346, WIDTH - 64, 446),
        radius=18,
        fill=BLUE_PALE,
    )
    draw.text(
        (88, 365),
        "RETRIEVAL METRICS",
        font=FONTS["metric_label"],
        fill=BLUE,
    )
    draw.text(
        (88, 400),
        "Vector systems: gold passage in top 5 (hit@5)",
        font=FONTS["metric_text"],
        fill=TEXT,
    )
    draw.text(
        (575, 400),
        "Agent: gold passage read during the run",
        font=FONTS["metric_text"],
        fill=TEXT,
    )

    result_card(
        draw,
        y=478,
        title="Dense",
        subtitle="Chroma",
        metric="HIT@5",
        retrieval_label="Retrieval",
        retrieval=53,
        correct=61,
    )
    result_card(
        draw,
        y=701,
        title="Hybrid",
        subtitle="Qdrant + SPLADE",
        metric="HIT@5",
        retrieval_label="Retrieval",
        retrieval=41,
        correct=39,
    )
    result_card(
        draw,
        y=924,
        title="Agent",
        subtitle="Filesystem",
        metric="GOLD ACCESS",
        retrieval_label="Gold access",
        retrieval=88,
        correct=82,
    )

    draw.rounded_rectangle(
        (64, 1161, WIDTH - 64, 1263),
        radius=18,
        fill=GREEN_PALE,
    )
    draw.text(
        (88, 1181),
        "TRADE-OFF",
        font=FONTS["metric_label"],
        fill=GREEN,
    )
    draw.text(
        (88, 1217),
        "Agent: 192s and ~$0.015/question  ·  Vector: 6–7s and ~$0.0002/question",
        font=FONTS["note"],
        fill=TEXT,
    )

    draw.text(
        (64, 1303),
        "Metrics differ; compare retrieval directionally. All systems completed 100/100 questions.",
        font=FONTS["note"],
        fill=MUTED,
        anchor="lm",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    print(f"Wrote {OUTPUT} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
