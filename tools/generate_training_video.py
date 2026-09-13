#!/usr/bin/env python3
"""Render a captioned, colour-coded MLP training walkthrough as an MP4.

The animation intentionally describes this repository's binary classifier:
two ReLU hidden layers, a sigmoid output, and binary cross-entropy.  Softmax
appears only as a final comparison for a multiclass variant.

Run from the repository root:

    python tools/generate_training_video.py

The default output is a 64-second H.264 MP4 at
``images/mlp-training-walkthrough.mp4``.  Pillow and OpenCV generate the
frames; VLC is used when available to make the MP4 broadly browser-compatible.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1280
HEIGHT = 720
DEFAULT_FPS = 24
DEFAULT_SECONDS = 64

BACKGROUND = "#f7fafc"
INK = "#102a43"
MUTED = "#627d98"
LIGHT_BORDER = "#d9e2ec"
BLUE = "#2563eb"
LIGHT_BLUE = "#dbeafe"
PURPLE = "#7c3aed"
LIGHT_PURPLE = "#ede9fe"
ORANGE = "#ea580c"
LIGHT_ORANGE = "#ffedd5"
RED = "#dc2626"
LIGHT_RED = "#fee2e2"
GREEN = "#059669"
LIGHT_GREEN = "#d1fae5"
GREY_NODE = "#e5e7eb"
GREY_STROKE = "#94a3b8"

REGULAR_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BOLD_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def rgb(value: str) -> tuple[int, int, int]:
    """Convert a hexadecimal colour to an RGB tuple."""
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def blend(first: str | tuple[int, int, int], second: str | tuple[int, int, int], amount: float):
    """Return a colour interpolated from first to second."""
    first_rgb = rgb(first) if isinstance(first, str) else first
    second_rgb = rgb(second) if isinstance(second, str) else second
    amount = max(0.0, min(1.0, amount))
    return tuple(round(a + (b - a) * amount) for a, b in zip(first_rgb, second_rgb))


def ease(value: float) -> float:
    """Use a smooth animation curve for values in the 0..1 range."""
    value = max(0.0, min(1.0, value))
    return value * value * (3 - 2 * value)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load the diagram font at a predictable size."""
    path = BOLD_FONT if bold else REGULAR_FONT
    return ImageFont.truetype(path, size=size)


def text_width(draw: ImageDraw.ImageDraw, value: str, selected_font) -> float:
    return draw.textlength(value, font=selected_font)


def draw_centered(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    value: str,
    selected_font,
    fill,
) -> None:
    """Draw text centered horizontally at an x coordinate."""
    draw.text((x - text_width(draw, value, selected_font) / 2, y), value, font=selected_font, fill=fill)


def wrap_lines(draw: ImageDraw.ImageDraw, value: str, selected_font, max_width: float) -> list[str]:
    """Split text into words that fit the requested width."""
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or text_width(draw, candidate, selected_font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    value: str,
    selected_font,
    fill,
    max_width: float,
    line_gap: int = 5,
) -> int:
    """Draw wrapped text and return its occupied height."""
    lines = wrap_lines(draw, value, selected_font, max_width)
    bbox = selected_font.getbbox("Hg")
    line_height = bbox[3] - bbox[1] + line_gap
    for index, line in enumerate(lines):
        draw.text((x, y + index * line_height), line, font=selected_font, fill=fill)
    return len(lines) * line_height


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill,
    outline=None,
    radius: int = 18,
    width: int = 2,
) -> None:
    """Draw a consistently styled rounded card."""
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill,
    width: int = 4,
    head: int = 12,
) -> None:
    """Draw a line with an arrowhead at its end."""
    start_x, start_y = start
    end_x, end_y = end
    draw.line((start_x, start_y, end_x, end_y), fill=fill, width=width)
    angle = math.atan2(end_y - start_y, end_x - start_x)
    left = (
        end_x - head * math.cos(angle - math.pi / 6),
        end_y - head * math.sin(angle - math.pi / 6),
    )
    right = (
        end_x - head * math.cos(angle + math.pi / 6),
        end_y - head * math.sin(angle + math.pi / 6),
    )
    draw.polygon((end, left, right), fill=fill)


def draw_dot(draw: ImageDraw.ImageDraw, x: float, y: float, radius: int, fill, outline=None) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline)


def point_on_line(start: tuple[float, float], end: tuple[float, float], amount: float) -> tuple[float, float]:
    return (
        start[0] + (end[0] - start[0]) * amount,
        start[1] + (end[1] - start[1]) * amount,
    )


def draw_header(draw: ImageDraw.ImageDraw, time_seconds: float, scene_number: int, title: str) -> None:
    """Draw the shared video heading and the 64-second progress bar."""
    rounded_box(draw, (28, 22, WIDTH - 28, 108), rgb("#ffffff"), rgb(LIGHT_BORDER), 22, 2)
    draw.text((56, 39), "MULTILAYER PERCEPTRON · VISUAL TRAINING GUIDE", font=font(15, True), fill=rgb(BLUE))
    draw.text((56, 62), title, font=font(28, True), fill=rgb(INK))

    badge = f"STEP {scene_number} / 9"
    rounded_box(draw, (1045, 41, 1214, 73), rgb(LIGHT_PURPLE), None, 16, 0)
    draw_centered(draw, 1129, 50, badge, font(13, True), rgb(PURPLE))

    progress = clamp(time_seconds / DEFAULT_SECONDS)
    rounded_box(draw, (56, 91, 1214, 99), rgb("#e7eef5"), None, 4, 0)
    rounded_box(draw, (56, 91, 56 + int(1158 * progress), 99), rgb(BLUE), None, 4, 0)


def draw_caption(
    draw: ImageDraw.ImageDraw,
    colour: str,
    heading: str,
    body: str,
) -> None:
    """Draw the prominent, readable caption bar used as the silent narration."""
    rounded_box(draw, (44, 602, WIDTH - 44, 694), blend(BACKGROUND, colour, 0.10), rgb(LIGHT_BORDER), 20, 2)
    draw_dot(draw, 76, 633, 14, rgb(colour))
    draw.text((102, 617), heading, font=font(18, True), fill=rgb(INK))
    draw_wrapped(draw, 102, 644, body, font(15), rgb(MUTED), 1080, 4)


def draw_layer_heading(draw: ImageDraw.ImageDraw, x: int, title: str, detail: str, colour: str) -> None:
    draw_centered(draw, x, 155, title, font(16, True), rgb(INK))
    draw_centered(draw, x, 178, detail, font(13), rgb(colour))


def draw_network(
    draw: ImageDraw.ImageDraw,
    clock: float,
    *,
    forward: bool = False,
    backward: bool = False,
    h1_active: tuple[bool, bool, bool, bool] = (True, True, True, True),
    h2_active: tuple[bool, bool, bool, bool] = (True, True, True, True),
    probability: float | None = None,
    show_labels: bool = True,
    update_strength: float = 0.0,
) -> None:
    """Draw a representative 30 → 24 → 24 → 1 fully connected network."""
    inputs = [(190, 238), (190, 322), (190, 406), (190, 490)]
    hidden_one = [(470, 238), (470, 322), (470, 406), (470, 490)]
    hidden_two = [(750, 238), (750, 322), (750, 406), (750, 490)]
    output = (1040, 364)

    if show_labels:
        draw_layer_heading(draw, 190, "Input layer", "30 scaled features", BLUE)
        draw_layer_heading(draw, 470, "Hidden layer 1", "24 ReLU neurons", PURPLE)
        draw_layer_heading(draw, 750, "Hidden layer 2", "24 ReLU neurons", PURPLE)
        draw_layer_heading(draw, 1040, "Output layer", "1 sigmoid neuron", GREEN)

    def draw_connections(source_nodes, target_nodes, kind: int) -> None:
        for source_index, source in enumerate(source_nodes):
            for target_index, target in enumerate(target_nodes):
                index = source_index * len(target_nodes) + target_index
                base = rgb("#c9d8e8")
                if update_strength:
                    base = blend("#c9d8e8", GREEN, 0.30 * update_strength)
                draw.line((source, target), fill=base, width=2)

                if forward:
                    phase = (clock * 1.25 + index * 0.095 + kind * 0.23) % 1.0
                    if phase < 0.92:
                        pulse = point_on_line(source, target, phase)
                        draw_dot(draw, *pulse, 4, rgb(BLUE))

                if backward and (source_index + target_index + kind) % 2 == 0:
                    phase = (clock * 1.4 + index * 0.13) % 1.0
                    pulse = point_on_line(source, target, 1.0 - phase)
                    draw_dot(draw, *pulse, 4, rgb(RED))

    draw_connections(inputs, hidden_one, 0)
    draw_connections(hidden_one, hidden_two, 1)
    for index, source in enumerate(hidden_two):
        draw.line((source, output), fill=blend("#c9d8e8", GREEN, 0.24 * update_strength), width=2)
        if forward:
            phase = (clock * 1.25 + index * 0.16 + 0.2) % 1.0
            draw_dot(draw, *point_on_line(source, output, phase), 4, rgb(BLUE))
        if backward and index % 2 == 0:
            phase = (clock * 1.4 + index * 0.17) % 1.0
            draw_dot(draw, *point_on_line(source, output, 1.0 - phase), 4, rgb(RED))

    input_names = ("x1", "x2", "…", "x30")
    for position, name in zip(inputs, input_names):
        draw_dot(draw, *position, 22, rgb(LIGHT_BLUE), rgb(BLUE))
        draw_centered(draw, position[0], position[1] - 8, name, font(13, True), rgb(BLUE))

    hidden_names = ("h1", "h2", "⋮", "h24")
    for position, name, is_active in zip(hidden_one, hidden_names, h1_active):
        node_fill = rgb(LIGHT_PURPLE) if is_active else rgb(GREY_NODE)
        node_outline = rgb(PURPLE) if is_active else rgb(GREY_STROKE)
        draw_dot(draw, *position, 27, node_fill, node_outline)
        draw_centered(draw, position[0], position[1] - 9, name, font(13, True), node_outline)
        if not is_active:
            draw_centered(draw, position[0], position[1] + 31, "0", font(12, True), rgb(RED))

    for position, name, is_active in zip(hidden_two, hidden_names, h2_active):
        node_fill = rgb(LIGHT_PURPLE) if is_active else rgb(GREY_NODE)
        node_outline = rgb(PURPLE) if is_active else rgb(GREY_STROKE)
        draw_dot(draw, *position, 27, node_fill, node_outline)
        draw_centered(draw, position[0], position[1] - 9, name, font(13, True), node_outline)
        if not is_active:
            draw_centered(draw, position[0], position[1] + 31, "0", font(12, True), rgb(RED))

    draw_dot(draw, *output, 56, rgb(LIGHT_GREEN), rgb(GREEN))
    draw_centered(draw, output[0], output[1] - 20, "p", font(20, True), rgb(GREEN))
    probability_text = "?" if probability is None else f"{probability:.2f}"
    draw_centered(draw, output[0], output[1] + 8, probability_text, font(18, True), rgb(INK))
    draw_centered(draw, output[0], output[1] + 77, "P(malignant)", font(13), rgb(GREEN))


def draw_data_cards(draw: ImageDraw.ImageDraw, clock: float) -> None:
    """Show a patient row becoming a scaled mini-batch input."""
    cards = (
        (70, 190, 390, 550, "1. Raw patient row", BLUE, LIGHT_BLUE),
        (480, 190, 800, 550, "2. Min-Max scale", PURPLE, LIGHT_PURPLE),
        (890, 190, 1210, 550, "3. Mini-batch", GREEN, LIGHT_GREEN),
    )
    for left, top, right, bottom, title, colour, light in cards:
        rounded_box(draw, (left, top, right, bottom), rgb("#ffffff"), rgb(LIGHT_BORDER), 22, 2)
        rounded_box(draw, (left + 16, top + 16, left + 48, top + 48), rgb(colour), None, 16, 0)
        draw_centered(draw, left + 32, top + 23, title[0], font(15, True), rgb("#ffffff"))
        draw.text((left + 61, top + 21), title[3:], font=font(17, True), fill=rgb(INK))

    features = (("radius", "14.2"), ("texture", "20.1"), ("…", "…"), ("worst area", "645"))
    for index, (name, value) in enumerate(features):
        y = 270 + index * 52
        rounded_box(draw, (96, y, 360, y + 38), rgb("#f8fbff"), rgb("#bfdbfe"), 9, 1)
        draw.text((112, y + 9), name, font=font(14), fill=rgb(MUTED))
        draw.text((302, y + 9), value, font=font(14, True), fill=rgb(BLUE))
    draw_centered(draw, 230, 494, "30 numerical features", font(15, True), rgb(BLUE))

    draw.text((512, 272), "Move each feature into a", font=font(15), fill=rgb(MUTED))
    draw.text((512, 294), "shared 0 to 1 range.", font=font(15), fill=rgb(MUTED))
    for index, value in enumerate((0.28, 0.63, 0.46, 0.81)):
        y = 340 + index * 38
        rounded_box(draw, (512, y, 758, y + 18), rgb("#e9e7ff"), None, 9, 0)
        width = int(246 * value * (0.90 + 0.10 * math.sin(clock * 3 + index)))
        rounded_box(draw, (512, y, 512 + width, y + 18), rgb(PURPLE), None, 9, 0)
        draw.text((765, y - 1), f"{value:.2f}", font=font(13, True), fill=rgb(PURPLE))
    draw_centered(draw, 640, 494, "scaling avoids one feature dominating", font(13), rgb(PURPLE))

    draw.text((920, 272), "Shuffle the training rows", font=font(15), fill=rgb(MUTED))
    draw.text((920, 294), "and take 8 at a time.", font=font(15), fill=rgb(MUTED))
    for row in range(5):
        y = 335 + row * 33
        left = 925 + int(8 * math.sin(clock * 2 + row))
        rounded_box(draw, (left, y, left + 208, y + 25), blend(LIGHT_GREEN, "#ffffff", row * 0.10), rgb("#a7f3d0"), 7, 1)
        for column in range(7):
            x = left + 13 + column * 25
            draw_dot(draw, x, y + 13, 4, rgb(GREEN) if (row + column) % 3 else rgb("#86efac"))
    draw_centered(draw, 1050, 494, "X_batch: 8 rows × 30 values", font(14, True), rgb(GREEN))


def draw_relu_panel(draw: ImageDraw.ImageDraw, clock: float) -> None:
    """Draw the ReLU equation and two example activations."""
    rounded_box(draw, (825, 435, 1205, 574), rgb("#ffffff"), rgb("#ddd6fe"), 18, 2)
    draw.text((852, 455), "ReLU keeps useful positive signals", font=font(15, True), fill=rgb(PURPLE))
    draw.text((852, 484), "ReLU(z) = max(0, z)", font=font(20, True), fill=rgb(INK))
    examples = (("z = -0.8", "0", RED), ("z = +0.7", "+0.7", PURPLE))
    for index, (before, after, colour) in enumerate(examples):
        x = 853 + index * 176
        draw.text((x, 520), before, font=font(14), fill=rgb(MUTED))
        draw_arrow(draw, (x + 75, 541), (x + 120, 541), rgb(colour), 3, 9)
        draw.text((x + 129, 529), after, font=font(15, True), fill=rgb(colour))


def draw_pattern_panel(draw: ImageDraw.ImageDraw, clock: float) -> None:
    """Show the second hidden layer collecting simple signals into patterns."""
    rounded_box(draw, (838, 435, 1210, 574), rgb("#ffffff"), rgb("#ddd6fe"), 18, 2)
    draw.text((864, 454), "Layer 2 combines activations", font=font(16, True), fill=rgb(PURPLE))
    draw.text((864, 480), "into richer patterns.", font=font(15), fill=rgb(MUTED))
    centres = ((902, 529), (1023, 529), (1144, 529))
    labels = ("shape", "texture", "size")
    for index, ((x, y), label) in enumerate(zip(centres, labels)):
        radius = 19 + int(3 * math.sin(clock * 2 + index))
        draw_dot(draw, x, y, radius, rgb(LIGHT_PURPLE), rgb(PURPLE))
        draw_centered(draw, x, 558, label, font(12, True), rgb(PURPLE))


def draw_sigmoid_panel(draw: ImageDraw.ImageDraw, probability: float) -> None:
    """Explain how the one output score becomes a probability."""
    rounded_box(draw, (845, 438, 1210, 574), rgb("#ffffff"), rgb("#a7f3d0"), 18, 2)
    draw.text((870, 455), "Sigmoid converts one score into", font=font(16, True), fill=rgb(GREEN))
    draw.text((870, 479), "a probability from 0 to 1.", font=font(15), fill=rgb(MUTED))
    draw.text((870, 515), "sigmoid(z) = 1 / (1 + e^-z)", font=font(15, True), fill=rgb(INK))
    rounded_box(draw, (870, 541, 1187, 561), rgb("#eaf8f0"), None, 10, 0)
    rounded_box(draw, (870, 541, 870 + int(317 * probability), 561), rgb(GREEN), None, 10, 0)
    draw_centered(draw, 1028, 542, f"P(malignant) = {probability:.2f}", font(14, True), rgb(INK))


def draw_loss_panel(draw: ImageDraw.ImageDraw, pulse: float) -> None:
    """Draw the prediction-versus-label comparison and BCE loss."""
    rounded_box(draw, (115, 190, 1165, 550), rgb("#ffffff"), rgb("#fed7aa"), 24, 2)
    draw_centered(draw, 640, 216, "Measure the error with binary cross-entropy", font(23, True), rgb(ORANGE))
    draw_centered(draw, 640, 248, "The prediction is compared with the known label from the training data.", font(15), rgb(MUTED))

    rounded_box(draw, (205, 290, 515, 450), rgb(LIGHT_GREEN), rgb("#a7f3d0"), 18, 2)
    draw_centered(draw, 360, 315, "MODEL PREDICTION", font(14, True), rgb(GREEN))
    draw_centered(draw, 360, 344, "p = 0.84", font(34, True), rgb(INK))
    draw_centered(draw, 360, 397, "The network is 84% confident", font(14), rgb(MUTED))
    draw_centered(draw, 360, 417, "that this row is malignant.", font(14), rgb(MUTED))

    rounded_box(draw, (765, 290, 1075, 450), rgb(LIGHT_ORANGE), rgb("#fed7aa"), 18, 2)
    draw_centered(draw, 920, 315, "TRUE LABEL", font(14, True), rgb(ORANGE))
    draw_centered(draw, 920, 344, "y = 1", font(34, True), rgb(INK))
    draw_centered(draw, 920, 397, "M (malignant) becomes 1.", font(14), rgb(MUTED))
    draw_centered(draw, 920, 417, "B (benign) becomes 0.", font(14), rgb(MUTED))

    draw_arrow(draw, (528, 370), (750, 370), rgb(ORANGE), 5, 16)
    draw_centered(draw, 640, 340, "compare", font(16, True), rgb(ORANGE))
    draw_centered(draw, 640, 407, "binary cross-entropy", font(16, True), rgb(ORANGE))
    draw_centered(draw, 640, 475, "BCE = -mean[y log(p) + (1-y) log(1-p)]", font(18, True), rgb(INK))
    loss_fill = blend(LIGHT_ORANGE, ORANGE, 0.25 + 0.12 * math.sin(pulse * math.pi * 4) ** 2)
    rounded_box(draw, (360, 500, 920, 522), rgb("#fff7ed"), None, 11, 0)
    rounded_box(draw, (360, 500, 360 + int(560 * 0.32), 522), loss_fill, None, 11, 0)
    draw_centered(draw, 640, 498, "loss: 0.32 — smaller is better", font(14, True), rgb(INK))


def draw_backprop_formula(draw: ImageDraw.ImageDraw) -> None:
    """Put the exact fused output gradient beside the backprop animation."""
    rounded_box(draw, (820, 435, 1210, 574), rgb("#ffffff"), rgb("#fecaca"), 18, 2)
    draw.text((846, 454), "Backpropagation", font=font(18, True), fill=rgb(RED))
    draw.text((846, 486), "output gradient:  dZ = p - y", font=font(17, True), fill=rgb(INK))
    draw.text((846, 520), "Gradients show each weight", font=font(14), fill=rgb(MUTED))
    draw.text((846, 541), "how it contributed to the error.", font=font(14), fill=rgb(MUTED))


def draw_update_scene(draw: ImageDraw.ImageDraw, local: float) -> None:
    """Show a weight update followed by loss/accuracy improving through epochs."""
    rounded_box(draw, (70, 162, 650, 558), rgb("#ffffff"), rgb("#a7f3d0"), 24, 2)
    draw.text((102, 191), "Update every weight and bias", font=font(21, True), fill=rgb(GREEN))
    draw.text((102, 224), "W ← W - learning_rate × dW", font=font(22, True), fill=rgb(INK))
    draw.text((102, 254), "b ← b - learning_rate × db", font=font(19, True), fill=rgb(INK))
    draw.text((102, 287), "The learning rate controls the size of each correction.", font=font(14), fill=rgb(MUTED))

    before_widths = (3, 7, 4, 8, 5)
    after_widths = (6, 4, 7, 5, 8)
    y_positions = (350, 390, 430, 470, 510)
    for index, y in enumerate(y_positions):
        progress = ease((local - 0.4) / 0.8)
        width = round(before_widths[index] + (after_widths[index] - before_widths[index]) * progress)
        draw.line((135, y, 560, y), fill=blend("#bbf7d0", GREEN, 0.35), width=2)
        draw.line((135, y, 440, y), fill=rgb(GREEN), width=width)
        draw_dot(draw, 135, y, 9, rgb(LIGHT_GREEN), rgb(GREEN))
        draw_dot(draw, 440, y, 9, rgb(LIGHT_GREEN), rgb(GREEN))

    rounded_box(draw, (700, 162, 1210, 558), rgb("#ffffff"), rgb("#bfdbfe"), 24, 2)
    draw.text((732, 191), "Repeat: batch → update → epoch", font=font(21, True), fill=rgb(BLUE))
    draw.text((732, 219), "After every epoch, record training metrics.", font=font(14), fill=rgb(MUTED))
    graph_box = (742, 270, 1168, 500)
    draw.line((graph_box[0], graph_box[3], graph_box[2], graph_box[3]), fill=rgb(LIGHT_BORDER), width=2)
    draw.line((graph_box[0], graph_box[1], graph_box[0], graph_box[3]), fill=rgb(LIGHT_BORDER), width=2)
    draw.text((742, 512), "epoch 1", font=font(12), fill=rgb(MUTED))
    draw.text((1110, 512), "epoch 84", font=font(12), fill=rgb(MUTED))

    progress = clamp(local / 7.5)
    points_loss = []
    points_accuracy = []
    for index in range(16):
        fraction = index / 15
        x = graph_box[0] + fraction * (graph_box[2] - graph_box[0])
        loss_y = graph_box[1] + 28 + 155 * (fraction ** 0.62)
        accuracy_y = graph_box[3] - 28 - 145 * (fraction ** 0.65)
        points_loss.append((x, loss_y))
        points_accuracy.append((x, accuracy_y))
    visible = max(2, int(len(points_loss) * progress))
    draw.line(points_loss[:visible], fill=rgb(ORANGE), width=4)
    draw.line(points_accuracy[:visible], fill=rgb(BLUE), width=4)
    if visible:
        draw_dot(draw, *points_loss[visible - 1], 5, rgb(ORANGE))
        draw_dot(draw, *points_accuracy[visible - 1], 5, rgb(BLUE))
    draw.text((760, 286), "loss ↓", font=font(14, True), fill=rgb(ORANGE))
    draw.text((1085, 286), "accuracy ↑", font=font(14, True), fill=rgb(BLUE))


def draw_final_scene(draw: ImageDraw.ImageDraw, clock: float) -> None:
    """Show prediction on a new record and clarify sigmoid versus softmax."""
    rounded_box(draw, (70, 174, 535, 555), rgb("#ffffff"), rgb("#bfdbfe"), 22, 2)
    draw.text((102, 203), "A new patient record", font=font(20, True), fill=rgb(BLUE))
    draw.text((102, 233), "uses the learned weights — no label needed.", font=font(14), fill=rgb(MUTED))
    for index, value in enumerate(("0.22", "0.74", "0.41", "0.09")):
        y = 285 + index * 48
        draw.text((110, y), f"feature {index + 1}", font=font(14), fill=rgb(MUTED))
        rounded_box(draw, (244, y + 3, 465, y + 21), rgb("#e7eff8"), None, 9, 0)
        rounded_box(draw, (244, y + 3, 244 + int(221 * float(value)), y + 21), rgb(BLUE), None, 9, 0)
        draw.text((477, y), value, font=font(14, True), fill=rgb(BLUE))

    draw_arrow(draw, (550, 365), (695, 365), rgb(BLUE), 5, 16)
    rounded_box(draw, (705, 225, 1015, 505), rgb("#ffffff"), rgb("#ddd6fe"), 22, 2)
    draw_centered(draw, 860, 252, "trained 30 → 24 → 24 → 1", font(17, True), rgb(PURPLE))
    for x, y, colour in ((760, 330, BLUE), (825, 330, PURPLE), (890, 330, PURPLE), (955, 330, GREEN)):
        draw_dot(draw, x, y, 24, blend(colour, "#ffffff", 0.80), rgb(colour))
    for left, right in ((760, 825), (825, 890), (890, 955)):
        draw_arrow(draw, (left + 26, 330), (right - 27, 330), rgb("#bfd3ea"), 3, 8)
    draw_centered(draw, 860, 390, "P(malignant) = 0.08", font(25, True), rgb(GREEN))
    rounded_box(draw, (773, 430, 947, 472), rgb(LIGHT_GREEN), None, 18, 0)
    draw_centered(draw, 860, 441, "benign (class 0)", font(17, True), rgb(GREEN))

    rounded_box(draw, (1045, 174, 1210, 555), rgb("#ffffff"), rgb("#fed7aa"), 22, 2)
    draw_centered(draw, 1128, 202, "Output choice", font(17, True), rgb(ORANGE))
    draw.text((1069, 247), "This project", font=font(14, True), fill=rgb(INK))
    draw.text((1069, 271), "sigmoid", font=font(20, True), fill=rgb(GREEN))
    draw.text((1069, 300), "two classes", font=font(13), fill=rgb(MUTED))
    draw.text((1069, 353), "Multiclass", font=font(14, True), fill=rgb(INK))
    draw.text((1069, 377), "softmax", font=font(20, True), fill=rgb(ORANGE))
    draw.text((1069, 406), "one of 3+ classes", font=font(13), fill=rgb(MUTED))
    draw.line((1069, 329, 1185, 329), fill=rgb(LIGHT_BORDER), width=2)


def scene_at(time_seconds: float) -> tuple[int, str, float]:
    """Return the scene number, short title, and time since that scene began."""
    scenes = (
        (6.5, 1, "A neural network learns patterns"),
        (13.5, 2, "Prepare one mini-batch"),
        (23.0, 3, "Forward pass: first ReLU layer"),
        (30.5, 4, "Forward pass: second ReLU layer"),
        (37.5, 5, "Sigmoid produces a probability"),
        (44.5, 6, "Measure the prediction error"),
        (53.0, 7, "Backpropagation sends error backward"),
        (60.5, 8, "Update weights and repeat"),
        (64.0, 9, "Use the trained network"),
    )
    start = 0.0
    for end, number, title in scenes:
        if time_seconds < end:
            return number, title, time_seconds - start
        start = end
    return 9, "Use the trained network", 3.5


def render_frame(time_seconds: float) -> np.ndarray:
    """Render one RGB video frame for a point on the 64-second timeline."""
    image = Image.new("RGB", (WIDTH, HEIGHT), rgb(BACKGROUND))
    draw = ImageDraw.Draw(image)
    scene, title, local = scene_at(time_seconds)
    draw_header(draw, time_seconds, scene, title)

    if scene == 1:
        draw_network(draw, local, forward=True, probability=None)
        draw_centered(draw, 640, 530, "30 inputs  →  24 ReLU  →  24 ReLU  →  1 sigmoid probability", font(19, True), rgb(INK))
        draw_caption(
            draw,
            BLUE,
            "Start with the architecture",
            "Each patient record enters as 30 scaled values. The network learns weighted connections through two hidden layers before producing one binary probability.",
        )
    elif scene == 2:
        draw_data_cards(draw, local)
        draw_caption(
            draw,
            BLUE,
            "Prepare the data before training",
            "Rows are shuffled, numerical features are scaled using the training data, and the model processes a small mini-batch of 8 patient records at a time.",
        )
    elif scene == 3:
        draw_network(
            draw,
            local,
            forward=True,
            h1_active=(True, False, True, False),
            h2_active=(True, True, True, True),
        )
        draw_relu_panel(draw, local)
        draw_caption(
            draw,
            PURPLE,
            "First hidden layer: ReLU activation",
            "Every neuron calculates a weighted sum plus a bias. ReLU turns negative values into zero, allowing active signals to move forward and keeping the network efficient.",
        )
    elif scene == 4:
        draw_network(
            draw,
            local,
            forward=True,
            h1_active=(True, False, True, True),
            h2_active=(True, True, False, True),
        )
        draw_pattern_panel(draw, local)
        draw_caption(
            draw,
            PURPLE,
            "Second hidden layer: combine patterns",
            "The second ReLU layer receives the first layer's active signals and combines them into richer patterns that can help distinguish benign and malignant records.",
        )
    elif scene == 5:
        probability = 0.38 + 0.46 * ease((local % 4.0) / 4.0)
        draw_network(draw, local, forward=True, probability=probability)
        draw_sigmoid_panel(draw, probability)
        draw_caption(
            draw,
            GREEN,
            "Output layer: sigmoid probability",
            "For this binary classifier, sigmoid maps the final score to a number from 0 to 1. Here the network estimates an 84% probability of malignancy.",
        )
    elif scene == 6:
        draw_loss_panel(draw, local)
        draw_caption(
            draw,
            ORANGE,
            "Compare the prediction with the known answer",
            "Binary cross-entropy measures how wrong the probability is. A smaller loss means the prediction is closer to the correct diagnosis for this training row.",
        )
    elif scene == 7:
        draw_network(
            draw,
            local,
            backward=True,
            h1_active=(True, False, True, False),
            h2_active=(True, True, False, True),
            probability=0.84,
        )
        draw_backprop_formula(draw)
        draw_caption(
            draw,
            RED,
            "Backpropagation: send the error backward",
            "The output gradient is prediction minus label. It travels backward through every layer to calculate how each weight and bias should change; inactive ReLU units pass no gradient.",
        )
    elif scene == 8:
        draw_update_scene(draw, local)
        draw_caption(
            draw,
            GREEN,
            "Update, then repeat",
            "The learning rate applies a small correction to weights and biases. The process repeats for every mini-batch and epoch while loss and accuracy are recorded.",
        )
    else:
        draw_final_scene(draw, local)
        draw_caption(
            draw,
            GREEN,
            "Prediction after training",
            "New patient data follows the learned network to a class prediction. This project uses sigmoid for two classes; softmax is the usual output choice for a multiclass problem.",
        )

    return np.asarray(image)


def transcode_h264(source: Path, destination: Path) -> bool:
    """Use VLC to convert the interim MPEG-4 stream to broadly compatible H.264."""
    executable = shutil.which("cvlc")
    if executable is None:
        return False

    destination.unlink(missing_ok=True)
    stream_output = (
        "#transcode{vcodec=h264,vb=1800,acodec=none}:"
        f"standard{{access=file,mux=mp4,dst={destination.resolve()}}}"
    )
    command = [
        executable,
        "-I",
        "dummy",
        "--no-audio",
        "--sout-file-overwrite",
        str(source.resolve()),
        "--sout",
        stream_output,
        "vlc://quit",
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return completed.returncode == 0 and destination.exists() and destination.stat().st_size > 0


def write_video(output: Path, seconds: float, fps: int, transcode: bool) -> Path:
    """Render the requested timeline and return its final video path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(seconds * fps)
    temporary = output.with_name(f"{output.stem}.intermediate.mp4")
    temporary.unlink(missing_ok=True)
    output.unlink(missing_ok=True)

    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (WIDTH, HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 writer.")

    try:
        for frame_index in range(frame_count):
            time_seconds = frame_index / fps
            rgb_frame = render_frame(time_seconds)
            writer.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            if frame_index % (fps * 4) == 0 or frame_index == frame_count - 1:
                print(f"Rendered {frame_index + 1:4d}/{frame_count} frames", flush=True)
    finally:
        writer.release()

    if transcode and transcode_h264(temporary, output):
        temporary.unlink(missing_ok=True)
        return output

    temporary.replace(output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the colour-coded MLP training walkthrough.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("images/mlp-training-walkthrough.mp4"),
        help="Destination MP4 path (default: images/mlp-training-walkthrough.mp4).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help=f"Video duration in seconds (default: {DEFAULT_SECONDS}).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Frames per second (default: {DEFAULT_FPS}).",
    )
    parser.add_argument(
        "--skip-transcode",
        action="store_true",
        help="Keep OpenCV's MP4 stream instead of attempting H.264 transcoding through VLC.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seconds <= 0 or args.fps <= 0:
        raise ValueError("--seconds and --fps must both be greater than zero.")

    output = write_video(args.output, args.seconds, args.fps, not args.skip_transcode)
    print(f"Saved {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
