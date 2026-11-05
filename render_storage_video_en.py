# /// script
# requires-python = ">=3.10"
# dependencies = ["av==19.0.2", "pillow==13.0.0"]
# ///
"""Render the English presentation directly from the original storage recording.

Run with: uv run render_storage_video_en.py
No simulation, model request, native-capture reconstruction, or phase interpolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

ROOT = Path(__file__).resolve().parent
BACKGROUND = "#101722"
PANEL = "#182330"
WHITE = "#eef3f8"
MUTED = "#adbdcd"
MINT = "#8ddac7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def default_font(bold: bool) -> Path:
    names = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for name in names:
        if Path(name).is_file():
            return Path(name)
    raise FileNotFoundError("Supply --font and --bold-font paths to installed TrueType fonts")


def scene_geometry(image: Image.Image) -> tuple[tuple[int, int, int, int], bool]:
    """Locate the sky boundary independently on every original frame.

    The three probes are above the workcell and away from the status overlay.
    Original layout has two fully visible viewport positions, not a moving camera.
    Reject unknown layouts rather than silently cropping a physical event.
    """
    pixels = image.load()
    for top in range(250):
        colors = [pixels[x, top] for x in (250, 450, 650)]
        if all(r > 60 and g > 70 and b > 80 and b - r >= 15 and g - r >= 4
               for r, g, b in colors):
            break
    else:
        raise ValueError("Cannot locate the original scene's sky boundary")
    if top not in (177, 66):
        raise ValueError(f"Unexpected scene top {top}; inspect the source before rendering")
    overlay = max(pixels[43, top + 25]) < 65
    # Fractional CSS sizing rasterizes to 502 rows before scrolling, 503 after.
    bottom = 679 if top == 177 else 569
    return (24, top, 696, bottom), overlay


def presentation(font: Path, bold_font: Path) -> Image.Image:
    image = Image.new("RGB", (1280, 960), BACKGROUND)
    draw = ImageDraw.Draw(image)

    def text(x: int, y: int, value: str, size: int = 20,
             color: str = WHITE, bold: bool = False) -> None:
        face = ImageFont.truetype(str(bold_font if bold else font), size)
        draw.text((x, y), value, font=face, fill=color)

    text(24, 18, "GPT PROPOSAL + FIXED EXECUTION", 18, MINT, True)
    text(24, 46, "Bowl storage | One approved plan", 32, WHITE, True)
    text(24, 92, "Original MuJoCo footage: storage, contact-driven drawer closure, and return home", 20, MUTED)
    draw.rectangle((24, 132, 923, 805), fill="#090e15")
    draw.rounded_rectangle((948, 132, 1256, 806), radius=12, fill=PANEL)
    text(968, 154, "CONTROL MODEL", 18, MINT, True)
    for index, line in enumerate((
        "One subscription proposal",
        "approved before execution.",
        "Then a fixed complete",
        "storage sequence runs.",
        "Not per-skill GPT control.",
    )):
        text(968, 188 + index * 29, line, 20)
    draw.line((968, 347, 1236, 347), fill="#354454", width=1)
    text(968, 371, "SEQUENCE REFERENCE", 18, MINT, True)
    text(968, 402, "Static guide, not live status", 18, MUTED)
    for index, line in enumerate((
        "01  Prepare / approach",
        "02  Descend / close",
        "03  Lift / hold",
        "04  Transfer / place / release",
        "05  Withdraw gripper",
        "06  Push drawer closed",
        "07  Withdraw / return home",
    )):
        text(968, 447 + index * 42, line, 18)
    text(968, 759, "Original timing retained", 18, MUTED)
    text(24, 831, "Recorded run result: bowl stored, drawer closed, robot home.", 22, WHITE, True)
    text(24, 869, "Original recording | English presentation | No new simulation or model request", 20, MUTED)
    text(24, 908, "Full footage retained, including the approval wait and final dwell.", 18, MUTED)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "media/grasp-demo.mp4")
    parser.add_argument("--evidence", type=Path, default=ROOT / "results/subscription_storage.json")
    parser.add_argument("--output", type=Path, default=ROOT / "media/grasp-demo-en.mp4")
    parser.add_argument("--metadata", type=Path, default=ROOT / "results/storage_video_english.json")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--bold-font", type=Path)
    parser.add_argument("--crf", type=int, default=22)
    parser.add_argument("--max-rate", default="760k")
    parser.add_argument("--buffer-size", default="1520k")
    args = parser.parse_args()
    if args.output.resolve() in (args.source.resolve(), args.evidence.resolve()):
        parser.error("The output must not overwrite an original source")
    font = args.font or default_font(False)
    bold_font = args.bold_font or default_font(True)
    evidence = json.loads(args.evidence.read_text())
    terminal = evidence["terminal"]["skill"]
    if terminal["status"] != "succeeded" or not all(
        terminal["storage"][name]
        for name in ("bowl_inside_drawer", "drawer_closed", "home_reached")
    ):
        raise ValueError("The recorded-result caption does not match the evidence")
    template = presentation(font, bold_font)
    overlay_title_font = ImageFont.truetype(str(bold_font), 14)
    overlay_body_font = ImageFont.truetype(str(font), 13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    layouts: list[dict] = []
    timestamp_digest = hashlib.sha256()
    count = 0
    first_pts = last_pts = last_duration = None
    durations: dict[int, int] = {}
    packet_count = 0
    with av.open(str(args.source)) as source:
        video = source.streams.video[0]
        if (video.width, video.height) != (1440, 1050):
            raise ValueError("This renderer requires the original 1440x1050 storage recording")
        time_base = video.time_base
        source_facts = {
            "path": portable_path(args.source), "sha256": sha256(args.source),
            "width": video.width, "height": video.height,
            "time_base": str(time_base), "average_rate": str(video.average_rate),
            "declared_frames": video.frames, "stream_duration_ticks": video.duration,
            "container_duration_s": source.duration / av.time_base,
            "audio_streams": len(source.streams.audio),
        }
        encoder_options = {
            "preset": "slow", "crf": str(args.crf), "maxrate": args.max_rate,
            "bufsize": args.buffer_size, "profile": "high",
        }
        with av.open(str(args.output), "w", options={"movflags": "+faststart"}) as output:
            stream = output.add_stream("libx264", rate=video.average_rate, options=encoder_options)
            stream.width, stream.height = 1280, 960
            stream.pix_fmt = "yuv420p"
            stream.time_base = time_base
            stream.codec_context.time_base = time_base

            def mux(packet: av.Packet) -> None:
                nonlocal packet_count
                original_pts = int(Fraction(packet.pts) * packet.time_base / time_base)
                packet.duration = int(Fraction(durations.pop(original_pts)) * time_base / packet.time_base)
                output.mux(packet)
                packet_count += 1

            for frame in source.decode(video):
                if frame.pts is None or (last_pts is not None and frame.pts <= last_pts):
                    raise ValueError("Source timestamps must exist and increase strictly")
                if not frame.duration:
                    raise ValueError("Original frame duration is required; do not invent timing")
                if first_pts is None:
                    first_pts = frame.pts
                last_pts, last_duration = frame.pts, frame.duration
                timestamp_digest.update(f"{frame.pts}:{frame.duration}\n".encode("ascii"))
                durations[frame.pts] = frame.duration
                original = frame.to_image()
                crop, has_overlay = scene_geometry(original)
                if not layouts or layouts[-1]["crop_xyxy"] != list(crop) or layouts[-1]["has_overlay"] != has_overlay:
                    layouts.append({
                        "first_frame": count, "first_pts": frame.pts,
                        "start_s": float(frame.pts * time_base), "crop_xyxy": list(crop),
                        "has_overlay": has_overlay,
                    })
                if has_overlay:
                    # Replace only text inside the existing opaque status box.
                    # No hidden physical pixels are synthesized or inpainted.
                    top = crop[1]
                    draw = ImageDraw.Draw(original)
                    draw.rectangle((41, top + 19, 177, top + 61), fill="#142a30")
                    draw.text((44, top + 21), "FIXED SEQUENCE", font=overlay_title_font, fill=MINT)
                    draw.text((44, top + 42), "Original footage", font=overlay_body_font, fill=WHITE)
                scene = original.crop(crop)
                scaled_height = round(scene.height * 900 / scene.width)
                scene = scene.resize((900, scaled_height), Image.Resampling.LANCZOS)
                rendered = template.copy()
                rendered.paste(scene, (24, 132 + (674 - scaled_height) // 2))
                encoded_frame = av.VideoFrame.from_image(rendered)
                encoded_frame.pts = frame.pts
                encoded_frame.time_base = time_base
                for packet in stream.encode(encoded_frame):
                    mux(packet)
                count += 1
            for packet in stream.encode():
                mux(packet)
    # Container facts are required provenance, not a playback/language QA claim.
    with av.open(str(args.output)) as completed:
        video = completed.streams.video[0]
        output_facts = {
            "path": portable_path(args.output), "sha256": sha256(args.output),
            "size_bytes": args.output.stat().st_size, "width": video.width,
            "height": video.height, "codec": video.codec_context.name,
            "pixel_format": video.codec_context.format.name,
            "time_base": str(video.time_base), "average_rate": str(video.average_rate),
            "declared_frames": video.frames, "encoded_packets": packet_count,
            "stream_duration_ticks": video.duration,
            "container_duration_s": completed.duration / av.time_base,
            "audio_streams": len(completed.streams.audio), "faststart_requested": True,
        }
    source_facts.update({
        "decoded_frames": count, "first_pts": first_pts, "last_pts": last_pts,
        "last_duration_ticks": last_duration,
        "frame_pts_duration_sha256": timestamp_digest.hexdigest(),
        "frame_pts_duration_hash_format": "ASCII rows: pts:duration\\n, in presentation order",
    })
    provenance = {
        "kind": "english_presentation_of_original_storage_recording",
        "source": source_facts,
        "evidence": {
            "path": portable_path(args.evidence), "sha256": sha256(args.evidence),
            "native_recording": evidence["recording"],
            "control_model": "One approved subscription proposal followed by a fixed complete storage sequence; not per-skill GPT closed-loop control",
            "recorded_result": {"status": terminal["status"], "storage": terminal["storage"]},
        },
        "geometry": {
            "coordinate_convention": "Crop xyxy is half-open; overlay rectangle xyxy is inclusive",
            "method": "Every decoded frame: find first blue-gray sky row jointly at x=250,450,650; reject any top other than 177 or 66",
            "layout_intervals": layouts,
            "destination_scene_box_xyxy": [24, 132, 924, 806],
            "scaling": "Isotropic fit to width 900, nearest-integer height, vertically centered; no workcell cropping",
            "overlay_presence": "Source pixel x=43,y=scene_top+25 has all RGB channels below 65",
            "original_overlay_box": "Approximately x=38..179, y=scene_top+12..67; rounded original opaque box retained",
            "replacement_rectangle": {"x": [41, 177], "y_relative_to_scene_top": [19, 61]},
            "replacement_text": ["FIXED SEQUENCE", "Original footage"],
            "hidden_pixels": "No reconstruction or inpainting; only original opaque overlay text region is replaced",
        },
        "timing": {
            "policy": "One output frame per original decoded frame; original integer PTS and durations supplied to encoder/muxer in original time base",
            "cuts": [], "speed_factor": 1,
            "native_capture_vs_encoded_source": "Evidence reports 1701 native captures over 88.72879910469055 s; actual original MP4 contains 1779 frames at 20 fps over 88.95 s. This presentation preserves that encoded MP4, not unrecoverable native capture timestamps.",
            "phase_display": "Static sequence reference only, never a live phase indicator",
            "alignment_ambiguity": "Evidence phases and drawer_motion use simulation time. execute_video_offset_s alone does not establish exact simulation-to-encoded-frame phase alignment; no phase timing is guessed.",
            "rounding": "Original and output container time bases are recorded above; integer source PTS are passed without speed conversion. Scene height rounding affects only spatial scale, not timestamps.",
        },
        "render": {
            "script": portable_path(Path(__file__)), "script_sha256": sha256(Path(__file__)),
            "pyav_version": av.__version__, "pillow_version": pillow_version,
            "font": str(font), "font_sha256": sha256(font),
            "bold_font": str(bold_font), "bold_font_sha256": sha256(bold_font),
            "encoder": "libx264", "encoder_options": encoder_options,
            "canvas": [1280, 960], "new_simulation": False, "new_model_request": False,
        },
        "output": output_facts,
        "qa": "Final playback, language, and timing QA belong to the integrating parent; no such verification is claimed here",
    }
    args.metadata.write_text(json.dumps(provenance, indent=2, ensure_ascii=True) + "\n")
    if output_facts["size_bytes"] >= 9_900_000:
        raise ValueError("Output exceeds 9.9 MB; lower --max-rate and render again from the original source")
    print(json.dumps(output_facts, indent=2))


if __name__ == "__main__":
    main()
