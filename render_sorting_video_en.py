# /// script
# requires-python = ">=3.11"
# dependencies = ["av==19.0.2", "pillow==13.0.0"]
# ///
"""Render the English presentation from the original sorting recording.

Run: uv run render_sorting_video_en.py
Only presentation pixels change. Every decoded source frame, its original PTS,
and its display duration are retained, including approval waits and final dwell.
The three scene rectangles were measured from this recording; sky probes choose
between them per frame. The original in-scene status box is replaced, not
inpainted. The side panel is a static reference, never a live progress indicator.
"""

import argparse
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
BG = "#101722"
PANEL = "#182432"
WHITE = "#f1f5f9"
MUTED = "#bac8d7"
MINT = "#8ddac7"
SIZE = (1280, 960)
SCENE = (24, 132, 924, 806)


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def relative(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def sky(pixel):
    red, green, blue = pixel
    return 60 < red < 160 and green > red + 5 and blue > red + 15


def dark(pixel):
    return pixel[0] < 65 and pixel[1] < 85 and pixel[2] < 95


def geometry(image):
    pixels = image.load()
    if sky(pixels[1000, 200]) and sky(pixels[1350, 200]):
        return (50, 83, 1431, 713), True
    # Require the header-to-sky boundary, not just sky inside the viewport:
    # after scrolling, row 177 remains sky but is no longer the scene's top.
    for top, bottom in ((66, 570), (177, 680)):
        above_is_header = all(not sky(pixels[x, top - 2]) for x in (30, 650))
        below_is_scene = all(sky(pixels[x, y]) for x in (30, 650) for y in (top, top + 5))
        if above_is_header and below_is_scene:
            return (24, top, 696, bottom), False
    raise ValueError("Source frame has an unrecognized scene layout")


def status_box(image, rect, enlarged):
    # Sample the solid padding above both original status lines, not their text.
    left = 78 if enlarged else 38
    top = rect[1] + (16 if enlarged else 13)
    scan_y = rect[1] + 20
    pixels = image.load()
    if not dark(pixels[left + 10, scan_y]):
        return None
    for right in range(left + 10, 700 if enlarged else 400):
        if not dark(pixels[right, scan_y]):
            # Two pixels cover the compressed edge of the original UI box.
            return (left - 2, top - 2, right + 2, top + (71 if enlarged else 57))
    raise ValueError("Cannot locate the original status-box edge")


def make_template(font, evidence):
    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), "GPT-guided six-object sorting", font=font(32, True), fill=WHITE)
    draw.text((24, 73), "One approved GPT proposal | Local contact-feedback execution", font=font(20), fill=MUTED)
    draw.rectangle(SCENE, fill="#0b111a")
    draw.rounded_rectangle((948, 132, 1256, 874), radius=12, fill=PANEL)
    x = 966

    def lines(y, text, size=20, color=MUTED, bold=False, spacing=28):
        for line in text:
            draw.text((x, y), line, font=font(size, bold), fill=color)
            y += spacing

    lines(151, ["HOW THIS RUN WORKED"], 18, MINT, True)
    lines(185, ["One actual GPT proposal", "was explicitly approved.", "The local controller then", "executed its full queue.", "No per-object GPT replans."])
    draw.line((966, 340, 1238, 340), fill="#344659")
    lines(359, ["APPROVED QUEUE"], 20, WHITE, True)
    lines(391, ["Static reference, not progress"], 18)
    colors = {"yellow": "#ead570", "blue": "#8cb8f3", "red": "#f29d97"}
    steps = evidence["proposal_state"]["planner"]["decision"]["steps"]
    for index, step in enumerate(steps, 1):
        label = step["object_id"].replace("_", " ")
        lines(429 + (index - 1) * 35, [f"{index}.  {label.capitalize()}  >  {step['destination']}"], 20, colors[label.split()[0]])
    lines(653, ["Yellow first; all bowls", "before cups. Then home."], 20)
    draw.line((966, 720, 1238, 720), fill="#344659")
    lines(739, ["RECORDED RUN RESULT"], 18, MINT, True)
    final = evidence["final_state"]["sorting"]
    lines(772, [f"{final['sorted_count']}/{final['object_count']} sorted; empty hand.", "Home reached; 5 s hold.", "One run, not a success rate."], 19)
    draw.text((24, 829), "Requested mapping: yellow > A   |   blue > B   |   red > C", font=font(21, True), fill=WHITE)
    draw.text((24, 864), "Zone IDs define destinations; table patch colors are unchanged.", font=font(19), fill=MUTED)
    draw.line((24, 908, 1256, 908), fill="#344659")
    draw.text((24, 925), "English presentation of the original recording | Full timing, waits and final dwell retained", font=font(19), fill=MUTED)
    return image


def packet_facts(path, presentation_durations=None):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        samples = sorted((p.pts, p.duration) for p in container.demux(stream) if p.pts is not None)
        end = max(pts + duration for pts, duration in samples)
        # MP4 packet durations follow decode order. With source B-frames they
        # need not equal display intervals; adjacent sorted PTS define those.
        presentation = [
            (pts, (samples[index + 1][0] if index + 1 < len(samples) else end) - pts)
            for index, (pts, _) in enumerate(samples)
        ]
        if presentation_durations is not None:
            presentation_durations.update(presentation)
        return {
            "width": stream.width,
            "height": stream.height,
            "codec": stream.codec_context.name,
            "pixel_format": stream.codec_context.format.name,
            "time_base": str(stream.time_base),
            "stream_duration_ticks": stream.duration,
            "container_duration_s": container.duration / av.time_base,
            "declared_frames": stream.frames,
            "encoded_samples": len(samples),
            "average_rate": str(stream.average_rate),
            "first_pts": samples[0][0],
            "last_pts": samples[-1][0],
            "last_duration_ticks": samples[-1][1],
            "presentation_end_ticks": max(pts + duration for pts, duration in samples),
            "sample_durations_ticks": dict(Counter(duration for _, duration in samples)),
            "pts_and_durations_sha256": hashlib.sha256(json.dumps(samples, separators=(",", ":")).encode()).hexdigest(),
            "presentation_pts_sha256": hashlib.sha256(json.dumps([pts for pts, _ in samples], separators=(",", ":")).encode()).hexdigest(),
            "presentation_pts_and_intervals_sha256": hashlib.sha256(json.dumps(presentation, separators=(",", ":")).encode()).hexdigest(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "media/gpt-sorting-contact-feedback-demo.mp4")
    parser.add_argument("--evidence", type=Path, default=ROOT / "results/gpt_sorting_contact_feedback_demo.json")
    parser.add_argument("--output", type=Path, default=ROOT / "media/gpt-sorting-contact-feedback-demo-en.mp4")
    parser.add_argument("--metadata", type=Path, default=ROOT / "results/sorting_video_english.json")
    parser.add_argument("--font-dir", type=Path, default=Path("/System/Library/Fonts/Supplemental"))
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("The original recording must not be overwritten")
    evidence = json.loads(args.evidence.read_text())
    font_cache = {}

    def font(size, bold=False):
        key = (size, bold)
        if key not in font_cache:
            font_cache[key] = ImageFont.truetype(str(args.font_dir / ("Arial Bold.ttf" if bold else "Arial.ttf")), size)
        return font_cache[key]

    template = make_template(font, evidence)
    intervals = []
    masks = Counter()
    samples = []
    durations = {}
    source_facts = packet_facts(args.source, durations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(args.source)) as source:
        incoming = source.streams.video[0]
        if (incoming.width, incoming.height) != (1440, 1080):
            raise ValueError("This renderer requires the original 1440x1080 recording")
        time_base = incoming.time_base
        with av.open(str(args.output), "w", options={"movflags": "+faststart", "video_track_timescale": str(time_base.denominator)}) as output:
            stream = output.add_stream("libx264", rate=6)
            stream.width, stream.height = SIZE
            stream.pix_fmt = "yuv420p"
            stream.time_base = time_base
            stream.codec_context.time_base = time_base
            stream.options = {"preset": "slow", "crf": "24", "maxrate": "140k", "bufsize": "280k", "bf": "0", "g": "120"}

            def mux(packet):
                source_pts = int(Fraction(packet.pts) * packet.time_base / time_base)
                packet.duration = int(Fraction(durations[source_pts]) * time_base / packet.time_base)
                output.mux(packet)

            for index, original in enumerate(source.decode(incoming)):
                image = original.to_image()
                rect, enlarged = geometry(image)
                box = status_box(image, rect, enlarged)
                if not intervals or intervals[-1]["source_crop_xyxy"] != list(rect):
                    intervals.append({"first_frame": index, "first_pts": original.pts, "first_time_s": float(original.time), "source_crop_xyxy": list(rect)})
                intervals[-1]["last_frame"] = index
                intervals[-1]["last_pts"] = original.pts
                if box:
                    masks[box] += 1
                    draw = ImageDraw.Draw(image)
                    draw.rectangle(box, fill=BG)
                    overlay_font = font(28 if enlarged else 17, True)
                    draw.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), "Recorded physics", font=overlay_font, fill=MINT, anchor="mm")
                scene = image.crop(rect)
                # One uniform scale factor preserves the recorded scene geometry.
                scale = min(900 / scene.width, 674 / scene.height)
                width, height = round(scene.width * scale), round(scene.height * scale)
                scene = scene.resize((width, height), Image.Resampling.LANCZOS)
                canvas = template.copy()
                canvas.paste(scene, (24 + (900 - width) // 2, 132 + (674 - height) // 2))
                frame = av.VideoFrame.from_image(canvas)
                frame.pts = original.pts
                frame.time_base = time_base
                frame.duration = durations[original.pts]
                samples.append((original.pts, durations[original.pts]))
                for packet in stream.encode(frame):
                    mux(packet)
            for packet in stream.encode():
                mux(packet)
    size = args.output.stat().st_size
    if size >= 9_900_000:
        raise RuntimeError(f"Encoded output exceeds the attachment limit: {size} bytes")
    provenance = {
        "kind": "english_presentation_edit_of_existing_recording",
        "renderer": relative(Path(__file__)),
        "dependencies": {"av": av.__version__, "pillow": Image.__version__},
        "source": {"path": relative(args.source), "sha256": sha256(args.source), "bytes": args.source.stat().st_size, **source_facts},
        "evidence": {"path": relative(args.evidence), "sha256": sha256(args.evidence), "native_capture_metadata": evidence["recording"], "native_capture_note": "690 native captures are not the encoded frame count; this renderer processes all 3032 decoded frames."},
        "output": {"path": relative(args.output), "sha256": sha256(args.output), "bytes": size, **packet_facts(args.output), "audio": False, "faststart_requested": True},
        "presentation": {
            "canvas_size": SIZE,
            "scene_destination_box_xyxy": SCENE,
            "crop_coordinate_convention": "Source pixels, left/top inclusive and right/bottom exclusive; entire physical viewport, excluding dashboard chrome.",
            "layout_selection": "Per-frame probes require a non-sky header immediately above the sky boundary at both scene edges. This distinguishes the y177 initial viewport from y66 after scrolling, even though y177 remains sky inside the scrolled viewport. Independent wide-scene probes select the enlarged layout.",
            "layout_intervals": intervals,
            "status_box_method": "Detect original dark box right edge in solid top padding. Opaque English label replaces both original lines, including two-pixel compressed edges. No inpainting or fabricated physical content.",
            "status_box_rectangles": [{"source_xyxy_inclusive": list(box), "frame_count": count} for box, count in sorted(masks.items())],
            "scaling": "Isotropic Lanczos resizing and centered letterboxing; source camera/layout changes remain.",
            "panel_semantics": "Static approved-queue reference and explicitly historical recorded-run result. No invented live progress or phase alignment.",
            "requested_mapping": evidence["requested_mapping"],
            "requested_priority": evidence["requested_priority"],
            "approved_queue": evidence["proposal_state"]["planner"]["decision"]["steps"],
            "gpt_calls_in_evidence": evidence["proposal_state"]["planner"]["calls"],
            "execution": "One real GPT proposal was explicitly approved; the local bounded contact-feedback controller executed the complete queue. Not online per-object GPT replanning.",
            "fonts": {"regular_sha256": sha256(args.font_dir / "Arial.ttf"), "bold_sha256": sha256(args.font_dir / "Arial Bold.ttf")},
        },
        "timing": {
            "decoded_source_frames_processed": len(samples),
            "source_time_base": str(time_base),
            "processed_pts_and_display_intervals_sha256": hashlib.sha256(json.dumps(samples, separators=(",", ":")).encode()).hexdigest(),
            "method": "One output frame per decoded source frame. Original PTS and adjacent-PTS display intervals assigned to encoder/muxer in source time base; no trimming, acceleration, frame duplication or simulation replay.",
            "packet_duration_note": "Source B-frame packet durations are decode-order sample durations, not display intervals. The output has no B-frames, so packet-duration associations differ at two source gaps while PTS and actual display intervals are preserved. Compare presentation_pts_and_intervals_sha256, not the packet-duration fingerprint.",
            "rounding": "Source 1/12288-second time base is requested for both codec and MP4 track, avoiding a new timestamp quantization grid. MP4 container duration is independently reported in microseconds.",
            "wall_vs_simulation": "Video duration includes approval wait and post-success dwell; it is not simulation execution time.",
        },
        "encoding": {"codec": "libx264", "preset": "slow", "crf": 24, "maxrate": "140k", "bufsize": "280k", "b_frames": 0, "keyframe_interval_frames": 120},
        "verification_scope": "Renderer provenance only. Final playback, English-language and visual completeness QA are performed separately by the integrating agent.",
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": relative(args.output), "bytes": size, "frames_processed": len(samples), "metadata": relative(args.metadata)}))


if __name__ == "__main__":
    main()
