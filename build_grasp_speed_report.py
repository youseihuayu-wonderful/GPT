# /// script
# requires-python = ">=3.12"
# dependencies = ["av==19.0.2", "pillow==13.0.0", "reportlab==5.1.0"]
# ///
"""Build the English handout from the archived verified Level 1 batch.

This reads existing evidence and video frames. It does not run a simulation,
train a model, request GPT, or alter the frozen experimental artifacts.
Run: uv run --script build_grasp_speed_report.py
"""
from io import BytesIO
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

import av
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

ROOT = Path(__file__).resolve().parent
BATCH = ROOT / "results/grasp_level1_verified_20261104"
TITLE = "Active Learning for Adaptive Grasp Speed Selection"
OUTPUT = ROOT / f"{TITLE}.pdf"
REPOSITORY = "https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms"
WIDTH = A4[0] - 96
INK = colors.HexColor("#172c3c")
BLUE = colors.HexColor("#176587")
MUTED = colors.HexColor("#536574")
LINE = colors.HexColor("#ced8df")
LIGHT = colors.HexColor("#f1f6f8")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def video_still(relative_path, seconds, scene_box):
    """Extract the first existing presentation frame at/after the requested time."""
    with av.open(str(ROOT / relative_path)) as container:
        stream = container.streams.video[0]
        container.seek(int(seconds / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time >= seconds:
                image = frame.to_image().crop(scene_box)
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                buffer.seek(0)
                return buffer, image.size, float(frame.time)
    raise RuntimeError(f"No frame at {seconds}s in {relative_path}")


def build():
    summary = load_json(BATCH / "summary.json")
    protocol = load_json(BATCH / "protocol.json")
    metadata = load_json(BATCH / "model_metadata.json")
    environment = load_json(BATCH / "environment.json")
    audit = load_json(BATCH / "posthoc_audit.json")
    predictions = load_json(BATCH / "recommendations.json")["recommendations"]
    outcomes = load_json(BATCH / "recommendation_outcomes.json")["rows"]
    storage = load_json(ROOT / "results/storage_video_english.json")
    sorting = load_json(ROOT / "results/sorting_video_english.json")
    if not summary["level1_completed"] or not all(summary["level1_checks"].values()):
        raise RuntimeError("The archived batch does not satisfy its Level 1 checks")
    if metadata["sha256"] != summary["model_sha256"]:
        raise RuntimeError("Model metadata and study summary disagree")
    if audit["condition_groups_with_differing_speed_labels"]:
        raise RuntimeError("The frozen batch no longer supports this report's interpretation")

    rl_config.invariant = 1
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=22,
                              leading=27, textColor=INK, spaceAfter=12, alignment=TA_LEFT))
    styles.add(ParagraphStyle("SectionTitle", fontName="Helvetica-Bold", fontSize=17,
                              leading=22, textColor=INK, spaceAfter=12))
    styles.add(ParagraphStyle("Subheading", fontName="Helvetica-Bold", fontSize=11,
                              leading=15, textColor=BLUE, spaceBefore=11, spaceAfter=5))
    styles.add(ParagraphStyle("BodyReport", fontName="Helvetica", fontSize=10,
                              leading=14, textColor=INK, spaceAfter=7))
    styles.add(ParagraphStyle("SmallReport", parent=styles["BodyReport"], fontSize=8.5,
                              leading=11.5, textColor=MUTED, spaceAfter=5))
    styles.add(ParagraphStyle("CellReport", parent=styles["BodyReport"], fontSize=9,
                              leading=12, spaceAfter=0))
    styles.add(ParagraphStyle("CodeReport", fontName="Courier", fontSize=8,
                              leading=11, textColor=INK, spaceAfter=8))
    story = []

    def p(text, style="BodyReport"):
        return Paragraph(text, styles[style])

    def add(text, style="BodyReport"):
        story.append(p(text, style))

    def heading(text):
        add(text, "Subheading")

    def table(rows, widths):
        cells = [[p(escape(str(cell)), "CellReport") for cell in row] for row in rows]
        result = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
            ("TEXTCOLOR", (0, 0), (-1, 0), INK),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, LINE),
            ("LINEBELOW", (0, 1), (-1, -1), 0.35, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.extend([result, Spacer(1, 8)])

    def note(text):
        box = Table([[p(text)]], colWidths=[WIDTH])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.6, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 11),
            ("RIGHTPADDING", (0, 0), (-1, -1), 11),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.extend([box, Spacer(1, 8)])

    def link(relative, label):
        return f'<a href="{REPOSITORY}/blob/main/{relative}" color="#176587">{escape(label)}</a>'

    # Page 1: feasibility, observed data, and measurement scope.
    add("VERIFIED LEVEL 1 STUDY | 4 NOVEMBER 2026", "SmallReport")
    add(TITLE, "ReportTitle")
    add("End-to-end feasibility results and research limitations", "Subheading")
    note("<b>Engineering feasibility demonstrated; research advantage not yet demonstrated.</b> "
         "The system collected physical success/failure labels, trained a classifier, saved "
         "three-speed predictions, and executed its recommendations in MuJoCo. This is not "
         "an active-learning comparison or evidence that adaptive speed selection beats a fixed fast setting.")
    heading("Research question")
    add("Can active learning use fewer grasp trials than random sampling to learn an equally "
        "effective or better speed-selection policy? Level 1 checks whether the physical data "
        "and execution pipeline works before that hypothesis is tested.")
    heading("Environment and task")
    add(f"MuJoCo {environment['packages']['mujoco']} with a Franka Panda arm and gripper. "
        "The existing black-bowl/drawer workcell supplies two concrete targets: a black bowl "
        "and a wine bottle. Each trial performs approach, descent, closure, lift, and hold. "
        "<b>Drawer placement/closure and six-object sorting are not part of this classifier experiment.</b>")
    heading("Collected labels")
    count_rows = [["Split", "Attempts", "Success", "Failure", "Invalid setup"]]
    for key, label in [("training", "Training"), ("evaluation", "Evaluation")]:
        data = summary[key]
        count_rows.append([label, data["attempted"], data["successes"], data["failures"], data["invalid_initial_states"]])
    table(count_rows, [WIDTH * x for x in (.25, .18, .18, .18, .21)])
    add("Invalid initial states have no label. Expected controller failures remain negative "
        "examples; internal/numerical errors abort instead of becoming fabricated failures.", "SmallReport")
    heading("Observed collection cost")
    table([
        ["Measurement", "Observed result"],
        ["Valid-label trials", f"{audit['valid_labels']} trials; {audit['labelled_trial_mean_wall_s']:.3f} wall s/trial"],
        ["Complete study function", f"{summary['elapsed_wall_s']:.2f} wall s"],
        ["Classifier training", f"{summary['training_wall_s']:.3f} wall s"],
    ], [WIDTH * .49, WIDTH * .51])
    add("Measured on Apple M5 Pro / 48 GB, headless, with a separate live preview. Per-trial "
        "timing includes construction, settling, execution, hold observation, trace saving and "
        "cleanup; it excludes the final row-JSON write. These are wall-clock costs, not robot "
        "task durations. A larger collection campaign has not yet been measured.", "SmallReport")

    # Page 2: features, leakage controls, and honest probability interpretation.
    story.append(PageBreak())
    add("Features, model, and physical verification", "SectionTitle")
    table([
        ["Input", "Definition in this pilot"],
        ["Object category", "Two one-hot columns: bowl and bottle"],
        ["Initial position", "Prescribed reset X/Y offsets, each within +/-0.08 m"],
        ["Initial orientation", "Prescribed reset yaw within +/-0.20 rad"],
        ["Descent speed", ", ".join(f"{s:.2f}" for s in protocol["speeds_m_s"]) + " m/s trajectory settings"],
    ], [WIDTH * .30, WIDTH * .70])
    add("The six columns are category (2), reset X/Y (2), yaw (1), and speed (1). Actual "
        "settled poses are archived separately, not used as these input features. There is "
        "no visual recognition. Speed is a trajectory parameter, not a measured constant TCP speed.", "SmallReport")
    add("Each object's geometry, mass, friction, and grasp rule stay fixed. Initial opening "
        "is 0.08 m for the bowl and 0.04 m for the bottle; closure-trigger height is 0.")
    heading("Split and execution order")
    add(f"Seed {protocol['seed']}: {len(protocol['training_cases'])} training conditions and "
        f"{len(protocol['evaluation_cases'])} held-out initial conditions, each tried at three speeds. "
        "Both splits use the same two object assets: this is not unseen-object generalization. "
        "The three outcomes within a condition are correlated.")
    add("Predictions and recommendations were saved before any evaluation execution. The "
        "recommended speed was executed first, then the alternatives from identical initial-state "
        "fingerprints. Evaluation labels were not used for retraining.")
    heading("Predictor and selection rule")
    add(f"Random forest: {metadata['settings']['n_estimators']} trees, maximum depth "
        f"{metadata['settings']['max_depth']}, one worker. Select the fastest speed with raw "
        f"predicted probability at least {protocol['threshold']:.1f}. Otherwise select the "
        "highest score, breaking ties toward the slower speed, and mark the threshold unmet. "
        "This pilot uses neither GPT, a 27B action model, nor MRIoC to choose speed.")
    examples = ["bowl.evaluation_00", "bottle.evaluation_02", "bottle.evaluation_08"]
    pred_by_id = {row["case"]["case_id"]: row for row in predictions}
    out_by_id = {row["case"]["case_id"]: row for row in outcomes}
    rows = [["Held-out condition", "0.04", "0.10", "0.20", "Chosen m/s", "Observed"]]
    for case_id in examples:
        row = pred_by_id[case_id]
        score_by_speed = {c["speed_m_s"]: c["predicted_success_probability"] for c in row["candidates"]}
        rows.append([case_id, *[f"{score_by_speed[s]:.1f}" for s in protocol["speeds_m_s"]],
                     f"{row['selected_speed_m_s']:.2f}" + (" *" if not row["threshold_met"] else ""),
                     "Success" if out_by_id[case_id]["success"] else "Failure"])
    table(rows, [WIDTH * x for x in (.31, .10, .10, .10, .16, .13)])
    add("* Threshold unmet; execution still followed the predeclared simulation policy. "
        "These probabilities are uncalibrated, not safety guarantees. The final row is an "
        "explicit wrong prediction: a zero-score grasp actually succeeded.", "SmallReport")
    heading("A positive label requires physical evidence")
    add(f"Lift at least {protocol['minimum_lift_m']:.2f} m, bilateral grip, table clearance and "
        f"stable relative pose for {protocol['required_hold_s']:.0f} continuous simulation seconds. "
        "An independent observer checks every 2 ms; traces are saved every 20 ms and at phase changes. "
        "The older controller's one-second success flag alone does not suffice.")

    # Page 3: task outcomes and concise answers to the professor.
    story.append(PageBreak())
    add("Task performance and interpretation", "SectionTitle")
    comparison = summary["speed_comparison"]
    policy_rows = [["Policy", "Successes", "Mean successful-task duration (sim s)"]]
    for key, label in [("0.04", "Fixed 0.04 m/s"), ("0.1", "Fixed 0.10 m/s"),
                       ("0.2", "Fixed 0.20 m/s"), ("recommended", "Random-forest recommendation")]:
        row = comparison["methods"][key]
        policy_rows.append([label, f"{row['successes']}/{row['attempted']}", f"{row['mean_success_execution_sim_s']:.3f}"])
    table(policy_rows, [WIDTH * .46, WIDTH * .16, WIDTH * .38])
    add("Durations are success-only simulation time, not video length or collection wall time. "
        "Here, the same 16 conditions succeeded under all four policies.", "SmallReport")
    delta = comparison["recommended_minus_fastest_on_joint_success"]["mean_sim_s"]
    note(f"The learned recommendation was <b>{delta:.3f} simulation seconds slower</b> than the "
         "fixed fastest setting on joint successes. Across both splits, every valid condition "
         "had the same success/failure label at all three speeds. No fastest-fails/slower-succeeds "
         "case was observed. There is therefore no demonstrated adaptive-speed benefit.")
    metrics = summary["prediction_metrics"]
    add(f"Descriptive prediction accuracy: {metrics['accuracy_at_0_5']:.2%}; Brier score: "
        f"{metrics['brier_score']:.5f}. Classification accuracy is not evidence of faster or "
        "more reliable speed selection. Probabilities remain uncalibrated.")
    heading("Failures and safety boundary")
    add("Training failures: 15 descent-arrival failures, one lift timeout and two hold failures. "
        "All six evaluation failures were descent-arrival failures. Unexpected contacts occurred "
        "in nine evaluation trials, including three successful grasps. No numerical warnings "
        "were recorded. Success does not imply collision-free operation or real-robot readiness.")
    heading("Answers to the professor's three questions")
    add("<b>1. Is the simulator ready and fast enough?</b> Yes for a small classifier feasibility "
        "study: the measured batch completed real-label collection, training and recommended "
        "physical execution. The environment is already operational rather than a proposed build.")
    add("<b>2. What changes besides speed?</b> Object category, reset X/Y and yaw. Per-object "
        "physical parameters and grasp rules stay fixed. The scope is two existing assets and "
        "held-out initial states, not arbitrary new objects.")
    add("<b>3. Why not D-optimal?</b> It is a legitimate candidate control, not something this "
        "pilot rules out. Active learning, random sampling and D-optimal design have not yet "
        "been compared. A fair comparison requires a specified model/design criterion and "
        "the same initial data, label budget and independent evaluation set.")
    heading("Next research decision")
    add("First verify how the speed settings affect actual trajectories and seek reasonable, "
        "predeclared conditions with meaningful success/time tradeoffs. Do not manufacture "
        "negative labels. Only then test learning curves and task performance. If speed choice "
        "remains trivial, revise the question with the professor rather than claiming an advantage.")

    # Page 4: demonstration context, reproducibility, and limits of validation.
    story.append(PageBreak())
    add("Demonstrations and reproducible evidence", "SectionTitle")
    add("These two independent historical recordings show the broader prototype. Their English "
        "editions retain the original motion and presentation timestamps; they are not new "
        "simulation runs or new GPT requests. <b>Neither is a classifier evaluation trial.</b>")
    storage_image, storage_size, storage_t = video_still(storage["output"]["path"], 36.0,
                                                        storage["geometry"]["destination_scene_box_xyxy"])
    sorting_image, sorting_size, sorting_t = video_still(sorting["output"]["path"], 20.0,
                                                        sorting["presentation"]["scene_destination_box_xyxy"])
    image_width = (WIDTH - 14) / 2
    image_table = Table([[
        Image(storage_image, width=image_width, height=image_width * storage_size[1] / storage_size[0]),
        Image(sorting_image, width=image_width, height=image_width * sorting_size[1] / sorting_size[0]),
    ], [
        p(f"<b>Black-bowl storage.</b> Pick, place in the drawer, close it, return home. "
          f"Frame at {storage_t:.2f} s; one approved proposal followed by a fixed local sequence.", "SmallReport"),
        p(f"<b>Six-object sorting.</b> Three bowls and three cups. Frame at {sorting_t:.2f} s; "
          "one approved GPT queue followed by local contact-feedback execution.", "SmallReport"),
    ]], colWidths=[WIDTH / 2, WIDTH / 2])
    image_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(image_table)
    add(link(storage["output"]["path"], "English storage video") + " | " +
        link(sorting["output"]["path"], "English sorting video"), "SmallReport")
    heading("Reproduce the Level 1 experiment")
    story.append(Preformatted(
        "uv sync --locked\n"
        "uv run --locked python run_grasp_learning.py \\\n"
        "  --output results/grasp_level1_reproduction", styles["CodeReport"]))
    add("Run from the repository root; the output directory must not exist. Python "
        f"{environment['python']}, MuJoCo {environment['packages']['mujoco']}, "
        f"scikit-learn {environment['packages']['scikit-learn']}. Only load joblib files you trust.", "SmallReport")
    heading("Evidence and provenance")
    batch_rel = BATCH.relative_to(ROOT).as_posix()
    add(f'<font name="Courier" size="8">{batch_rel}/</font>', "SmallReport")
    add(" | ".join(link(f"{batch_rel}/{name}", label) for name, label in [
        ("protocol.json", "Frozen protocol"), ("environment.json", "Environment / fingerprints"),
        ("summary.json", "Summary"), ("recommendations.json", "Predictions"),
        ("recommendation_outcomes.json", "Executed outcomes"), ("record.md", "Detailed record"),
    ]), "SmallReport")
    add("Raw trials, compressed physical traces, training/evaluation data and the saved model "
        "are retained alongside these files. This verified batch replays the earlier batch's "
        "frozen protocol; it is not an independent dataset. Do not pool the two batches "
        "into 324 independent samples.", "SmallReport")
    summary_hash = hashlib.sha256((BATCH / "summary.json").read_bytes()).hexdigest()
    add(f'Summary SHA-256:<br/><font name="Courier" size="7">{summary_hash}</font>', "SmallReport")
    heading("Verification scope")
    add("All 34 added Level 1 checks passed in the initial 319-test run. That run also had nine "
        "errors across three historical sorting methods/subtests due to fingerprint mismatch. "
        "After test-fixture fixes, nine relevant protocol/physics tests and one server-layout "
        "test passed; the complete suite was not rerun. Original logs remain archived.", "SmallReport")
    add("This report regenerates from archived evidence via "
        '<font name="Courier" size="8">uv run --script build_grasp_speed_report.py</font>. '
        "Linked repository artifacts require GitHub access while the repository is private.", "SmallReport")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(48, 40, A4[0] - 48, 40)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(48, 27, "Level 1 feasibility | Verified batch: 2026-11-04")
        canvas.drawRightString(A4[0] - 48, 27, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=48, rightMargin=48,
                                 topMargin=42, bottomMargin=54, title=TITLE,
                                 author="GPT robotic grasping project", subject="Verified Level 1 results and limitations")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(json.dumps({"output": OUTPUT.name, "bytes": OUTPUT.stat().st_size,
                      "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                      "source_summary_sha256": summary_hash,
                      "new_experiment": False}, indent=2))


if __name__ == "__main__":
    build()
