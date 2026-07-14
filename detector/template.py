import json
import os
from datetime import datetime

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")


def _ensure_dir():
    os.makedirs(TEMPLATE_DIR, exist_ok=True)


def list_templates():
    _ensure_dir()
    templates = []
    for f in sorted(os.listdir(TEMPLATE_DIR)):
        if f.endswith(".json"):
            path = os.path.join(TEMPLATE_DIR, f)
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                templates.append({
                    "filename": f,
                    "name": data.get("name", f[:-5]),
                    "created": data.get("created", "unknown"),
                    "num_subjective": data.get("num_subjective", 0),
                    "num_objective_answers": len(data.get("answer_key", {})),
                })
            except Exception:
                pass
    return templates


def load_template(name):
    _ensure_dir()
    if not name.endswith(".json"):
        name += ".json"
    path = os.path.join(TEMPLATE_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def save_template(template_data):
    _ensure_dir()
    name = template_data.get("name", "template")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    filename = f"{safe_name}.json"
    path = os.path.join(TEMPLATE_DIR, filename)

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(template_data, fp, ensure_ascii=False, indent=2)

    print(f"[INFO] Template saved: {filename}")
    return filename


def create_template(image, answer_key, num_questions,
                    num_subjective=0, name="template"):
    template = {
        "name": name,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "answer_key": {str(k): v for k, v in answer_key.items()},
        "num_questions": num_questions,
        "num_subjective": num_subjective,
        "objective_roi": None,
        "subjective_rois": [],
    }

    from utils.roi_selector import select_roi_interactive

    if num_subjective > 0:
        subjective_rois = []
        for i in range(num_subjective):
            prompt = (f"Subjective Q{i+1}/{num_subjective} (Template)\n"
                      f"Drag to select, Enter to add, N to finish")
            rois = select_roi_interactive(
                image,
                title=f"Template - Subjective Q{i+1}",
                prompt=prompt,
                allow_multiple=True,
            )
            if rois:
                subjective_rois.append(rois)
            else:
                subjective_rois.append([])
        template["subjective_rois"] = subjective_rois

    prompt = "Select objective region for template"
    objective_roi = select_roi_interactive(
        image,
        title="Template - Objective Region",
        prompt=prompt,
    )
    template["objective_roi"] = objective_roi

    filename = save_template(template)
    return template, filename


def delete_template(name):
    _ensure_dir()
    if not name.endswith(".json"):
        name += ".json"
    path = os.path.join(TEMPLATE_DIR, name)
    if os.path.exists(path):
        os.remove(path)
        print(f"[INFO] Template deleted: {name}")
        return True
    return False
