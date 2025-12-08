import logging
from logging.handlers import RotatingFileHandler

# ============================
# Logging Configuration
# ============================
LOG_FILE = "app_logs.log"

logger = logging.getLogger("SymptomCheckerLogger")
logger.setLevel(logging.INFO)

# Rotating log — keeps last 5 files, each max 1 MB
handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=5)
formatter = logging.Formatter(
    "%(asctime)s — %(levelname)s — %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


from __future__ import annotations

import os
import json
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai


# =========================
# 0) Gemini client setup
# =========================

load_dotenv()  # loads GEMINI_API_KEY from .env if present

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Please set it in environment or .env file."
    )

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL_NAME = "gemini-2.5-flash"  # lightweight model, good for JSON


# =========================
# 1) Disease list (15 diseases)
# =========================

DISEASES = [
    "Viral Fever (without warning signs)",
    "Gastritis / Acid Reflux",
    "Migraine / Tension Headache",
    "Skin Infection / Cellulitis (mild)",
    "Moderate Hypertension (BP 140–160/90–100)",
    "Pregnancy Complications (Bleeding / Pain)",
    "Severe Dehydration",
    "Snake Bite (Suspected)",
    "Tuberculosis (with cough >2 weeks)",
    "Seizure / Fits",
    "Heart Attack",
    "Unconscious / Coma",
    "Stroke (CVA)",
    "Severe Head Injury",
    "Severe Trauma with Bleeding / Fracture",
]

ALLOWED_ZONES = ["Red", "Orange", "Yellow"]

ZONE_LABELS = {
    "Red": "Zone: 🔴 Red – उच्च धोक्याची पातळी",
    "Orange": "Zone: 🟠 Orange – मध्यम धोक्याची पातळी",
    "Yellow": "Zone: 🟡 Yellow – कमी धोक्याची पातळी",
}


# =========================
# 2) Gemini-based classifier
# =========================

def classify_with_gemini(text: str) -> Dict:
    """
    Use Gemini to:
      - Choose exactly ONE disease from DISEASES
      - Choose ONE zone from [Red, Orange, Yellow]
      - Generate Marathi symptom & action lines
    """

    system_instructions = f"""
You are an AI symptom checker triage assistant.

The patient speaks Marathi. Respond in SIMPLE Marathi for all patient-facing text.

You must:
1) Read the patient's free-text complaint (in Marathi).
2) Choose exactly ONE disease from the following list (no other disease allowed):

   1. Viral Fever (without warning signs) – viral fever with mild to moderate symptoms, no danger signs.
   2. Gastritis / Acid Reflux – burning in chest or upper abdomen, acidity, related to food.
   3. Migraine / Tension Headache – repeated or severe headache, sometimes with vomiting or light sensitivity.
   4. Skin Infection / Cellulitis (mild) – local redness, swelling, pain, mild fever.
   5. Moderate Hypertension (BP 140–160/90–100) – raised blood pressure with mild symptoms like headache, giddiness.
   6. Pregnancy Complications (Bleeding / Pain) – pregnant woman with vaginal bleeding or abdominal pain.
   7. Severe Dehydration – very weak, dry mouth, very little urine, dizziness, especially with diarrhea or vomiting.
   8. Snake Bite (Suspected) – history of snake bite or strong suspicion, with or without swelling.
   9. Tuberculosis (with cough >2 weeks) – cough more than 2 weeks, weight loss, night sweats.
   10. Seizure / Fits – episode of convulsions, loss of control, tongue bite, post-ictal confusion.
   11. Heart Attack (Myocardial Infarction) – severe chest pain, chest heaviness, breathlessness, sweating, radiating pain.
   12. Unconscious / Coma – not responding, very drowsy, not following commands.
   13. Stroke (CVA) – sudden weakness of one side, facial droop, slurred speech, difficulty walking.
   14. Severe Head Injury – head trauma with loss of consciousness, vomiting, confusion, bleeding.
   15. Severe Trauma with Bleeding / Fracture – major accident, heavy bleeding, suspected fracture, limb deformity.

3) Decide the triage zone:
   - "Red"    = Emergency / life threatening → needs immediate ER / 108 call.
   - "Orange" = Urgent (high risk but not immediate death) → needs doctor same day / within few hours.
   - "Yellow" = Mild / stable → can manage at home + OPD visit if needed.

You MUST output:

- "disease": exactly one string from the above list.
- "zone": exactly one of: {ALLOWED_ZONES}.
- "symptoms_line": ONE short Marathi sentence summarizing the main symptoms.
- "action_line": ONE or TWO short Marathi sentences telling the patient what to do now
  (ER / call 108 / go to hospital / home care + when to see doctor).

IMPORTANT:
- For Red zone, clearly tell the patient to go to emergency / call 108.
- For Orange zone, clearly tell to see a doctor or hospital as soon as possible.
- For Yellow zone, focus on home care + OPD visit if not improving.

OUTPUT FORMAT:
Return ONLY valid JSON with keys:
  disease, zone, symptoms_line, action_line

No extra text, no explanation, no markdown, no backticks.
"""

    user_context = f"""
PATIENT COMPLAINT (Marathi):
{text}
"""

    prompt = system_instructions + "\n" + user_context

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
        )
        raw_text = response.text.strip()
    except Exception as e:
        print("Gemini API error:", e)
        raise RuntimeError("Gemini call failed") from e

    # Try to parse JSON
    try:
        data = json.loads(raw_text)
    except Exception:
        cleaned = raw_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)

    # Validate keys
    for key in ["disease", "zone", "symptoms_line", "action_line"]:
        if key not in data:
            raise RuntimeError(f"Missing key in Gemini response: {key}")

    # Safety clamp: make sure disease & zone are valid
    disease = data["disease"]
    zone = data["zone"]

    if disease not in DISEASES:
        # Fallback if hallucinated disease name
        disease = "Viral Fever (without warning signs)"

    if zone not in ALLOWED_ZONES:
        zone = "Yellow"

    return {
        "disease": disease,
        "zone": zone,
        "symptoms_line": data["symptoms_line"],
        "action_line": data["action_line"],
    }


# =========================
# 3) FastAPI app & models
# =========================

app = FastAPI(title="AI Symptom Checker - 15 Diseases (Gemini)")

# CORS – useful for Android / web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict to your domain/app later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    complaint: str


class AnalyzeResponse(BaseModel):
    zone: str
    zone_label: str
    disease: str
    patient_symptoms_line: str
    patient_action_line: str


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Main API:
      - takes Marathi complaint
      - uses Gemini to classify into one of 15 diseases + triage zone
      - returns Marathi symptom + action lines
    """
    print("bhavana")
    logger.info(f"User Complaint: {req.complaint}")
    result = classify_with_gemini(req.complaint)
    logger.info(f"AI Response Zone: {result['zone']}")
    logger.info(f"Symptoms Line: {result['patient_symptoms_line']}")


    zone = result["zone"]
    zone_label = ZONE_LABELS.get(zone, f"Zone: {zone}")

    return AnalyzeResponse(
        zone=zone,
        zone_label=zone_label,
        disease=result["disease"],
        patient_symptoms_line=result["symptoms_line"],
        patient_action_line=result["action_line"],
    )
