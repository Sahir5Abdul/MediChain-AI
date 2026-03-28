# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import httpx
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests as req

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# n8n webhook URL (server-side only). Remote users call POST /webhook/run-agents on FastAPI; this proxies to n8n.
# Override if n8n listens elsewhere: set env N8N_WEBHOOK_URL=http://127.0.0.1:5678/webhook/run-agents
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://127.0.0.1:5678/webhook/run-agents").strip()

app = FastAPI(title="Healthcare Supply Chain AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3:latest"

# ==============================================================================
# HELPERS
# ==============================================================================

def load_json(path):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{path} not found. Run /run-pipeline first.")
    with open(path) as f:
        return json.load(f)

def load_json_optional(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def ask_llm(prompt: str) -> str:
    response = httpx.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }, timeout=180)
    return response.json()["response"]

# ==============================================================================
# HEALTH
# ==============================================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "system": "Healthcare Supply Chain AI Agent",
        "timestamp": datetime.now().isoformat(),
        "dashboard": "/dashboard",
        "dashboard_aliases": ["/app", "/ui"],
        "pipeline_proxy": "/webhook/run-agents",
        "note": "Open /dashboard (or /app) in the browser (same origin as API when using ngrok). "
        "Run Pipeline posts to /webhook/run-agents (proxied to n8n on the server). "
        "If /dashboard returns 404, restart uvicorn so it loads this main.py.",
    }


def _dashboard_html():
    """Load dashboard HTML from the folder that contains main.py (works regardless of process cwd)."""
    path = os.path.join(BASE_DIR, "medichain_dashboard.html")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="medichain_dashboard.html not found next to main.py at {}".format(path),
        )
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), media_type="text/html")


@app.get("/dashboard")
@app.get("/app")
@app.get("/ui")
def serve_dashboard():
    """Serve the SPA so testers can use https://<ngrok>/dashboard with same-origin API calls."""
    return _dashboard_html()


@app.get("/dashboard/")
def serve_dashboard_trailing_slash():
    return RedirectResponse(url="/dashboard", status_code=307)


@app.post("/webhook/run-agents")
async def proxy_n8n_webhook(request: Request):
    """
    Proxy n8n webhook for remote testers: the browser only talks to FastAPI (e.g. via ngrok).
    This server forwards the JSON body to the local n8n workflow URL.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    url = N8N_WEBHOOK_URL
    if not url:
        raise HTTPException(
            status_code=503,
            detail="N8N_WEBHOOK_URL is not set. Configure the n8n webhook URL on the server.",
        )
    try:
        r = httpx.post(url, json=body, timeout=300.0)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail="Cannot reach n8n at {}. Is n8n running? Error: {}".format(url, e),
        )
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail="n8n returned HTTP {}: {}".format(r.status_code, (r.text or "")[:800]),
        )
    if not (r.text or "").strip():
        return {"status": "ok", "forwarded_to": url}
    try:
        return r.json()
    except Exception:
        return {"status": "ok", "forwarded_to": url, "body": (r.text or "")[:2000]}

# ==============================================================================
# DATA PIPELINE  (rule-based, generates all JSON files)
# ==============================================================================

@app.post("/run-pipeline")
def run_pipeline():
    try:
        os.makedirs("data", exist_ok=True)
        # NOTE: previously this was seeded with a constant (42), which made every run identical.
        # Use a time-based seed so each run produces fresh simulated inputs.
        np.random.seed(int(datetime.now().timestamp()))

        # Fetch FDA data (real-delta mode: prefer recent time window from upstream API)
        def fetch_paginated(url, total=500, date_fields=None, lookback_days=45):
            results = []
            limit, skip = 100, 0
            end_dt = datetime.utcnow()
            start_dt = end_dt - timedelta(days=lookback_days)
            start_s = start_dt.strftime("%Y%m%d")
            end_s = end_dt.strftime("%Y%m%d")

            # Try multiple candidate date fields for each OpenFDA endpoint.
            # If no date field works for an endpoint, fall back to unfiltered paging.
            search_candidates = []
            for f in (date_fields or []):
                search_candidates.append('{0}:[{1}+TO+{2}]'.format(f, start_s, end_s))
            search_candidates.append(None)

            active_search = None
            for candidate in search_candidates:
                test_params = {"limit": 1, "skip": 0}
                if candidate:
                    test_params["search"] = candidate
                test = req.get(url, params=test_params)
                if test.status_code == 200:
                    active_search = candidate
                    break

            while len(results) < total:
                params = {"limit": limit, "skip": skip}
                if active_search:
                    params["search"] = active_search
                r = req.get(url, params=params)
                if r.status_code != 200:
                    break
                batch = r.json().get("results", [])
                if not batch:
                    break
                results.extend(batch)
                skip += limit
            return results[:total]

        shortages = fetch_paginated(
            "https://api.fda.gov/drug/shortages.json",
            500,
            date_fields=["report_date", "update_date", "decision_date"],
            lookback_days=45,
        )
        recalls = fetch_paginated(
            "https://api.fda.gov/drug/enforcement.json",
            500,
            date_fields=["recall_initiation_date", "report_date", "classification_date"],
            lookback_days=45,
        )

        with open("data/shortages.json", "w") as f:
            json.dump(shortages, f)
        with open("data/recalls.json", "w") as f:
            json.dump(recalls, f)

        # Hospital inventory: use flat hospital_inventory.csv (not timeseries).
        inv_root = os.path.join(BASE_DIR, "hospital_inventory.csv")
        inv_data = os.path.join(BASE_DIR, "data", "hospital_inventory.csv")
        inv_path = inv_root if os.path.exists(inv_root) else inv_data

        if os.path.exists(inv_path):
            inventory = pd.read_csv(inv_path)
            required = {
                "hospital", "drug", "current_stock_units", "daily_consumption",
                "days_remaining", "reorder_point", "supplier", "supplier_country", "requires_cold_chain",
            }
            missing = required - set(inventory.columns)
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail="hospital_inventory.csv missing columns: " + ", ".join(sorted(missing)),
                )
            _cc = inventory["requires_cold_chain"]
            if _cc.dtype == bool:
                inventory["requires_cold_chain"] = _cc
            else:
                inventory["requires_cold_chain"] = _cc.astype(str).str.lower().isin(["true", "1", "yes"])
            inventory["daily_consumption"] = inventory["daily_consumption"].replace({0: 1})
            out_inv = os.path.join(BASE_DIR, "data", "hospital_inventory.csv")
            inventory.to_csv(out_inv, index=False)

            weather = [
                {"city": "Chennai",   "max_temp_72h": 33.5, "cold_chain_risk": "HIGH"},
                {"city": "Delhi",     "max_temp_72h": 32.4, "cold_chain_risk": "HIGH"},
                {"city": "Mumbai",    "max_temp_72h": 29.7, "cold_chain_risk": "LOW"},
                {"city": "Bangalore", "max_temp_72h": 34.8, "cold_chain_risk": "HIGH"},
                {"city": "Hyderabad", "max_temp_72h": 36.7, "cold_chain_risk": "CRITICAL"},
            ]
            with open(os.path.join(BASE_DIR, "data", "weather_risk.json"), "w") as f:
                json.dump(weather, f)
        else:
            drugs = ["Insulin", "Amoxicillin", "Vancomycin", "Epinephrine", "Morphine",
                     "Cisplatin", "Methotrexate", "Dexamethasone", "Heparin", "Furosemide"]
            hospitals = ["Apollo Chennai", "AIIMS Delhi", "Fortis Mumbai", "Manipal Bangalore", "KIMS Hyderabad"]
            suppliers = ["Sun Pharma", "Cipla", "Dr Reddys", "Aurobindo", "Lupin"]

            rows = []
            for h in hospitals:
                for d in drugs:
                    daily = int(np.random.randint(10, 80))
                    stock = int(np.random.randint(50, 500))
                    rows.append({
                        "hospital": h, "drug": d,
                        "current_stock_units": stock,
                        "daily_consumption": daily,
                        "days_remaining": round(stock / daily, 1),
                        "reorder_point": int(np.random.randint(30, 100)),
                        "supplier": str(np.random.choice(suppliers)),
                        "supplier_country": str(np.random.choice(["India", "China", "USA", "Germany"])),
                        "requires_cold_chain": d in ["Insulin", "Cisplatin", "Methotrexate"],
                    })

            inventory = pd.DataFrame(rows)
            inventory.to_csv(os.path.join(BASE_DIR, "data", "hospital_inventory.csv"), index=False)

            weather = [
                {"city": "Chennai",   "max_temp_72h": 33.5, "cold_chain_risk": "HIGH"},
                {"city": "Delhi",     "max_temp_72h": 32.4, "cold_chain_risk": "HIGH"},
                {"city": "Mumbai",    "max_temp_72h": 29.7, "cold_chain_risk": "LOW"},
                {"city": "Bangalore", "max_temp_72h": 34.8, "cold_chain_risk": "HIGH"},
                {"city": "Hyderabad", "max_temp_72h": 36.7, "cold_chain_risk": "CRITICAL"},
            ]
            with open(os.path.join(BASE_DIR, "data", "weather_risk.json"), "w") as f:
                json.dump(weather, f)

        # Agent 1 - Disruption detection
        def match_shortages(records, inventory):
            flags = []
            inv_drugs = inventory["drug"].str.lower().tolist()
            for s in records:
                name = (s.get("generic_name","") or s.get("product_description","")).lower()
                if not name:
                    continue
                for inv_drug in inv_drugs:
                    if inv_drug in name or name in inv_drug:
                        for _, row in inventory[inventory["drug"].str.lower()==inv_drug].iterrows():
                            flags.append({
                                "type": "FDA_SHORTAGE",
                                "drug": row["drug"], "hospital": row["hospital"],
                                "days_remaining": row["days_remaining"],
                                "supplier": row.get("supplier", None),
                                "supplier_country": row.get("supplier_country", None),
                                "fda_status": s.get("status","Unknown"),
                                "severity": "CRITICAL" if row["days_remaining"]<7 else "HIGH" if row["days_remaining"]<14 else "MEDIUM"
                            })
            return flags

        def match_cold_chain(weather, inventory):
            flags = []
            city_risk = {w["city"]: w["cold_chain_risk"] for w in weather}
            city_map  = {
                "Apollo Chennai":"Chennai","AIIMS Delhi":"Delhi",
                "Fortis Mumbai":"Mumbai","Manipal Bangalore":"Bangalore","KIMS Hyderabad":"Hyderabad"
            }
            for _, row in inventory[inventory["requires_cold_chain"]==True].iterrows():
                city = city_map.get(row["hospital"])
                risk = city_risk.get(city,"LOW")
                if risk in ["HIGH","CRITICAL"]:
                    flags.append({
                        "type": "COLD_CHAIN_RISK", "drug": row["drug"],
                        "hospital": row["hospital"], "cold_chain_risk": risk,
                        "days_remaining": row["days_remaining"],
                        "supplier": row.get("supplier", None),
                        "supplier_country": row.get("supplier_country", None),
                        "severity": "CRITICAL" if risk=="CRITICAL" else "HIGH"
                    })
            return flags

        def detect_low_stock(inventory):
            flags = []
            low = inventory[inventory["days_remaining"] < inventory["reorder_point"]/inventory["daily_consumption"]]
            for _, row in low.iterrows():
                flags.append({
                    "type": "LOW_STOCK", "drug": row["drug"], "hospital": row["hospital"],
                    "days_remaining": row["days_remaining"], "supplier": row["supplier"],
                    "supplier_country": row["supplier_country"],
                    "severity": "CRITICAL" if row["days_remaining"]<5 else "HIGH"
                })
            return flags

        all_flags = (
            match_shortages(shortages, inventory) +
            match_shortages(recalls, inventory) +
            match_cold_chain(weather, inventory) +
            detect_low_stock(inventory)
        )
        all_flags.sort(key=lambda x: {"CRITICAL":0,"HIGH":1,"MEDIUM":2}.get(x["severity"],3))

        disruption_out = {
            "timestamp": datetime.now().isoformat(),
            "total_flags": len(all_flags),
            "critical": sum(1 for f in all_flags if f["severity"]=="CRITICAL"),
            "high":     sum(1 for f in all_flags if f["severity"]=="HIGH"),
            "medium":   sum(1 for f in all_flags if f["severity"]=="MEDIUM"),
            "flags": all_flags
        }
        with open("data/disruption_signals.json","w") as f:
            json.dump(disruption_out, f)

        # Agent 2 - Ripple modeling
        SUPPLIER_RISK  = {"China":0.85,"USA":0.70,"Germany":0.50,"India":0.20}
        PATIENT_IMPACT = {
            "Morphine":12,"Insulin":35,"Vancomycin":8,"Epinephrine":5,
            "Cisplatin":6,"Methotrexate":7,"Amoxicillin":20,
            "Dexamethasone":10,"Heparin":9,"Furosemide":11
        }
        DRUG_COST = {
            "Morphine":45,"Insulin":180,"Vancomycin":850,"Epinephrine":320,
            "Cisplatin":2400,"Methotrexate":1100,"Amoxicillin":25,
            "Dexamethasone":60,"Heparin":420,"Furosemide":30
        }
        ALT_SUPPLIERS = {
            "Sun Pharma":["Cipla","Dr Reddys"],"Cipla":["Sun Pharma","Aurobindo"],
            "Dr Reddys":["Lupin","Sun Pharma"],"Aurobindo":["Cipla","Lupin"],
            "Lupin":["Dr Reddys","Aurobindo"]
        }

        seen, ripple_reports = set(), []
        for flag in [f for f in all_flags if f["severity"]=="CRITICAL"]:
            key = f"{flag['drug']}_{flag['hospital']}"
            if key in seen:
                continue
            seen.add(key)
            inv_row = inventory[(inventory["drug"]==flag["drug"]) & (inventory["hospital"]==flag["hospital"])]
            if inv_row.empty:
                continue
            row = inv_row.iloc[0]
            dr  = flag["days_remaining"]
            ripple_reports.append({
                "drug": flag["drug"], "hospital": flag["hospital"],
                "flag_type": flag["type"], "severity": flag["severity"],
                "days_remaining": dr,
                "stockout_date": (datetime.now()+timedelta(days=dr)).strftime("%Y-%m-%d"),
                "current_supplier": row["supplier"],
                "supplier_country": row["supplier_country"],
                "supplier_risk_score": SUPPLIER_RISK.get(row["supplier_country"],0.5),
                "patients_at_risk": int(PATIENT_IMPACT.get(flag["drug"],10)*max(0,30-dr)),
                "financial_impact_inr": int(row["daily_consumption"]*DRUG_COST.get(flag["drug"],100)*2.5*max(0,30-dr)),
                "alternative_suppliers": ALT_SUPPLIERS.get(row["supplier"],["Contact CDSCO registry"]),
                "recommended_action": "EMERGENCY_REORDER" if dr<5 else "EXPEDITE_REORDER" if dr<14 else "SCHEDULE_REORDER"
            })

        ripple_out = {
            "timestamp": datetime.now().isoformat(),
            "total_critical_scenarios": len(ripple_reports),
            "total_patients_at_risk": sum(r["patients_at_risk"] for r in ripple_reports),
            "total_financial_impact_inr": sum(r["financial_impact_inr"] for r in ripple_reports),
            "emergency_reorders_needed": sum(1 for r in ripple_reports if r["recommended_action"]=="EMERGENCY_REORDER"),
            "ripple_reports": ripple_reports
        }
        with open("data/ripple_model.json","w") as f:
            json.dump(ripple_out, f)

        # Agent 3 - Compliance
        ICD10_MAP = {
            "Morphine":     {"code":"Z79.891","description":"Long-term use of opioid analgesic","controlled":True},
            "Insulin":      {"code":"Z79.4",  "description":"Long-term use of insulin",         "controlled":False},
            "Vancomycin":   {"code":"Z79.2",  "description":"Long-term antibiotic use",         "controlled":False},
            "Epinephrine":  {"code":"T48.291","description":"Emergency vasopressor",            "controlled":False},
            "Cisplatin":    {"code":"Z79.899","description":"Chemotherapy agent",               "controlled":False},
            "Methotrexate": {"code":"Z79.899","description":"Immunosuppressant/chemo",          "controlled":False},
            "Amoxicillin":  {"code":"Z79.2",  "description":"Antibiotic use",                   "controlled":False},
            "Dexamethasone":{"code":"Z79.52", "description":"Long-term steroid use",            "controlled":False},
            "Heparin":      {"code":"Z79.01", "description":"Long-term anticoagulant use",      "controlled":False},
            "Furosemide":   {"code":"Z79.899","description":"Long-term diuretic use",           "controlled":False}
        }

        def icd10_for_drug(drug_name: str):
            """
            Prefer known mappings; otherwise generate a stable, non-identical placeholder ICD-10 code per drug.
            (Your current dataset includes many drugs not covered by the small hardcoded map above.)
            """
            if drug_name in ICD10_MAP:
                return ICD10_MAP[drug_name]

            # Stable pseudo-code based on drug name (same drug => same code across runs)
            h = sum((i + 1) * ord(c) for i, c in enumerate(str(drug_name))) % 900
            code = "Z79.{0:03d}".format(100 + (h % 900))
            return {
                "code": code,
                "description": "Medication management (placeholder mapping)",
                "controlled": False,
            }
        PROC_RULES = {
            "EMERGENCY_REORDER": {
                "requires_cdsco_approval":True,"requires_dual_supplier":True,
                "max_country_risk":0.6,"controlled_check":True
            },
            "EXPEDITE_REORDER": {
                "requires_cdsco_approval":False,"requires_dual_supplier":True,
                "max_country_risk":0.7,"controlled_check":True
            },
            "SCHEDULE_REORDER": {
                "requires_cdsco_approval":False,"requires_dual_supplier":False,
                "max_country_risk":0.85,"controlled_check":False
            }
        }

        compliance_results = []
        for r in ripple_reports:
            drug   = r["drug"]
            action = r["recommended_action"]
            rules  = PROC_RULES[action]
            icd    = icd10_for_drug(drug)
            v, w, p = [], [], []

            if icd["controlled"] and rules["controlled_check"]:
                if action == "EMERGENCY_REORDER":
                    w.append(f"{drug} is a controlled substance (ICD-10: {icd['code']}) - NDPS Act compliance required")
                p.append("Controlled substance flagged for regulatory review")

            if r["supplier_risk_score"] > rules["max_country_risk"]:
                v.append(f"Supplier risk score {r['supplier_risk_score']} exceeds threshold {rules['max_country_risk']}")
            else:
                p.append(f"Supplier risk score {r['supplier_risk_score']} within threshold")

            if rules["requires_dual_supplier"] and len(r["alternative_suppliers"]) < 2:
                v.append("Dual supplier requirement not met")
            else:
                p.append("Dual supplier requirement met")

            if rules["requires_cdsco_approval"]:
                w.append("CDSCO approval required before executing emergency procurement")

            status = "BLOCKED" if v else "APPROVED_WITH_WARNINGS" if w else "APPROVED"
            compliance_results.append({
                "drug": drug, "hospital": r["hospital"], "action": action,
                "icd10_code": icd["code"], "icd10_description": icd["description"],
                "compliance_status": status, "violations": v,
                "warnings": w, "checks_passed": p,
                "checked_at": datetime.now().isoformat()
            })

        compliance_out = {
            "timestamp": datetime.now().isoformat(),
            "total_checked": len(compliance_results),
            "approved":               sum(1 for r in compliance_results if r["compliance_status"]=="APPROVED"),
            "approved_with_warnings": sum(1 for r in compliance_results if r["compliance_status"]=="APPROVED_WITH_WARNINGS"),
            "blocked":                sum(1 for r in compliance_results if r["compliance_status"]=="BLOCKED"),
            "results": compliance_results
        }
        with open("data/compliance_results.json","w") as f:
            json.dump(compliance_out, f)

        # Agent 4 - Audit
        run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trail  = []
        trail.append({
            "run_id":run_id,"step":1,"agent":"DisruptionDetectionAgent",
            "timestamp":disruption_out["timestamp"],
            "decision":f"Escalated {disruption_out['critical']} critical flags",
            "status":"COMPLETED"
        })
        trail.append({
            "run_id":run_id,"step":2,"agent":"RippleModelingAgent",
            "timestamp":ripple_out["timestamp"],
            "decision":f"Modeled {ripple_out['total_critical_scenarios']} scenarios",
            "status":"COMPLETED"
        })
        for r in compliance_results:
            trail.append({
                "run_id":run_id,"step":3,"agent":"ComplianceGuardrailAgent",
                "timestamp":r["checked_at"],
                "drug":r["drug"],"hospital":r["hospital"],
                "decision":r["compliance_status"],"status":"COMPLETED"
            })

        approved_items = [r for r in compliance_results if r["compliance_status"]=="APPROVED"]
        warned_items   = [r for r in compliance_results if r["compliance_status"]=="APPROVED_WITH_WARNINGS"]
        blocked_items  = [r for r in compliance_results if r["compliance_status"]=="BLOCKED"]

        trail.append({
            "run_id":run_id,"step":4,"agent":"HumanApprovalGate",
            "timestamp":datetime.now().isoformat(),
            "decision":f"{len(approved_items)} auto-executed | {len(warned_items)} pending review | {len(blocked_items)} blocked",
            "status":"AWAITING_HUMAN_INPUT" if warned_items else "COMPLETED"
        })

        audit_out = {"run_id":run_id,"total_steps":len(trail),"audit_trail":trail}
        with open("data/audit_log.json","w") as f:
            json.dump(audit_out, f)

        return {
            "status": "success",
            "run_id": run_id,
            "disruption": {
                "total_flags": disruption_out["total_flags"],
                "critical": disruption_out["critical"]
            },
            "ripple": {
                "scenarios": ripple_out["total_critical_scenarios"],
                "patients_at_risk": ripple_out["total_patients_at_risk"],
                "financial_impact_inr": ripple_out["total_financial_impact_inr"]
            },
            "compliance": {
                "approved": compliance_out["approved"],
                "warned": compliance_out["approved_with_warnings"],
                "blocked": compliance_out["blocked"]
            },
            "audit": {"run_id": run_id, "total_steps": len(trail)}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# READ ENDPOINTS
# ==============================================================================

@app.get("/summary")
def get_summary():
    disruption = load_json("data/disruption_signals.json")
    ripple     = load_json("data/ripple_model.json")
    compliance = load_json("data/compliance_results.json")
    audit      = load_json("data/audit_log.json")

    # Agent outputs (LLM reasoning panel)
    a1 = load_json_optional("data/agent1_output.json") or {}
    a2 = load_json_optional("data/agent2_output.json") or {}
    a3 = load_json_optional("data/agent3_output.json") or {}
    a4 = load_json_optional("data/agent4_output.json") or {}

    agent_outputs = {
        "disruption_analysis": (a1.get("analysis") or {}).get("analysis") if isinstance(a1.get("analysis"), dict) else "",
        "cascade_summary": (a2.get("analysis") or {}).get("cascade_summary") if isinstance(a2.get("analysis"), dict) else "",
        "compliance_summary": (a3.get("analysis") or {}).get("compliance_summary") if isinstance(a3.get("analysis"), dict) else "",
        "playbook_summary": (a4.get("playbook") or {}).get("playbook_summary") if isinstance(a4.get("playbook"), dict) else "",
    }

    critical_flags = int(disruption.get("critical", 0))
    total_flags = int(disruption.get("total_flags", 0))
    patients = int(ripple.get("total_patients_at_risk", 0))
    financial_inr = int(ripple.get("total_financial_impact_inr", 0))
    approved = int(compliance.get("approved", 0))
    warned = int(compliance.get("approved_with_warnings", 0))
    blocked = int(compliance.get("blocked", 0))

    run_id = audit.get("run_id", "—")

    metrics = {
        "run_id": run_id,
        "run_id_short": run_id[-6:] if isinstance(run_id, str) and len(run_id) > 6 else run_id,
        "critical_flags": critical_flags,
        "patients_at_risk": patients,
        "financial_impact": "₹{:.1f}Cr".format(financial_inr / 1e7) if financial_inr else "₹0",
	"financial_impact_inr": financial_inr,
        "actions_approved": approved,
        "pending_approvals": warned,
        "total_flags": total_flags,
        "total_scenarios": int(ripple.get("total_critical_scenarios", 0)),
        "flags_subtitle": f"From {total_flags:,} total flags",
        "scenarios_subtitle": f"Across {int(ripple.get('total_critical_scenarios', 0)):,} critical scenarios",
        "compliance_subtitle": f"{warned} pending · {blocked} blocked",
        "nav_disruptions_badge": str(critical_flags) if critical_flags else "—",
        "nav_compliance_badge": f"{warned} warned" if warned else "—",
        "nav_approvals_badge": f"{warned} pending" if warned else "—",
        "nav_agents_badge": "4 live" if total_flags else "—",
        "pipe_1_state": "done" if total_flags else "idle",
        "pipe_2_state": "done" if patients else "idle",
        "pipe_3_state": "done" if (approved + warned + blocked) else "idle",
        "pipe_4_state": "waiting" if warned else ("done" if (approved + blocked) else "idle"),
        "pipe_disruption_count": f"{total_flags:,} flags" if total_flags else "—",
        "pipe_ripple_count": f"{patients:,} at risk" if patients else "—",
        "pipe_compliance_count": f"{blocked} blocked" if blocked else (f"{approved} approved" if approved else "—"),
        "pipe_approval_count": f"{warned} pending" if warned else "none pending",
    }

    return {
        "run_id": run_id,
        "last_run": run_id,
        "metrics": metrics,
        "agent_outputs": agent_outputs,
        # Keep legacy fields for backward compatibility
        "disruption": {"total_flags": total_flags, "critical": critical_flags},
        "ripple": {"patients_at_risk": patients, "financial_impact_inr": financial_inr},
        "compliance": {"approved": approved, "warned": warned, "blocked": blocked},
    }

@app.get("/disruptions")
def get_disruptions():
    data = load_json("data/disruption_signals.json")
    return {
        "timestamp": data["timestamp"],
        "summary": {
            "total": data["total_flags"],
            "critical": data["critical"],
            "high": data["high"],
            "medium": data["medium"]
        },
        "flags": data["flags"]
    }

@app.get("/disruptions/critical")
def get_critical_disruptions():
    data = load_json("data/disruption_signals.json")
    return {"flags": [f for f in data["flags"] if f["severity"] == "CRITICAL"]}

@app.get("/ripple")
def get_ripple():
    return load_json("data/ripple_model.json")

@app.get("/compliance")
def get_compliance():
    return load_json("data/compliance_results.json")

@app.get("/compliance/blocked")
def get_blocked():
    data = load_json("data/compliance_results.json")
    return {"blocked": [r for r in data["results"] if r["compliance_status"] == "BLOCKED"]}

@app.get("/audit")
def get_audit():
    return load_json("data/audit_log.json")

# ==============================================================================
# HUMAN APPROVAL ACTIONS
# ==============================================================================

@app.post("/approve/{drug}/{hospital}")
def approve_action(drug: str, hospital: str):
    data = load_json("data/compliance_results.json")
    for r in data["results"]:
        if r["drug"].lower() == drug.lower() and r["hospital"].lower() == hospital.lower():
            r["compliance_status"] = "APPROVED_BY_HUMAN"
            r["approved_at"] = datetime.now().isoformat()
            with open("data/compliance_results.json","w") as f:
                json.dump(data, f, indent=2)
            return {"status": "approved", "drug": drug, "hospital": hospital}
    raise HTTPException(status_code=404, detail="Record not found")

@app.post("/block/{drug}/{hospital}")
def block_action(drug: str, hospital: str):
    data = load_json("data/compliance_results.json")
    for r in data["results"]:
        if r["drug"].lower() == drug.lower() and r["hospital"].lower() == hospital.lower():
            r["compliance_status"] = "BLOCKED_BY_HUMAN"
            r["blocked_at"] = datetime.now().isoformat()
            with open("data/compliance_results.json","w") as f:
                json.dump(data, f, indent=2)
            return {"status": "blocked", "drug": drug, "hospital": hospital}
    raise HTTPException(status_code=404, detail="Record not found")

# ==============================================================================
# DRUG QUERY ENDPOINT — conversational drug-specific supply chain intelligence
# ==============================================================================

class DrugQueryRequest(BaseModel):
    drug: str
    question: str = "Give me a full supply chain status report for this drug."

@app.post("/query")
def query_drug(req: DrugQueryRequest):
    """
    Ask a natural language question about a specific drug.
    Pulls all relevant data from disruption, ripple, compliance, and inventory
    files, then sends to LLM for a focused analysis.
    """
    drug_name = req.drug.strip()
    question  = req.question.strip()

    # ── 1. Load all data files ──────────────────────────────────────────────
    try:
        disruption = load_json("data/disruption_signals.json")
    except Exception:
        disruption = {"flags": []}
    try:
        ripple = load_json("data/ripple_model.json")
    except Exception:
        ripple = {"ripple_reports": []}
    try:
        compliance = load_json("data/compliance_results.json")
    except Exception:
        compliance = {"results": []}
    try:
        inventory_df = pd.read_csv("data/hospital_inventory.csv")
        inventory_records = inventory_df[
            inventory_df["drug"].str.lower() == drug_name.lower()
        ].to_dict(orient="records")
    except Exception:
        inventory_records = []
    try:
        weather = load_json("data/weather_risk.json")
    except Exception:
        weather = []

    # ── 2. Filter each dataset to the requested drug ────────────────────────
    drug_disruptions = [
        f for f in disruption.get("flags", [])
        if f.get("drug", "").lower() == drug_name.lower()
    ]
    drug_ripple = [
        r for r in ripple.get("ripple_reports", [])
        if r.get("drug", "").lower() == drug_name.lower()
    ]
    drug_compliance = [
        c for c in compliance.get("results", [])
        if c.get("drug", "").lower() == drug_name.lower()
    ]

    # ── 3. Check if we know this drug at all ────────────────────────────────
    if not drug_disruptions and not drug_ripple and not drug_compliance and not inventory_records:
        return {
            "drug": drug_name,
            "question": question,
            "found": False,
            "answer": f"No supply chain data found for '{drug_name}'. "
                      f"Please run /run-pipeline first, or check the spelling. "
                      f"Known drugs: Insulin, Heparin, Vancomycin, Epinephrine, Morphine, "
                      f"Cisplatin, Methotrexate, Dexamethasone, Amoxicillin, Furosemide."
        }

    # ── 4. Summarise key metrics for context ────────────────────────────────
    total_patients = sum(r.get("patients_at_risk", 0) for r in drug_ripple)
    total_financial = sum(r.get("financial_impact_inr", 0) for r in drug_ripple)
    hospitals_affected = list({r.get("hospital") for r in drug_ripple})
    most_urgent = min(drug_ripple, key=lambda x: x.get("days_remaining", 999), default=None)
    blocked_count = sum(1 for c in drug_compliance if c.get("compliance_status") == "BLOCKED")
    emergency_count = sum(1 for r in drug_ripple if r.get("recommended_action") == "EMERGENCY_REORDER")

    # ── 5. Build plain-text prompt (llama3 is unreliable with strict JSON) ──
    hospital_lines = "\n".join([
        f"  - {r['hospital']}: {r['days_remaining']} days left | {r['recommended_action']} | supplier: {r['current_supplier']} ({r['supplier_country']}) | risk: {r['supplier_risk_score']}"
        for r in sorted(drug_ripple, key=lambda x: x["days_remaining"])
    ])
    compliance_lines = "\n".join([
        f"  - {c['hospital']}: {c['compliance_status']} | violations: {c.get('violations',[])} | warnings: {c.get('warnings',[])}"
        for c in drug_compliance
    ])

    prompt = (
        f"You are a healthcare supply chain AI analyst for Indian hospitals.\n\n"
        f"A procurement officer has asked: \"{question}\"\n"
        f"Drug: {drug_name}\n\n"
        f"LIVE DATA:\n"
        f"- Hospitals affected: {', '.join(hospitals_affected)}\n"
        f"- Most urgent stockout: {most_urgent.get('hospital') if most_urgent else 'N/A'} in {most_urgent.get('days_remaining') if most_urgent else 'N/A'} days\n"
        f"- Total patients at risk: {total_patients}\n"
        f"- Financial exposure: Rs {total_financial:,}\n"
        f"- Emergency reorders needed: {emergency_count}\n"
        f"- Compliance blocked: {blocked_count}\n\n"
        f"HOSPITAL STOCK STATUS:\n{hospital_lines}\n\n"
        f"COMPLIANCE STATUS:\n{compliance_lines}\n\n"
        f"Respond using EXACTLY these 4 labeled sections:\n"
        f"ANSWER: [2-3 sentence direct answer]\n"
        f"CRITICAL ISSUES:\n- [issue 1]\n- [issue 2]\n"
        f"IMMEDIATE ACTIONS:\n1. [step 1]\n2. [step 2]\n3. [step 3]\n"
        f"COMPLIANCE BLOCKERS:\n- [blocker or NONE]"
    )

    # ── 6. Call LLM and parse plain-text sections into structured fields ─────
    import re
    result = ask_llm(prompt)

    def extract_section(text, label):
        pattern = label + r":\s*(.+?)(?=\n[A-Z][A-Z ]+:|\Z)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def extract_list(text, label):
        section = extract_section(text, label)
        items = re.split(r"\n[-]|\n\d+[.]", section)
        return [i.strip() for i in items if i.strip()]

    severity = "CRITICAL" if emergency_count > 0 else "HIGH" if total_patients > 200 else "MEDIUM"

    parsed = {
        "direct_answer":       extract_section(result, "ANSWER") or result[:400],
        "severity_level":      severity,
        "critical_issues":     extract_list(result, "CRITICAL ISSUES"),
        "immediate_actions":   extract_list(result, "IMMEDIATE ACTIONS"),
        "compliance_blockers": extract_list(result, "COMPLIANCE BLOCKERS"),
        "hospital_status": [
            {
                "hospital":           r["hospital"],
                "days_remaining":     r["days_remaining"],
                "recommended_action": r["recommended_action"],
                "stockout_date":      r.get("stockout_date"),
                "compliance_status":  next(
                    (c["compliance_status"] for c in drug_compliance if c["hospital"] == r["hospital"]),
                    "UNKNOWN"
                )
            }
            for r in sorted(drug_ripple, key=lambda x: x["days_remaining"])
        ],
        "cold_chain_warning": next(
            (f"{w['city']}: {w['cold_chain_risk']} risk ({w['max_temp_72h']}C)"
             for w in weather if w["cold_chain_risk"] in ["HIGH", "CRITICAL"]
             and any(w["city"] in h for h in hospitals_affected)),
            None
        )
    }


    return {
        "drug": drug_name,
        "question": question,
        "found": True,
        "data_summary": {
            "disruption_flags": len(drug_disruptions),
            "hospitals_affected": hospitals_affected,
            "total_patients_at_risk": total_patients,
            "total_financial_impact_inr": total_financial,
            "emergency_reorders_needed": emergency_count,
            "compliance_blocked": blocked_count,
            "most_urgent_stockout_days": most_urgent.get("days_remaining") if most_urgent else None,
        },
        "analysis": parsed,
        "timestamp": datetime.now().isoformat()
    }


# ==============================================================================
# LLM AGENTS
# ==============================================================================

@app.post("/agent/disruption")
def agent_disruption():
    data = load_json("data/disruption_signals.json")
    top_critical = [f for f in data["flags"] if f["severity"] == "CRITICAL"][:5]

    prompt = f"""You are a healthcare supply chain AI agent.

You have detected the following critical drug supply disruptions in Indian hospitals:

{json.dumps(top_critical, indent=2)}

Your job:
1. Identify which disruptions are most urgent and why
2. Explain the clinical risk each poses to patients
3. Rank them by priority (1 = most urgent)
4. Give a one-line reasoning for each ranking

Respond in this exact JSON format:
{{
  "analysis": "2-3 sentence overall situation summary",
  "ranked_disruptions": [
    {{
      "rank": 1,
      "drug": "drug name",
      "hospital": "hospital name",
      "urgency_reason": "why this is most urgent",
      "clinical_risk": "what happens to patients if not resolved"
    }}
  ]
}}

Respond with JSON only. No extra text."""

    result = ask_llm(prompt)
    try:
        parsed = json.loads(result)
    except Exception:
        parsed = {"raw_response": result}

    output = {
        "agent": "DisruptionDetectionAgent",
        "timestamp": datetime.now().isoformat(),
        "llm_model": MODEL,
        "input_flags": len(top_critical),
        "analysis": parsed
    }
    with open("data/agent1_output.json", "w") as f:
        json.dump(output, f, indent=2)
    return output

@app.post("/agent/ripple")
def agent_ripple():
    disruption = load_json("data/agent1_output.json")
    ripple     = load_json("data/ripple_model.json")
    top_reports = ripple["ripple_reports"][:5]

    prompt = f"""You are a healthcare supply chain ripple modeling AI agent.

The disruption detection agent has flagged these critical scenarios:
{json.dumps(disruption["analysis"], indent=2)}

Here is the quantitative ripple model for each scenario:
{json.dumps(top_reports, indent=2)}

Your job:
1. Explain why each disruption will cascade beyond the immediate hospital
2. Identify which hospitals face the worst compounding risk
3. Estimate the realistic patient harm if no action is taken in 48 hours

Respond in this exact JSON format:
{{
  "cascade_summary": "2-3 sentence summary of how disruptions compound",
  "high_risk_scenarios": [
    {{
      "drug": "drug name",
      "hospital": "hospital name",
      "cascade_reason": "why this cascades",
      "48hr_patient_harm": "specific harm if unresolved in 48 hours",
      "financial_exposure_inr": 0
    }}
  ]
}}

Respond with JSON only. No extra text."""

    result = ask_llm(prompt)
    try:
        parsed = json.loads(result)
    except Exception:
        parsed = {"raw_response": result}

    output = {
        "agent": "RippleModelingAgent",
        "timestamp": datetime.now().isoformat(),
        "llm_model": MODEL,
        "analysis": parsed
    }
    with open("data/agent2_output.json", "w") as f:
        json.dump(output, f, indent=2)
    return output

@app.post("/agent/compliance")
def agent_compliance():
    ripple     = load_json("data/agent2_output.json")
    compliance = load_json("data/compliance_results.json")
    blocked    = [r for r in compliance["results"] if r["compliance_status"] == "BLOCKED"][:5]

    prompt = f"""You are a healthcare procurement compliance AI agent for India.

The following reorder actions have been BLOCKED by the rules engine:
{json.dumps(blocked, indent=2)}

Applicable regulatory frameworks:
- NDPS Act 1985 (controlled substances like Morphine)
- CDSCO procurement guidelines
- ICD-10 coding requirements
- Dual supplier mandate for emergency procurement

Your job:
1. For each blocked action, explain the violation in plain English
2. Suggest the exact steps to resolve the violation and get it approved
3. Flag any that cannot be resolved without government intervention

Respond in this exact JSON format:
{{
  "compliance_summary": "2-3 sentence overall compliance situation",
  "blocked_resolutions": [
    {{
      "drug": "drug name",
      "hospital": "hospital name",
      "violation_explanation": "plain English explanation of why it was blocked",
      "resolution_steps": ["step 1", "step 2", "step 3"],
      "needs_govt_intervention": false,
      "estimated_resolution_hours": 24
    }}
  ]
}}

Respond with JSON only. No extra text."""

    result = ask_llm(prompt)
    try:
        parsed = json.loads(result)
    except Exception:
        parsed = {"raw_response": result}

    output = {
        "agent": "ComplianceGuardrailAgent",
        "timestamp": datetime.now().isoformat(),
        "llm_model": MODEL,
        "analysis": parsed
    }
    with open("data/agent3_output.json", "w") as f:
        json.dump(output, f, indent=2)
    return output

@app.post("/agent/playbook")
def agent_playbook():
    disruption = load_json("data/agent1_output.json")
    ripple     = load_json("data/agent2_output.json")
    compliance = load_json("data/agent3_output.json")

    prompt = f"""You are a healthcare supply chain playbook AI agent.

You have received intelligence from 3 upstream agents:

DISRUPTION ANALYSIS:
{json.dumps(disruption["analysis"], indent=2)}

RIPPLE/CASCADE ANALYSIS:
{json.dumps(ripple["analysis"], indent=2)}

COMPLIANCE RESOLUTION GUIDE:
{json.dumps(compliance["analysis"], indent=2)}

Your job:
Generate a prioritized actionable playbook for the procurement team to execute RIGHT NOW.
Be specific - name the drugs, hospitals, suppliers, and exact steps.

Respond in this exact JSON format:
{{
  "playbook_summary": "2-3 sentence executive summary for the procurement head",
  "priority_actions": [
    {{
      "priority": 1,
      "action_type": "EMERGENCY_REORDER",
      "drug": "drug name",
      "hospital": "hospital name",
      "action": "exact action to take",
      "responsible_team": "who does this",
      "deadline": "within X hours",
      "success_criteria": "how you know it is done"
    }}
  ],
  "escalation_required": false,
  "estimated_total_resolution_hours": 48
}}

Respond with JSON only. No extra text."""

    result = ask_llm(prompt)
    try:
        parsed = json.loads(result)
    except Exception:
        parsed = {"raw_response": result}

    output = {
        "agent": "PlaybookGeneratorAgent",
        "timestamp": datetime.now().isoformat(),
        "llm_model": MODEL,
        "playbook": parsed
    }
    with open("data/agent4_output.json", "w") as f:
        json.dump(output, f, indent=2)
    return output

@app.post("/agent/run-all")
def run_all_agents():
    a1 = agent_disruption()
    a2 = agent_ripple()
    a3 = agent_compliance()
    a4 = agent_playbook()
    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "agents_run": 4,
        "disruption_agent":  a1["analysis"].get("analysis", ""),
        "ripple_agent":      a2["analysis"].get("cascade_summary", ""),
        "compliance_agent":  a3["analysis"].get("compliance_summary", ""),
        "playbook_summary":  a4["playbook"].get("playbook_summary", "")
    }
