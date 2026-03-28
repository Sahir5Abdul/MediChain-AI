# MediChain AI — Healthcare Supply Chain Intelligence

> **ET AI Hackathon 2026 · Round 2 · Problem Statement 5 — Domain-Specialized AI Agents with Compliance Guardrails**

MediChain is a multi-agent AI system that detects drug supply disruptions across Indian hospitals in real time, models their downstream patient and financial impact, enforces regulatory compliance guardrails, and generates actionable procurement playbooks — with a fully auditable decision trail at every step.

---

## The Problem

Indian hospitals face critical drug shortages with no early warning system. Procurement teams react after stockouts happen. Compliance checks are manual. There is no cross-hospital visibility, no supplier risk scoring, and no automated corrective action.

MediChain solves this with four specialized AI agents working in sequence — from signal detection to actionable playbook — orchestrated through n8n and served via a FastAPI backend with a live dashboard.

---

## Architecture

```
OpenFDA API (shortages + recalls)
        +
Synthetic Hospital Inventory          ──►  Disruption Detection Agent
        +                                           │
Weather / Cold Chain Data                           ▼
                                        Ripple Modeling Agent
                                                    │
                                                    ▼
                                        Compliance Guardrail Agent
                                         (ICD-10 · NDPS · CDSCO)
                                                    │
                                                    ▼
                                        Playbook Generator Agent
                                                    │
                                          ┌─────────┴──────────┐
                                          ▼                     ▼
                                   Auto-Execute          Human Approval Gate
                                          │                     │
                                          └─────────┬───────────┘
                                                    ▼
                                          Immutable Audit Log
                                                    │
                                                    ▼
                                         React Dashboard (live)
```

**Orchestration:** n8n workflow triggered via webhook → calls each agent sequentially via FastAPI endpoints → routes to human approval gate based on compliance outcome.

**LLM Runtime:** Llama3 via Ollama running locally — powers all four agents with natural language reasoning grounded in real data.

---

## Agent Descriptions

### Agent 1 — Disruption Detection Agent
- Sources: OpenFDA drug shortages (500 records), FDA enforcement/recalls (500 records), hospital inventory (50 rows across 5 hospitals), weather cold chain risk
- Matches FDA shortage signals against hospital inventory by drug name
- Flags cold chain risk for temperature-sensitive drugs (Insulin, Cisplatin, Methotrexate) based on 72-hour temperature forecasts
- Detects low stock conditions independently of FDA data
- Output: 2,062 flags (1,213 critical, 479 high, 370 medium) with severity classification
- LLM layer: ranks top disruptions by clinical urgency with natural language reasoning

### Agent 2 — Ripple Modeling Agent
- Input: critical flags from Agent 1
- Models stockout date, patients at risk per drug per hospital, financial impact at 2.5x emergency procurement premium
- Scores supplier country concentration risk (China: 0.85, USA: 0.70, Germany: 0.50, India: 0.20)
- Recommends action type: EMERGENCY_REORDER / EXPEDITE_REORDER / SCHEDULE_REORDER
- Output: 27 unique drug × hospital scenarios, 7,522 patients at risk, ₹5.2 crore financial exposure
- LLM layer: explains cascade effects and 48-hour patient harm if unresolved

### Agent 3 — Compliance Guardrail Agent
- Runs inline on every proposed reorder action before execution
- Checks: ICD-10 coding requirements, NDPS Act 1985 (controlled substances — Morphine), CDSCO emergency procurement approval, dual supplier mandate, supplier country concentration limits
- Output: APPROVED / APPROVED_WITH_WARNINGS / BLOCKED per action, with cited rule references
- LLM layer: explains violations in plain English and generates resolution steps
- Results: 11 approved, 7 warned (sent for human review), 9 blocked

### Agent 4 — Playbook Generator Agent
- Synthesizes outputs from all three upstream agents
- Generates a prioritized, step-by-step corrective action plan per hospital
- Names specific drugs, alternative suppliers, responsible teams, deadlines, and success criteria
- Output: executive summary + ranked priority actions with estimated resolution time
- LLM layer: full natural language playbook written for procurement team consumption

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Runtime | Llama3 via Ollama (local) |
| Backend | FastAPI + Uvicorn |
| Orchestration | n8n (webhook-triggered workflow) |
| Data Sources | OpenFDA API, synthetic hospital inventory, Open-Meteo weather |
| Frontend | Vanilla HTML/CSS/JS dashboard |
| Tunneling | ngrok |
| Language | Python 3.11+ |

---

## Project Structure

```
MediChain-AI/
├── main.py                                    # FastAPI backend + all 4 LLM agents
├── pipeline.py                                # Data ingestion pipeline
├── medichain_dashboard.html                   # Live frontend dashboard
├── healthcare_supply_chain_workflow.json      # n8n orchestration workflow
├── requirements.txt                           # Python dependencies
└── README.md
```

> Note: The `data/` folder is excluded from the repo. It is generated at runtime by hitting `POST /run-pipeline`.

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- Node.js 18+ (for n8n)
- [Ollama](https://ollama.ai) installed locally
- ngrok account (free tier works)

### 1. Clone the repo
```bash
git clone https://github.com/Sahir5Abdul/MediChain-AI.git
cd MediChain-AI
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull the LLM model
```bash
ollama pull llama3
```

### 4. Install and start n8n
```bash
npm install -g n8n
n8n start
```

### 5. Import the n8n workflow
- Open `http://localhost:5678`
- Go to Workflows → Import from file
- Select `healthcare_supply_chain_workflow.json`
- Update the URL placeholders to your ngrok URL
- Activate the workflow

### 6. Start the FastAPI backend
```bash
uvicorn main:app --reload --port 8000
```

### 7. Start ngrok tunnel
```bash
ngrok http 8000
```

### 8. Open the dashboard
Open `medichain_dashboard.html` in your browser. Login with:
- Admin: `admin / admin123`
- Analyst: `analyst1 / pass123`
- Reviewer: `reviewer1 / pass123`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/summary` | Pipeline summary stats |
| POST | `/run-pipeline` | Run full data pipeline |
| GET | `/disruptions` | All disruption flags |
| GET | `/disruptions/critical` | Critical flags only |
| GET | `/ripple` | Ripple model + financial impact |
| GET | `/compliance` | Compliance results |
| GET | `/compliance/blocked` | Blocked actions |
| GET | `/audit` | Full audit trail |
| POST | `/agent/disruption` | Run disruption detection agent (LLM) |
| POST | `/agent/ripple` | Run ripple modeling agent (LLM) |
| POST | `/agent/compliance` | Run compliance guardrail agent (LLM) |
| POST | `/agent/playbook` | Run playbook generator agent (LLM) |
| POST | `/agent/run-all` | Run all 4 LLM agents in sequence |
| POST | `/approve/{drug}/{hospital}` | Human approve action |
| POST | `/block/{drug}/{hospital}` | Human block action |

Interactive API docs available at `http://localhost:8000/docs`

---

## Key Metrics (Live Run)

| Metric | Value |
|---|---|
| FDA records processed | 1,000 (500 shortages + 500 recalls) |
| Total disruption flags | 2,062 |
| Critical flags | 1,213 |
| Hospitals monitored | 5 |
| Drugs tracked | 10 |
| Patients at risk | 7,522 |
| Financial exposure | ₹5.2 crore |
| Compliance checks | 27 |
| Actions auto-approved | 11 |
| Actions pending human review | 7 |
| Actions blocked | 9 |
| Audit trail steps | 30 |

---

## Compliance Frameworks Enforced

- **ICD-10-CM** — Drug coding and classification per procedure
- **NDPS Act 1985** — Narcotic Drugs and Psychotropic Substances Act (Morphine flagged)
- **CDSCO Guidelines** — Central Drugs Standard Control Organisation emergency procurement rules
- **Dual Supplier Mandate** — Emergency reorders must have at least 2 alternative suppliers
- **Supplier Country Concentration** — Risk thresholds per action type (Emergency: 0.6, Expedite: 0.7, Schedule: 0.85)

---

## n8n Workflow

The workflow has 12 nodes:

```
Webhook Trigger (POST /webhook/run-agents)
        ↓
Run Data Pipeline
        ↓
Agent 1 — Disruption Detection (LLM)
        ↓
Agent 2 — Ripple Modeling (LLM)
        ↓
Agent 3 — Compliance Guardrail (LLM)
        ↓
Agent 4 — Playbook Generator (LLM)
        ↓
Check Human Approval Required
        ↓              ↓
   Yes → Notify    No → Auto Complete
        ↓              ↓
     Fetch Final Summary
        ↓
  Build Final Report
```

---

## Team

Built for the ET AI Hackathon 2026 — Round 2.
Problem Statement 5: Domain-Specialized AI Agents with Compliance Guardrails.

---

## License

MIT
