#!/usr/bin/env bash
# Smoke test for the HireMe ATS stack. Run this AFTER `docker compose up -d`
# and after the app has finished loading models (check `docker compose logs -f app`
# for "All pipeline models loaded." before running this).
#
# Usage:
#   ./scripts/smoke_test.sh                     # health + job creation only
#   ./scripts/smoke_test.sh path/to/resume.pdf   # also tests upload + matching

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
RESUME_FILE="${1:-}"

pp() { python3 -m json.tool 2>/dev/null || cat; }

echo "== 1. Health check =="
curl -sf "$BASE_URL/api/v1/" | pp
echo

echo "== 2. Create a job (GLiNER extraction + embedding) =="
JOB_RESPONSE=$(curl -sf -X POST "$BASE_URL/api/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{"description": "We are looking for a Senior Python Backend Engineer with 5+ years of experience, strong skills in FastAPI, PostgreSQL, and Docker. A Bachelor degree in Computer Science is required. Fluent English required."}')
echo "$JOB_RESPONSE" | pp
JOB_ID=$(echo "$JOB_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))")
echo "job_id: $JOB_ID"
echo

echo "== 3. List jobs =="
curl -sf "$BASE_URL/api/v1/jobs" | pp
echo

if [[ -z "$RESUME_FILE" ]]; then
  echo "No resume file given — skipping upload + match test."
  echo "Re-run as: ./scripts/smoke_test.sh path/to/resume.pdf"
  exit 0
fi

if [[ ! -f "$RESUME_FILE" ]]; then
  echo "File not found: $RESUME_FILE" >&2
  exit 1
fi

echo "== 4. Upload a candidate resume =="
CANDIDATE_RESPONSE=$(curl -sf -X POST "$BASE_URL/api/v1/candidates/upload" \
  -F "file=@${RESUME_FILE}")
echo "$CANDIDATE_RESPONSE" | pp
echo

if [[ -z "$JOB_ID" || "$JOB_ID" == "None" ]]; then
  echo "No job_id from step 2 — can't test matching." >&2
  exit 1
fi

echo "== 5. Cosine-similarity shortlist (no rerank) =="
curl -sf "$BASE_URL/api/v1/jobs/${JOB_ID}/shortlist?limit=50" | pp
echo

echo "== 6. Queue matching (Celery) =="
MATCH_RESPONSE=$(curl -sf -X POST "$BASE_URL/api/v1/jobs/${JOB_ID}/match" \
  -H "Content-Type: application/json" \
  -d '{"shortlist_limit": 50, "top_n": 5}')
echo "$MATCH_RESPONSE" | pp
TASK_ID=$(echo "$MATCH_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('task_id',''))")
echo "task_id: $TASK_ID"
echo

echo "== 7. Poll for match result (up to 60s) =="
for i in $(seq 1 20); do
  RESULT=$(curl -sf "$BASE_URL/api/v1/jobs/tasks/${TASK_ID}")
  STATUS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
  echo "  [$i] status: $STATUS"
  if [[ "$STATUS" == "done" || "$STATUS" == "failed" ]]; then
    echo "$RESULT" | pp
    break
  fi
  sleep 3
done

echo
echo "Smoke test complete."
