// HireMe HR Interview Console
// Talks to two services directly from the browser:
//   HR_BASE  -> the HR interview service (sessions, questions, answers)
//   ATS_BASE -> only used to preview a job before creating a session from it
//
// This mirrors the real request flow described in the platform README:
// the browser (or HR) never talks to Postgres, and the HR service's own
// ATS calls are reproduced here 1:1 so this console behaves exactly like
// the API does (same 404 / 503 semantics, same field names).

const el = (id) => document.getElementById(id);

const state = {
  hrBase: el('hrUrl').value.trim(),
  atsBase: el('atsUrl').value.trim(),
  session: null,       // { id, role, skills, job_id }
  currentQuestion: null, // { question, category }
  history: [],          // list of AnswerResponse-shaped objects
  awaitingAnswer: false, // true when a question is active but not yet answered
  recorder: null,
  recordedChunks: [],
  recordingSeconds: 0,
  recordTimer: null,
  audioCtx: null,
  analyser: null,
  waveTimer: null,
};

// ---------------------------------------------------------------- helpers

function showError(msg) {
  const box = el('globalError');
  box.textContent = msg;
  box.classList.add('show');
  clearTimeout(showError._t);
  showError._t = setTimeout(() => box.classList.remove('show'), 7000);
}

function clearError() { el('globalError').classList.remove('show'); }

async function apiCall(url, opts = {}) {
  let res;
  try {
    res = await fetch(url, opts);
  } catch (e) {
    throw { network: true, message: `Could not reach ${url.split('/')[2]}. Is the service running and reachable from your browser?` };
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || body.signal || JSON.stringify(body);
    } catch (_) {}
    throw { network: false, status: res.status, message: detail };
  }
  const ct = res.headers.get('content-type') || '';
  return ct.includes('application/json') ? res.json() : null;
}

function pingDot(dotEl, base, healthPath) {
  fetch(base + healthPath).then(r => {
    dotEl.className = 'dot ' + (r.ok ? 'ok' : 'bad');
  }).catch(() => { dotEl.className = 'dot bad'; });
}

function refreshPings() {
  state.hrBase = el('hrUrl').value.trim().replace(/\/$/, '');
  state.atsBase = el('atsUrl').value.trim().replace(/\/$/, '');
  pingDot(el('hrDot'), state.hrBase, '/health');
  // ATS has no /health in the router set shown, root "/" returns {status:"running"}
  pingDot(el('atsDot'), state.atsBase, '/');
}
el('hrUrl').addEventListener('change', refreshPings);
el('atsUrl').addEventListener('change', refreshPings);
refreshPings();
setInterval(refreshPings, 15000);

// ---------------------------------------------------------------- tabs

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    el('tab-job').style.display = tab.dataset.tab === 'job' ? 'block' : 'none';
    el('tab-manual').style.display = tab.dataset.tab === 'manual' ? 'block' : 'none';
  });
});

// ---------------------------------------------------------------- job lookup

let lastLookedUpJob = null;

el('lookupJobBtn').addEventListener('click', async () => {
  clearError();
  const jobId = el('jobIdInput').value.trim();
  if (!jobId) { showError('Enter a job id first.'); return; }
  el('lookupJobBtn').disabled = true;
  el('lookupJobBtn').textContent = 'Looking up…';
  try {
    const job = await apiCall(`${state.atsBase}/api/v1/jobs/${encodeURIComponent(jobId)}/summary`);
    lastLookedUpJob = job;
    el('jobPreviewTitle').textContent = job.job_title || '(no title extracted)';
    el('jobPreviewSkills').innerHTML = (job.hard_skills || [])
      .map(s => `<span class="chip">${escapeHtml(s)}</span>`).join('') || '<span class="hint">No hard skills extracted for this job.</span>';
    el('jobPreview').classList.add('show');
    el('startFromJobBtn').disabled = false;
  } catch (e) {
    lastLookedUpJob = null;
    el('jobPreview').classList.remove('show');
    el('startFromJobBtn').disabled = true;
    if (e.status === 404) showError(`No job with id "${jobId}" exists in the ATS.`);
    else showError(e.message || 'Lookup failed.');
  } finally {
    el('lookupJobBtn').disabled = false;
    el('lookupJobBtn').textContent = 'Look up job';
  }
});

el('startFromJobBtn').addEventListener('click', async () => {
  clearError();
  const jobId = el('jobIdInput').value.trim();
  if (!jobId) return;
  el('startFromJobBtn').disabled = true;
  el('startFromJobBtn').textContent = 'Starting…';
  try {
    // POST {HR}/sessions/from-job/{job_id} — HR fetches the summary itself
    // server-side; we already previewed it client-side above for the user.
    const sess = await apiCall(`${state.hrBase}/sessions/from-job/${encodeURIComponent(jobId)}`, { method: 'POST' });
    enterInterview(sess);
  } catch (e) {
    if (e.status === 404) showError('ATS returned 404 — job not found (it may have been created against a different ATS than the one configured above).');
    else if (e.status === 503) showError('HR could not reach the ATS service (503) after retrying. Try again shortly.');
    else showError(e.message || 'Could not start session.');
  } finally {
    el('startFromJobBtn').disabled = false;
    el('startFromJobBtn').textContent = 'Start session from this job';
  }
});

el('startManualBtn').addEventListener('click', async () => {
  clearError();
  const role = el('manualRole').value.trim();
  const skills = el('manualSkills').value.trim();
  if (!role || !skills) { showError('Both role and skills are required.'); return; }
  el('startManualBtn').disabled = true;
  try {
    const sess = await apiCall(`${state.hrBase}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role, skills }),
    });
    enterInterview(sess);
  } catch (e) {
    showError(e.message || 'Could not start session.');
  } finally {
    el('startManualBtn').disabled = false;
  }
});

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

// ---------------------------------------------------------------- session lifecycle

function enterInterview(sess) {
  state.session = sess;
  state.currentQuestion = null;
  state.history = [];
  el('startPanel').style.display = 'none';
  el('interviewPanel').style.display = 'block';
  el('sessRole').textContent = sess.role;
  el('sessId').textContent = 'session ' + sess.id.slice(0, 8);
  if (sess.job_id) {
    el('sessJobChip').style.display = 'inline-flex';
    el('sessJobChip').textContent = 'from ATS job ' + sess.job_id.slice(0, 10);
  } else {
    el('sessJobChip').style.display = 'none';
  }
  el('sessSkillsLine').style.display = 'block';
  el('sessSkillsLine').textContent = 'Skills in play: ' + sess.skills;
  el('questionArea').innerHTML = '<div class="question-placeholder">No question generated yet.</div>';
  el('historyList').innerHTML = '<div class="history-empty">No answers submitted yet.</div>';
  disableRecorder(true);
}

el('viewSkillsBtn').addEventListener('click', () => {
  const line = el('sessSkillsLine');
  line.style.display = line.style.display === 'none' ? 'block' : 'none';
});

el('endSessionBtn').addEventListener('click', async () => {
  if (!state.session) return;
  if (!confirm('End this session? This deletes it from the HR service\u2019s in-memory store — the transcript/evaluations will be gone unless you\u2019ve already viewed the summary.')) return;
  try {
    await apiCall(`${state.hrBase}/sessions/${state.session.id}`, { method: 'DELETE' });
  } catch (e) { /* even if delete fails, still reset the UI locally */ }
  resetToStart();
});

function resetToStart() {
  state.session = null;
  state.currentQuestion = null;
  state.history = [];
  el('interviewPanel').style.display = 'none';
  el('startPanel').style.display = 'block';
  el('jobPreview').classList.remove('show');
  el('startFromJobBtn').disabled = true;
  el('jobIdInput').value = '';
  el('manualRole').value = '';
  el('manualSkills').value = '';
}

// ---------------------------------------------------------------- question generation

el('genQuestionBtn').addEventListener('click', async () => {
  clearError();
  if (state.awaitingAnswer) {
    showError('Answer the current question before generating a new one.');
    return;
  }
  el('genQuestionBtn').disabled = true;
  el('genQuestionBtn').textContent = 'Generating…';
  el('questionArea').innerHTML = '<div class="question-placeholder">Generating question via the local LLM — this can take a few seconds…</div>';
  try {
    const q = await apiCall(`${state.hrBase}/sessions/${state.session.id}/questions`, { method: 'POST' });
    state.currentQuestion = q;
    state.awaitingAnswer = true;
    renderQuestion(q);
    disableRecorder(false);
  } catch (e) {
    if (e.status === 404) showError('Session not found — it may have expired or the HR service restarted (sessions are in-memory).');
    else showError(e.message || 'Could not generate a question.');
    el('questionArea').innerHTML = '<div class="question-placeholder">No question generated yet.</div>';
  } finally {
    el('genQuestionBtn').disabled = false;
    el('genQuestionBtn').textContent = 'Generate question';
  }
});

function renderQuestion(q) {
  el('questionArea').innerHTML = `
    <div class="chip cat-${q.category}">${q.category.replace('_', ' ')}</div>
    <div class="question-text">${escapeHtml(q.question)}</div>
  `;
}

// ---------------------------------------------------------------- recording

const WAVE_BARS = 24;
(function buildWave() {
  const wave = el('wave');
  for (let i = 0; i < WAVE_BARS; i++) {
    const b = document.createElement('div');
    b.className = 'bar';
    wave.appendChild(b);
  }
})();

function disableRecorder(disabled) {
  el('recBtn').style.pointerEvents = disabled ? 'none' : 'auto';
  el('recBtn').style.opacity = disabled ? .4 : 1;
  el('fileInput').disabled = disabled;
}

el('recBtn').addEventListener('click', () => {
  if (!state.recorder || state.recorder.state !== 'recording') startRecording();
  else stopRecording();
});

async function startRecording() {
  clearError();
  if (!state.currentQuestion) { showError('Generate a question before recording an answer.'); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    showError('Microphone access was denied or is unavailable — use the file upload option instead.');
    return;
  }

  const mimeCandidates = ['audio/webm', 'audio/ogg', 'audio/mp4'];
  const mime = mimeCandidates.find(m => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || '';
  state.recorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
  state.recordedChunks = [];
  state.recorder.ondataavailable = (e) => { if (e.data.size > 0) state.recordedChunks.push(e.data); };
  state.recorder.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    const blobType = state.recorder.mimeType || 'audio/webm';
    const blob = new Blob(state.recordedChunks, { type: blobType });
    const ext = blobType.includes('ogg') ? 'ogg' : blobType.includes('mp4') ? 'm4a' : 'webm';
    submitAnswer(blob, `answer.${ext}`);
  };
  state.recorder.start();

  el('recBtn').classList.add('recording');
  el('wave').classList.add('live');
  state.recordingSeconds = 0;
  el('recTime').textContent = '0:00';
  state.recordTimer = setInterval(() => {
    state.recordingSeconds++;
    const m = Math.floor(state.recordingSeconds / 60), s = state.recordingSeconds % 60;
    el('recTime').textContent = `${m}:${String(s).padStart(2, '0')}`;
  }, 1000);

  // live waveform via Web Audio analyser
  state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = state.audioCtx.createMediaStreamSource(stream);
  state.analyser = state.audioCtx.createAnalyser();
  state.analyser.fftSize = 64;
  source.connect(state.analyser);
  const data = new Uint8Array(state.analyser.frequencyBinCount);
  const bars = document.querySelectorAll('#wave .bar');
  state.waveTimer = setInterval(() => {
    state.analyser.getByteFrequencyData(data);
    bars.forEach((bar, i) => {
      const v = data[i % data.length] / 255;
      bar.style.height = Math.max(6, v * 30) + 'px';
    });
  }, 90);
}

function stopRecording() {
  if (state.recorder && state.recorder.state === 'recording') state.recorder.stop();
  el('recBtn').classList.remove('recording');
  el('wave').classList.remove('live');
  clearInterval(state.recordTimer);
  clearInterval(state.waveTimer);
  document.querySelectorAll('#wave .bar').forEach(b => b.style.height = '6px');
  if (state.audioCtx) { state.audioCtx.close(); state.audioCtx = null; }
}

el('fileInput').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  if (!state.currentQuestion) { showError('Generate a question before submitting an answer.'); e.target.value = ''; return; }
  submitAnswer(file, file.name);
  e.target.value = '';
});

// ---------------------------------------------------------------- submit + evaluate

async function submitAnswer(fileLike, filename) {
  clearError();
  el('evalStatus').innerHTML = `<div class="evaluating"><div class="spinner"></div>Transcribing &amp; evaluating the answer…</div>`;
  el('genQuestionBtn').disabled = true;

  const form = new FormData();
  form.append('file', fileLike, filename);

  try {
    const result = await apiCall(`${state.hrBase}/sessions/${state.session.id}/answers`, {
      method: 'POST',
      body: form,
    });
    state.awaitingAnswer = false;
    state.history.unshift(result);
    renderHistory();
    el('evalStatus').innerHTML = '';
    // Question answered — require a fresh "Generate question" before recording again.
    state.currentQuestion = null;
    el('questionArea').innerHTML = '<div class="question-placeholder">Answer recorded. Generate the next question when ready.</div>';
    disableRecorder(true);
  } catch (e) {
    el('evalStatus').innerHTML = '';
    if (e.status === 404) showError('Session not found on the HR service.');
    else showError(e.message || 'Could not submit the answer.');
  } finally {
    el('genQuestionBtn').disabled = false;
  }
}

function scoreRing(score) {
  const s = Math.max(0, Math.min(10, score || 0));
  const pct = s / 10;
  const r = 15, c = 2 * Math.PI * r;
  const color = s >= 7 ? 'var(--good)' : s >= 4 ? 'var(--warn)' : 'var(--danger)';
  return `
    <svg width="38" height="38" viewBox="0 0 38 38" class="score-ring">
      <circle cx="19" cy="19" r="${r}" fill="none" stroke="var(--border)" stroke-width="4"/>
      <circle cx="19" cy="19" r="${r}" fill="none" stroke="${color}" stroke-width="4"
        stroke-dasharray="${c}" stroke-dashoffset="${c * (1 - pct)}"
        stroke-linecap="round" transform="rotate(-90 19 19)"/>
      <text x="19" y="23" text-anchor="middle" font-family="var(--mono)" font-size="11" fill="${color}">${s}</text>
    </svg>`;
}

function renderResult(r) {
  const ev = r.evaluation || {};
  if (ev.status === 'error' || ev.status === 'invalid_answer') {
    return `
      <div class="result">
        <div class="result-head"><div class="result-q">${escapeHtml(r.question)}</div>${r.category ? `<span class="chip cat-${r.category}">${r.category.replace('_', ' ')}</span>` : ''}</div>
        <div class="flag">⚠ ${escapeHtml(ev.message || (ev.status === 'invalid_answer' ? 'Answer too short to evaluate.' : 'Something went wrong evaluating this answer.'))}</div>
      </div>`;
  }
  return `
    <div class="result">
      <div class="result-head">
        <div class="result-q">${escapeHtml(r.question)}</div>
        <div style="display:flex; align-items:center; gap:8px;">
          ${r.category ? `<span class="chip cat-${r.category}">${r.category.replace('_', ' ')}</span>` : ''}
          ${scoreRing(ev.score)}
        </div>
      </div>
      ${r.transcript ? `<div class="transcript">"${escapeHtml(r.transcript)}"</div>` : ''}
      <div class="feedback">${escapeHtml(ev.feedback || '')}</div>
    </div>`;
}

function renderHistory() {
  const list = el('historyList');
  if (state.history.length === 0) {
    list.innerHTML = '<div class="history-empty">No answers submitted yet.</div>';
    return;
  }
  list.innerHTML = state.history.map(renderResult).join('');
}

// ---------------------------------------------------------------- summary

el('finishBtn').addEventListener('click', async () => {
  if (!state.session) return;
  clearError();
  el('finishBtn').disabled = true;
  el('finishBtn').textContent = 'Loading…';
  try {
    const summary = await apiCall(`${state.hrBase}/sessions/${state.session.id}/summary`);
    el('finalScoreNum').textContent = summary.final_score;
    el('summaryTotal').textContent = `${summary.total_questions} question${summary.total_questions === 1 ? '' : 's'} asked`;
    el('summaryEvaluated').textContent = `${summary.evaluated} evaluated`;
    el('summaryList').innerHTML = summary.results.map(renderResult).join('') || '<div class="history-empty">No results recorded.</div>';
    el('summaryModal').classList.add('show');
  } catch (e) {
    if (e.status === 404) showError('Session not found — nothing to summarize.');
    else showError(e.message || 'Could not load the summary.');
  } finally {
    el('finishBtn').disabled = false;
    el('finishBtn').textContent = 'End interview & view summary';
  }
});

el('closeSummaryBtn').addEventListener('click', () => el('summaryModal').classList.remove('show'));
el('summaryModal').addEventListener('click', (e) => { if (e.target === el('summaryModal')) el('summaryModal').classList.remove('show'); });
