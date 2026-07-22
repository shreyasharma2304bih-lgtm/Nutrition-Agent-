# =============================================================================
# NutriWise AI – Personalized Nutrition Coach
# Built with Flask + IBM watsonx.ai Granite Models
# Multi-Agent Architecture: 4 Specialized Nutrition Agents
# =============================================================================

import os
import json
import urllib.request
import urllib.parse
import urllib.error
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# =============================================================================
# IBM watsonx.ai Configuration
# Credentials are read from environment variables for security.
# Set these before running:
#   WATSONX_API_KEY    – your IBM Cloud API key
#   WATSONX_PROJECT_ID – your watsonx.ai project ID
#   WATSONX_URL        – regional endpoint, e.g. https://us-south.ml.cloud.ibm.com
#
# NOTE: Uses the IBM watsonx.ai REST API directly via urllib (stdlib only),
#       so no heavy SDK / pandas dependency is required on Python 3.15+.
# =============================================================================

WATSONX_API_KEY    = os.environ.get("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")
WATSONX_URL        = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

# IBM IAM token endpoint
IAM_TOKEN_URL = "https://iam.cloud.ibm.com/identity/token"

_iam_token_cache = {"token": None, "expires_at": 0}


def _get_iam_token() -> str:
    """
    Obtain (or return cached) IBM IAM Bearer token from an API key.
    Tokens are valid for ~60 min; we re-fetch if within 5 min of expiry.
    """
    import time
    now = time.time()
    # Return cached token if still valid for > 5 minutes
    if _iam_token_cache["token"] and now < _iam_token_cache["expires_at"] - 300:
        return _iam_token_cache["token"]

    # --- IBM IAM token exchange (API key → Bearer token) ---
    data = urllib.parse.urlencode({
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": WATSONX_API_KEY,
    }).encode("utf-8")

    req = urllib.request.Request(
        IAM_TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    token = result["access_token"]
    expires_in = int(result.get("expires_in", 3600))
    _iam_token_cache["token"] = token
    _iam_token_cache["expires_at"] = now + expires_in
    return token


def generate_response(prompt: str) -> str:
    """
    Core reusable function – sends a prompt to the IBM watsonx.ai
    Granite model via the REST API and returns the generated text.

    All four agents call this single function — it is the only
    integration point with IBM watsonx.ai.

    REST endpoint: POST /ml/v1/text/generation
    Model: ibm/granite-3-8b-instruct
    """
    if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
        return (
            "⚠️ IBM watsonx.ai credentials are not configured. "
            "Please set WATSONX_API_KEY, WATSONX_PROJECT_ID, and WATSONX_URL "
            "as environment variables and restart the app."
        )
    try:
        token = _get_iam_token()

        # --- IBM watsonx.ai REST API call ---
        api_url = (
            f"{WATSONX_URL.rstrip('/')}/ml/v1/text/generation"
            f"?version=2023-05-29"
        )

        payload = json.dumps({
            "model_id": "ibm/granite-3-8b-instruct",   # IBM Granite Model
            "input": prompt,
            "parameters": {
                "max_new_tokens": 800,
                "min_new_tokens": 60,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            },
            "project_id": WATSONX_PROJECT_ID,
        }).encode("utf-8")

        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        generated = result["results"][0]["generated_text"]
        return generated.strip() if generated else "No response generated."

    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return f"❌ watsonx.ai HTTP {exc.code}: {body[:300]}"
    except Exception as exc:
        return f"❌ watsonx.ai error: {str(exc)}"


# =============================================================================
# AGENT 1 – Nutrition Knowledge Agent
# Answers general nutrition questions using IBM Granite.
# =============================================================================

def nutrition_knowledge_agent(question: str) -> str:
    """
    Routes a free-form nutrition question to IBM watsonx.ai Granite.
    Returns an educational, well-structured answer.
    """
    prompt = f"""You are a certified nutritionist and health educator.
Answer the following nutrition question in a clear, informative, and structured way.
Include practical examples, key nutrients, and evidence-based insights.
Keep the tone friendly and educational.

Question: {question}

Answer:"""
    return generate_response(prompt)


# =============================================================================
# AGENT 2 – Diet Planner Agent
# Generates a personalised full-day meal plan based on user profile.
# =============================================================================

def diet_planner_agent(age, gender, height, weight, diet_pref, activity, goal) -> str:
    """
    Accepts user demographics and fitness goals, then calls IBM watsonx.ai
    to generate a personalised meal plan with macro targets.
    """
    prompt = f"""You are an expert dietitian and meal planning specialist.
Create a detailed, personalised one-day meal plan for the following individual:

- Age: {age} years
- Gender: {gender}
- Height: {height} cm
- Weight: {weight} kg
- Dietary Preference: {diet_pref}
- Activity Level: {activity}
- Fitness Goal: {goal}

Please provide:
1. DAILY TARGETS: Estimated calorie target, protein (g), carbohydrates (g), fats (g)
2. BREAKFAST: 2–3 specific food items with portions
3. MID-MORNING SNACK: 1–2 items
4. LUNCH: 3–4 specific food items with portions
5. EVENING SNACK: 1–2 items
6. DINNER: 3–4 specific food items with portions
7. HYDRATION TIP
8. ONE KEY NUTRITION TIP for the stated goal

Format each section clearly with headings. Be specific with portion sizes.

Meal Plan:"""
    return generate_response(prompt)


# =============================================================================
# AGENT 3 – Health Advisory Agent
# Provides disease-specific dietary and lifestyle guidance.
# =============================================================================

def health_advisory_agent(conditions: list) -> str:
    """
    Receives a list of health conditions and calls IBM watsonx.ai Granite
    to produce tailored dietary recommendations and lifestyle advice.
    Always appends a medical disclaimer.
    """
    conditions_str = ", ".join(conditions) if conditions else "General Wellness"
    prompt = f"""You are a clinical nutrition advisor with expertise in therapeutic diets.
Provide comprehensive dietary and lifestyle recommendations for a person managing: {conditions_str}.

Structure your response as:

1. FOODS TO INCLUDE (with reasons)
2. FOODS TO AVOID (with reasons)
3. HEALTHY EATING HABITS
4. LIFESTYLE RECOMMENDATIONS
5. KEY NUTRITIONAL FOCUS AREAS

Be specific, practical, and evidence-based. Use bullet points for clarity.

Recommendations:"""

    response = generate_response(prompt)

    # Always append the mandatory medical disclaimer
    disclaimer = (
        "\n\n---\n"
        "⚕️ **Disclaimer:** This information is for educational purposes only. "
        "Please consult a qualified healthcare professional or registered dietitian "
        "before making significant changes to your diet or lifestyle, especially "
        "when managing medical conditions."
    )
    return response + disclaimer


# =============================================================================
# AGENT 4 – Meal Analysis Agent
# Analyses a user-described meal and provides nutritional feedback.
# =============================================================================

def meal_analysis_agent(meal_description: str) -> str:
    """
    Accepts a free-text description of meals eaten and sends it to
    IBM watsonx.ai Granite for nutritional quality analysis and suggestions.
    """
    prompt = f"""You are a nutrition analyst specialising in food quality assessment.
Analyse the following meal log and provide detailed nutritional feedback.

Meal Log:
{meal_description}

Provide your analysis in these sections:

1. OVERALL NUTRITIONAL QUALITY (score out of 10 with brief justification)
2. NUTRITIONAL STRENGTHS (what this meal does well)
3. NUTRITIONAL GAPS & DEFICIENCIES (what is missing or inadequate)
4. HEALTHIER ALTERNATIVES (specific swaps for less healthy items)
5. IMPROVEMENT RECOMMENDATIONS (practical, actionable suggestions)
6. ESTIMATED MACRO BALANCE (rough estimate: carbs/protein/fat ratio)

Be specific, constructive, and encouraging in your tone.

Analysis:"""
    return generate_response(prompt)


# =============================================================================
# AGENT ORCHESTRATOR
# Routes each incoming request to the correct specialised agent.
# =============================================================================

def orchestrate(agent_name: str, payload: dict) -> str:
    """
    Central orchestrator that dispatches requests to the correct agent.

    Supported agents:
      - 'nutrition_knowledge' → nutrition_knowledge_agent()
      - 'diet_planner'        → diet_planner_agent()
      - 'health_advisory'     → health_advisory_agent()
      - 'meal_analysis'       → meal_analysis_agent()
    """
    if agent_name == "nutrition_knowledge":
        return nutrition_knowledge_agent(payload.get("question", ""))

    elif agent_name == "diet_planner":
        return diet_planner_agent(
            age       = payload.get("age", "25"),
            gender    = payload.get("gender", "Male"),
            height    = payload.get("height", "170"),
            weight    = payload.get("weight", "70"),
            diet_pref = payload.get("diet_pref", "Vegetarian"),
            activity  = payload.get("activity", "Moderate"),
            goal      = payload.get("goal", "General Wellness"),
        )

    elif agent_name == "health_advisory":
        return health_advisory_agent(payload.get("conditions", []))

    elif agent_name == "meal_analysis":
        return meal_analysis_agent(payload.get("meal_description", ""))

    else:
        return "Unknown agent requested."


# =============================================================================
# HTML TEMPLATES (all inlined via render_template_string)
# =============================================================================

# --------------- Shared layout shell ----------------------------------------
BASE_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NutriWise AI – {{ page_title }}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css" rel="stylesheet"/>
  <style>
    :root {
      --primary:   #1a7f4b;
      --primary-d: #145f38;
      --accent:    #f0faf4;
      --sidebar-w: 260px;
    }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background:#f8fafb; color:#1a2233; }

    /* ── Sidebar ── */
    #sidebar {
      width: var(--sidebar-w); min-height:100vh; background:#0f3d24;
      position:fixed; top:0; left:0; z-index:1000;
      display:flex; flex-direction:column; padding-top:0;
      transition: transform .3s;
    }
    #sidebar .brand {
      background:#0a2918; padding:22px 20px 18px;
      border-bottom:1px solid rgba(255,255,255,.1);
    }
    #sidebar .brand h5 { color:#6debb4; margin:0; font-weight:700; font-size:1.05rem; }
    #sidebar .brand small { color:#a8d5bc; font-size:.75rem; }
    #sidebar .nav-link {
      color:#c8e6d8; padding:11px 22px; font-size:.9rem;
      display:flex; align-items:center; gap:10px; border-radius:0;
      transition:background .2s, color .2s;
    }
    #sidebar .nav-link:hover, #sidebar .nav-link.active {
      background:var(--primary); color:#fff;
    }
    #sidebar .nav-link i { font-size:1.05rem; width:20px; text-align:center; }
    #sidebar .nav-section {
      color:#7ab896; font-size:.7rem; font-weight:700;
      letter-spacing:.08em; text-transform:uppercase;
      padding:16px 22px 6px;
    }
    #sidebar .sidebar-footer {
      margin-top:auto; padding:16px 20px;
      border-top:1px solid rgba(255,255,255,.08);
      font-size:.72rem; color:#7ab896;
    }

    /* ── Main content ── */
    #main { margin-left:var(--sidebar-w); min-height:100vh; }
    .topbar {
      background:#fff; border-bottom:1px solid #e4eae6;
      padding:14px 28px; display:flex; align-items:center;
      gap:12px; position:sticky; top:0; z-index:900;
    }
    .topbar h6 { margin:0; font-weight:600; color:var(--primary-d); }
    .topbar .badge-ibm {
      background:#0f62fe; color:#fff; font-size:.65rem;
      font-weight:600; padding:3px 8px; border-radius:20px;
      letter-spacing:.04em;
    }
    .content-area { padding:28px; }

    /* ── Cards ── */
    .card { border:1px solid #dde8e2; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,.05); }
    .card-header { background:var(--accent); border-bottom:1px solid #dde8e2; font-weight:600; padding:14px 20px; }
    .agent-badge {
      display:inline-flex; align-items:center; gap:6px;
      background:#e6f4ed; color:var(--primary-d);
      font-size:.75rem; font-weight:600; padding:4px 12px; border-radius:20px;
    }

    /* ── Chat / response area ── */
    .response-box {
      background:#fff; border:1px solid #dde8e2; border-radius:10px;
      padding:22px; min-height:120px; white-space:pre-wrap;
      line-height:1.75; font-size:.9rem; color:#1a2233;
    }
    .response-box.empty { color:#9aadb6; font-style:italic; }

    /* ── Buttons ── */
    .btn-nutriwise {
      background:var(--primary); color:#fff; border:none;
      padding:10px 26px; border-radius:8px; font-weight:600;
      font-size:.9rem; transition:background .2s;
    }
    .btn-nutriwise:hover { background:var(--primary-d); color:#fff; }

    /* ── Hero cards on home ── */
    .agent-card {
      border-radius:14px; padding:24px 20px;
      border:1px solid #dde8e2; background:#fff;
      transition:transform .2s, box-shadow .2s;
      height:100%;
    }
    .agent-card:hover { transform:translateY(-4px); box-shadow:0 8px 24px rgba(0,0,0,.1); }
    .agent-card .icon-wrap {
      width:52px; height:52px; border-radius:12px;
      display:flex; align-items:center; justify-content:center;
      font-size:1.5rem; margin-bottom:14px;
    }

    /* ── Spinner ── */
    .spinner-wrap { display:none; align-items:center; gap:10px; color:var(--primary); font-size:.9rem; margin-top:12px; }
    .spinner-wrap.show { display:flex; }

    /* ── Form controls ── */
    .form-control:focus, .form-select:focus { border-color:var(--primary); box-shadow:0 0 0 .2rem rgba(26,127,75,.2); }
    label { font-weight:500; font-size:.875rem; color:#3a4a55; }

    /* ── Responsive ── */
    @media(max-width:768px){
      #sidebar { transform:translateX(-100%); }
      #sidebar.open { transform:translateX(0); }
      #main { margin-left:0; }
    }
  </style>
</head>
<body>

<!-- ═══════════════ SIDEBAR ═══════════════ -->
<nav id="sidebar">
  <div class="brand">
    <h5><i class="bi bi-activity me-2"></i>NutriWise AI</h5>
    <small>Personalized Nutrition Coach</small>
  </div>

  <div class="nav-section">Navigation</div>
  <a href="/" class="nav-link {% if active=='home' %}active{% endif %}">
    <i class="bi bi-house-door"></i> Home
  </a>
  <a href="/about" class="nav-link {% if active=='about' %}active{% endif %}">
    <i class="bi bi-info-circle"></i> About
  </a>

  <div class="nav-section">AI Agents</div>
  <a href="/nutrition-chat" class="nav-link {% if active=='chat' %}active{% endif %}">
    <i class="bi bi-chat-dots"></i> Nutrition Chat
  </a>
  <a href="/diet-planner" class="nav-link {% if active=='planner' %}active{% endif %}">
    <i class="bi bi-calendar2-heart"></i> Diet Planner
  </a>
  <a href="/health-advisor" class="nav-link {% if active=='advisor' %}active{% endif %}">
    <i class="bi bi-heart-pulse"></i> Health Advisor
  </a>
  <a href="/meal-analyzer" class="nav-link {% if active=='analyzer' %}active{% endif %}">
    <i class="bi bi-clipboard2-data"></i> Meal Analyzer
  </a>

  <div class="sidebar-footer">
    <i class="bi bi-cpu me-1"></i>Powered by IBM watsonx.ai<br>
    <span style="opacity:.7;">Granite Model · Multi-Agent AI</span>
  </div>
</nav>

<!-- ═══════════════ MAIN ═══════════════ -->
<div id="main">
  <!-- Top bar -->
  <div class="topbar">
    <button class="btn btn-sm d-md-none me-1" onclick="document.getElementById('sidebar').classList.toggle('open')">
      <i class="bi bi-list fs-5"></i>
    </button>
    <i class="bi bi-activity text-success fs-5"></i>
    <h6>{{ page_title }}</h6>
    <span class="badge-ibm ms-auto"><i class="bi bi-cpu me-1"></i>IBM watsonx.ai</span>
  </div>

  <!-- Page content -->
  <div class="content-area">
    {% block content %}{% endblock %}
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
{% block scripts %}{% endblock %}
</body>
</html>
"""

# --------------- Home page --------------------------------------------------
HOME_HTML = BASE_LAYOUT.replace("{% block content %}{% endblock %}", """
<!-- Hero -->
<div class="p-4 mb-4 rounded-3" style="background:linear-gradient(135deg,#0f3d24 0%,#1a7f4b 100%);color:#fff;">
  <div class="row align-items-center">
    <div class="col-md-8">
      <span class="badge bg-warning text-dark mb-2" style="font-size:.7rem;font-weight:700;">IBM HACKATHON PROJECT</span>
      <h1 class="fw-bold mb-2" style="font-size:2rem;">NutriWise AI</h1>
      <p class="mb-3" style="opacity:.9;font-size:1.05rem;">
        An Agentic AI Nutrition Coach powered by <strong>IBM watsonx.ai Granite Models</strong>.
        Four specialised agents — one unified nutrition intelligence platform.
      </p>
      <a href="/nutrition-chat" class="btn btn-light fw-600 me-2">
        <i class="bi bi-chat-dots me-1"></i>Ask a Question
      </a>
      <a href="/diet-planner" class="btn btn-outline-light">
        <i class="bi bi-calendar2-heart me-1"></i>Plan My Diet
      </a>
    </div>
    <div class="col-md-4 text-center d-none d-md-block">
      <i class="bi bi-activity" style="font-size:6rem;opacity:.3;"></i>
    </div>
  </div>
</div>

<!-- Stats row -->
<div class="row g-3 mb-4">
  <div class="col-6 col-md-3">
    <div class="card text-center p-3">
      <div class="fs-2 fw-bold text-success">4</div>
      <div class="text-muted" style="font-size:.8rem;">AI Agents</div>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="card text-center p-3">
      <div class="fs-2 fw-bold text-success">∞</div>
      <div class="text-muted" style="font-size:.8rem;">Personalised Plans</div>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="card text-center p-3">
      <div class="fs-2 fw-bold text-success">6</div>
      <div class="text-muted" style="font-size:.8rem;">Health Conditions</div>
    </div>
  </div>
  <div class="col-6 col-md-3">
    <div class="card text-center p-3">
      <div class="fs-2 fw-bold text-success">1</div>
      <div class="text-muted" style="font-size:.8rem;">Granite Model</div>
    </div>
  </div>
</div>

<!-- Agent Cards -->
<h5 class="fw-600 mb-3"><i class="bi bi-grid me-2 text-success"></i>Your AI Nutrition Agents</h5>
<div class="row g-4">
  <div class="col-md-6 col-lg-3">
    <div class="agent-card">
      <div class="icon-wrap" style="background:#e6f4ed;color:#1a7f4b;"><i class="bi bi-chat-dots"></i></div>
      <h6 class="fw-700">Nutrition Knowledge</h6>
      <p class="text-muted" style="font-size:.84rem;">Ask any nutrition question and get evidence-based answers powered by IBM Granite.</p>
      <a href="/nutrition-chat" class="btn btn-nutriwise btn-sm mt-1">Ask Now →</a>
    </div>
  </div>
  <div class="col-md-6 col-lg-3">
    <div class="agent-card">
      <div class="icon-wrap" style="background:#e8f0fe;color:#1a56db;"><i class="bi bi-calendar2-heart"></i></div>
      <h6 class="fw-700">Diet Planner</h6>
      <p class="text-muted" style="font-size:.84rem;">Generate a personalised meal plan tailored to your goals, body, and dietary preferences.</p>
      <a href="/diet-planner" class="btn btn-nutriwise btn-sm mt-1">Plan Diet →</a>
    </div>
  </div>
  <div class="col-md-6 col-lg-3">
    <div class="agent-card">
      <div class="icon-wrap" style="background:#fef3e2;color:#d97706;"><i class="bi bi-heart-pulse"></i></div>
      <h6 class="fw-700">Health Advisor</h6>
      <p class="text-muted" style="font-size:.84rem;">Get dietary advice tailored to health conditions like diabetes, PCOS, or hypertension.</p>
      <a href="/health-advisor" class="btn btn-nutriwise btn-sm mt-1">Get Advice →</a>
    </div>
  </div>
  <div class="col-md-6 col-lg-3">
    <div class="agent-card">
      <div class="icon-wrap" style="background:#fce7f3;color:#9d174d;"><i class="bi bi-clipboard2-data"></i></div>
      <h6 class="fw-700">Meal Analyzer</h6>
      <p class="text-muted" style="font-size:.84rem;">Describe what you ate and receive a detailed nutritional analysis with smart suggestions.</p>
      <a href="/meal-analyzer" class="btn btn-nutriwise btn-sm mt-1">Analyse →</a>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", "")

# --------------- Nutrition Chat page ----------------------------------------
CHAT_HTML = BASE_LAYOUT.replace("{% block content %}{% endblock %}", """
<div class="row justify-content-center">
  <div class="col-lg-8">
    <div class="d-flex align-items-center gap-2 mb-3">
      <span class="agent-badge"><i class="bi bi-chat-dots"></i> Agent 1</span>
      <h5 class="mb-0 ms-1 fw-700">Nutrition Knowledge Agent</h5>
    </div>
    <p class="text-muted mb-4" style="font-size:.88rem;">
      Ask any nutrition-related question. Powered by <strong>IBM watsonx.ai Granite</strong>.
    </p>

    <!-- Example questions -->
    <div class="mb-3">
      <small class="text-muted fw-600">TRY ASKING:</small>
      <div class="d-flex flex-wrap gap-2 mt-1">
        <button class="btn btn-outline-success btn-sm example-q">What are the benefits of oats?</button>
        <button class="btn btn-outline-success btn-sm example-q">Which foods are rich in protein?</button>
        <button class="btn btn-outline-success btn-sm example-q">Is paneer healthy for muscle gain?</button>
        <button class="btn btn-outline-success btn-sm example-q">What foods contain Vitamin B12?</button>
      </div>
    </div>

    <div class="card">
      <div class="card-header d-flex align-items-center gap-2">
        <i class="bi bi-chat-left-text text-success"></i> Ask Your Question
      </div>
      <div class="card-body p-4">
        <div class="mb-3">
          <label for="question" class="form-label">Your nutrition question</label>
          <textarea class="form-control" id="question" rows="3" placeholder="e.g. What are the health benefits of eating almonds daily?"></textarea>
        </div>
        <button class="btn btn-nutriwise" id="askBtn" onclick="askQuestion()">
          <i class="bi bi-send me-1"></i>Ask NutriWise AI
        </button>
        <div class="spinner-wrap" id="spinner1">
          <div class="spinner-border spinner-border-sm text-success"></div>
          <span>IBM Granite is thinking…</span>
        </div>
      </div>
    </div>

    <!-- Response -->
    <div class="card mt-4" id="responseCard" style="display:none;">
      <div class="card-header d-flex align-items-center gap-2">
        <i class="bi bi-stars text-warning"></i> AI Response
        <span class="badge bg-success ms-auto" style="font-size:.7rem;">IBM Granite</span>
      </div>
      <div class="card-body p-4">
        <div class="response-box" id="responseText"></div>
      </div>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
  // Pre-fill example questions on click
  document.querySelectorAll('.example-q').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('question').value = btn.textContent;
    });
  });

  async function askQuestion() {
    const question = document.getElementById('question').value.trim();
    if (!question) { alert('Please enter a question.'); return; }
    document.getElementById('askBtn').disabled = true;
    document.getElementById('spinner1').classList.add('show');
    document.getElementById('responseCard').style.display = 'none';

    const res = await fetch('/api/nutrition-chat', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question})
    });
    const data = await res.json();

    document.getElementById('responseText').textContent = data.response;
    document.getElementById('responseCard').style.display = 'block';
    document.getElementById('askBtn').disabled = false;
    document.getElementById('spinner1').classList.remove('show');
  }
</script>
""")

# --------------- Diet Planner page ------------------------------------------
PLANNER_HTML = BASE_LAYOUT.replace("{% block content %}{% endblock %}", """
<div class="row">
  <div class="col-lg-5">
    <div class="d-flex align-items-center gap-2 mb-3">
      <span class="agent-badge"><i class="bi bi-calendar2-heart"></i> Agent 2</span>
      <h5 class="mb-0 ms-1 fw-700">Diet Planner Agent</h5>
    </div>
    <p class="text-muted mb-3" style="font-size:.88rem;">Enter your details to get an AI-generated personalised meal plan.</p>

    <div class="card">
      <div class="card-header"><i class="bi bi-person-fill me-2 text-primary"></i>Your Profile</div>
      <div class="card-body p-4">
        <div class="row g-3">
          <div class="col-6">
            <label>Age (years)</label>
            <input type="number" class="form-control" id="age" value="28" min="10" max="90"/>
          </div>
          <div class="col-6">
            <label>Gender</label>
            <select class="form-select" id="gender">
              <option>Male</option><option>Female</option><option>Other</option>
            </select>
          </div>
          <div class="col-6">
            <label>Height (cm)</label>
            <input type="number" class="form-control" id="height" value="170" min="100" max="250"/>
          </div>
          <div class="col-6">
            <label>Weight (kg)</label>
            <input type="number" class="form-control" id="weight" value="70" min="30" max="300"/>
          </div>
          <div class="col-12">
            <label>Dietary Preference</label>
            <select class="form-select" id="diet_pref">
              <option>Vegetarian</option>
              <option>Vegan</option>
              <option>Non-Vegetarian</option>
              <option>Eggetarian</option>
              <option>Pescatarian</option>
            </select>
          </div>
          <div class="col-12">
            <label>Activity Level</label>
            <select class="form-select" id="activity">
              <option>Sedentary (little/no exercise)</option>
              <option>Light (1–3 days/week)</option>
              <option>Moderate (3–5 days/week)</option>
              <option>Active (6–7 days/week)</option>
              <option>Very Active (athlete/hard labour)</option>
            </select>
          </div>
          <div class="col-12">
            <label>Fitness Goal</label>
            <select class="form-select" id="goal">
              <option>Weight Loss</option>
              <option>Weight Gain</option>
              <option>Muscle Gain</option>
              <option>General Wellness</option>
              <option>Maintain Weight</option>
            </select>
          </div>
        </div>
        <button class="btn btn-nutriwise mt-4 w-100" id="planBtn" onclick="generatePlan()">
          <i class="bi bi-magic me-1"></i>Generate My Meal Plan
        </button>
        <div class="spinner-wrap justify-content-center mt-3" id="spinner2">
          <div class="spinner-border spinner-border-sm text-success"></div>
          <span>IBM Granite is crafting your plan…</span>
        </div>
      </div>
    </div>
  </div>

  <div class="col-lg-7 mt-4 mt-lg-0" id="planResult" style="display:none;">
    <div class="d-flex align-items-center gap-2 mb-3">
      <i class="bi bi-stars text-warning fs-5"></i>
      <h5 class="mb-0 fw-700">Your Personalised Meal Plan</h5>
      <span class="badge bg-success ms-auto" style="font-size:.7rem;">IBM Granite</span>
    </div>
    <div class="card">
      <div class="card-body p-4">
        <div class="response-box" id="planText"></div>
      </div>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
  async function generatePlan() {
    document.getElementById('planBtn').disabled = true;
    document.getElementById('spinner2').classList.add('show');
    document.getElementById('planResult').style.display = 'none';

    const payload = {
      age:      document.getElementById('age').value,
      gender:   document.getElementById('gender').value,
      height:   document.getElementById('height').value,
      weight:   document.getElementById('weight').value,
      diet_pref:document.getElementById('diet_pref').value,
      activity: document.getElementById('activity').value,
      goal:     document.getElementById('goal').value,
    };

    const res = await fetch('/api/diet-planner', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    document.getElementById('planText').textContent = data.response;
    document.getElementById('planResult').style.display = 'block';
    document.getElementById('planBtn').disabled = false;
    document.getElementById('spinner2').classList.remove('show');
  }
</script>
""")

# --------------- Health Advisor page ----------------------------------------
ADVISOR_HTML = BASE_LAYOUT.replace("{% block content %}{% endblock %}", """
<div class="row justify-content-center">
  <div class="col-lg-9">
    <div class="d-flex align-items-center gap-2 mb-3">
      <span class="agent-badge"><i class="bi bi-heart-pulse"></i> Agent 3</span>
      <h5 class="mb-0 ms-1 fw-700">Health Advisory Agent</h5>
    </div>
    <p class="text-muted mb-4" style="font-size:.88rem;">
      Select your health condition(s) to receive personalised dietary and lifestyle recommendations.
    </p>

    <div class="card mb-4">
      <div class="card-header"><i class="bi bi-activity me-2 text-danger"></i>Select Health Conditions</div>
      <div class="card-body p-4">
        <div class="row g-3 mb-4">
          <div class="col-6 col-md-4">
            <div class="form-check p-3 border rounded-3 h-100" style="background:#fff9f9;">
              <input class="form-check-input" type="checkbox" value="Diabetes" id="c1"/>
              <label class="form-check-label fw-500" for="c1">
                <i class="bi bi-droplet me-1 text-danger"></i>Diabetes
              </label>
            </div>
          </div>
          <div class="col-6 col-md-4">
            <div class="form-check p-3 border rounded-3 h-100" style="background:#fff9f9;">
              <input class="form-check-input" type="checkbox" value="Hypertension" id="c2"/>
              <label class="form-check-label fw-500" for="c2">
                <i class="bi bi-heart me-1 text-danger"></i>Hypertension
              </label>
            </div>
          </div>
          <div class="col-6 col-md-4">
            <div class="form-check p-3 border rounded-3 h-100" style="background:#fff9f9;">
              <input class="form-check-input" type="checkbox" value="Obesity" id="c3"/>
              <label class="form-check-label fw-500" for="c3">
                <i class="bi bi-person-fill me-1 text-warning"></i>Obesity
              </label>
            </div>
          </div>
          <div class="col-6 col-md-4">
            <div class="form-check p-3 border rounded-3 h-100" style="background:#fff9f9;">
              <input class="form-check-input" type="checkbox" value="Heart Disease" id="c4"/>
              <label class="form-check-label fw-500" for="c4">
                <i class="bi bi-heart-pulse me-1 text-danger"></i>Heart Disease
              </label>
            </div>
          </div>
          <div class="col-6 col-md-4">
            <div class="form-check p-3 border rounded-3 h-100" style="background:#fff9f9;">
              <input class="form-check-input" type="checkbox" value="PCOS" id="c5"/>
              <label class="form-check-label fw-500" for="c5">
                <i class="bi bi-gender-female me-1 text-pink"></i>PCOS
              </label>
            </div>
          </div>
          <div class="col-6 col-md-4">
            <div class="form-check p-3 border rounded-3 h-100" style="background:#fff9f9;">
              <input class="form-check-input" type="checkbox" value="High Cholesterol" id="c6"/>
              <label class="form-check-label fw-500" for="c6">
                <i class="bi bi-droplet-half me-1 text-warning"></i>High Cholesterol
              </label>
            </div>
          </div>
        </div>
        <button class="btn btn-nutriwise" id="advBtn" onclick="getAdvice()">
          <i class="bi bi-shield-heart me-1"></i>Get Health Recommendations
        </button>
        <div class="spinner-wrap mt-3" id="spinner3">
          <div class="spinner-border spinner-border-sm text-success"></div>
          <span>IBM Granite is analysing…</span>
        </div>
      </div>
    </div>

    <div class="card" id="advResult" style="display:none;">
      <div class="card-header d-flex align-items-center gap-2">
        <i class="bi bi-stars text-warning"></i> Health Recommendations
        <span class="badge bg-success ms-auto" style="font-size:.7rem;">IBM Granite</span>
      </div>
      <div class="card-body p-4">
        <div class="alert alert-warning py-2 mb-3" style="font-size:.82rem;">
          <i class="bi bi-exclamation-triangle me-1"></i>
          <strong>Disclaimer:</strong> Educational information only. Consult a healthcare professional for medical advice.
        </div>
        <div class="response-box" id="advText"></div>
      </div>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
  async function getAdvice() {
    const checkboxes = document.querySelectorAll('input[type=checkbox]:checked');
    const conditions = Array.from(checkboxes).map(c => c.value);
    if (conditions.length === 0) { alert('Please select at least one health condition.'); return; }

    document.getElementById('advBtn').disabled = true;
    document.getElementById('spinner3').classList.add('show');
    document.getElementById('advResult').style.display = 'none';

    const res = await fetch('/api/health-advisor', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({conditions})
    });
    const data = await res.json();

    document.getElementById('advText').textContent = data.response;
    document.getElementById('advResult').style.display = 'block';
    document.getElementById('advBtn').disabled = false;
    document.getElementById('spinner3').classList.remove('show');
  }
</script>
""")

# --------------- Meal Analyzer page -----------------------------------------
ANALYZER_HTML = BASE_LAYOUT.replace("{% block content %}{% endblock %}", """
<div class="row justify-content-center">
  <div class="col-lg-8">
    <div class="d-flex align-items-center gap-2 mb-3">
      <span class="agent-badge"><i class="bi bi-clipboard2-data"></i> Agent 4</span>
      <h5 class="mb-0 ms-1 fw-700">Meal Analysis Agent</h5>
    </div>
    <p class="text-muted mb-4" style="font-size:.88rem;">
      Describe what you ate today and get an AI-powered nutritional breakdown and smart suggestions.
    </p>

    <div class="card mb-4">
      <div class="card-header"><i class="bi bi-pencil-square me-2 text-success"></i>Enter Your Meal Log</div>
      <div class="card-body p-4">
        <div class="mb-3">
          <label class="form-label">Describe your meals (include portions where possible)</label>
          <textarea class="form-control" id="mealLog" rows="8"
            placeholder="Example:
Breakfast:
2 Rotis with ghee
1 bowl dal tadka
1 glass milk

Lunch:
1 cup rice
Paneer butter masala
Cucumber salad

Evening Snack:
1 banana
Chai with sugar

Dinner:
2 Rotis
Mixed vegetable curry
1 glass buttermilk"></textarea>
        </div>
        <button class="btn btn-nutriwise" id="analyzeBtn" onclick="analyzeMeal()">
          <i class="bi bi-search me-1"></i>Analyse My Meals
        </button>
        <div class="spinner-wrap mt-3" id="spinner4">
          <div class="spinner-border spinner-border-sm text-success"></div>
          <span>IBM Granite is analysing your meals…</span>
        </div>
      </div>
    </div>

    <div class="card" id="analyzeResult" style="display:none;">
      <div class="card-header d-flex align-items-center gap-2">
        <i class="bi bi-bar-chart-line text-success"></i> Nutritional Analysis
        <span class="badge bg-success ms-auto" style="font-size:.7rem;">IBM Granite</span>
      </div>
      <div class="card-body p-4">
        <div class="response-box" id="analyzeText"></div>
      </div>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", """
<script>
  async function analyzeMeal() {
    const mealLog = document.getElementById('mealLog').value.trim();
    if (!mealLog) { alert('Please enter your meal description.'); return; }

    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('spinner4').classList.add('show');
    document.getElementById('analyzeResult').style.display = 'none';

    const res = await fetch('/api/meal-analyzer', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({meal_description: mealLog})
    });
    const data = await res.json();

    document.getElementById('analyzeText').textContent = data.response;
    document.getElementById('analyzeResult').style.display = 'block';
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('spinner4').classList.remove('show');
  }
</script>
""")

# --------------- About page -------------------------------------------------
ABOUT_HTML = BASE_LAYOUT.replace("{% block content %}{% endblock %}", """
<div class="row justify-content-center">
  <div class="col-lg-9">
    <h4 class="fw-700 mb-1">About NutriWise AI</h4>
    <p class="text-muted mb-4">Architecture, technology, and how the multi-agent system works.</p>

    <!-- IBM watsonx.ai integration card -->
    <div class="card mb-4" style="border-left:4px solid #0f62fe;">
      <div class="card-body p-4">
        <div class="d-flex align-items-center gap-3 mb-3">
          <div style="background:#e8f0fe;color:#0f62fe;width:48px;height:48px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;">
            <i class="bi bi-cpu"></i>
          </div>
          <div>
            <h5 class="mb-0 fw-700">IBM watsonx.ai Integration</h5>
            <small class="text-muted">Granite Model · Foundation AI Engine</small>
          </div>
        </div>
        <p style="font-size:.9rem;line-height:1.75;">
          All four agents share a single <code>generate_response(prompt)</code> function that calls the
          <strong>IBM watsonx.ai Granite Model</strong> (<code>ibm/granite-3-8b-instruct</code>).
          Credentials are loaded from environment variables (<code>WATSONX_API_KEY</code>,
          <code>WATSONX_PROJECT_ID</code>, <code>WATSONX_URL</code>) — no hardcoded secrets.
        </p>
        <div class="row g-3 mt-1">
          <div class="col-md-4">
            <div class="p-3 rounded-3" style="background:#f0f4ff;font-size:.83rem;">
              <strong>Model</strong><br>ibm/granite-3-8b-instruct
            </div>
          </div>
          <div class="col-md-4">
            <div class="p-3 rounded-3" style="background:#f0f4ff;font-size:.83rem;">
              <strong>Max Tokens</strong><br>800 per response
            </div>
          </div>
          <div class="col-md-4">
            <div class="p-3 rounded-3" style="background:#f0f4ff;font-size:.83rem;">
              <strong>Temperature</strong><br>0.7 (balanced creativity)
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Four agents -->
    <h5 class="fw-700 mb-3"><i class="bi bi-diagram-3 me-2 text-success"></i>Multi-Agent Architecture</h5>
    <div class="row g-3 mb-4">
      <div class="col-md-6">
        <div class="card h-100">
          <div class="card-body p-4">
            <span class="agent-badge mb-2 d-inline-block"><i class="bi bi-chat-dots"></i> Agent 1</span>
            <h6 class="fw-700">Nutrition Knowledge Agent</h6>
            <p style="font-size:.84rem;color:#555;">
              Handles open-ended nutrition Q&amp;A. Constructs an expert-nutritionist prompt and sends it
              to IBM Granite, returning educational, evidence-based answers.
            </p>
            <code style="font-size:.78rem;">nutrition_knowledge_agent(question)</code>
          </div>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card h-100">
          <div class="card-body p-4">
            <span class="agent-badge mb-2 d-inline-block"><i class="bi bi-calendar2-heart"></i> Agent 2</span>
            <h6 class="fw-700">Diet Planner Agent</h6>
            <p style="font-size:.84rem;color:#555;">
              Accepts age, gender, height, weight, diet preference, activity level, and fitness goal.
              Builds a detailed dietitian-style prompt and returns a full-day structured meal plan.
            </p>
            <code style="font-size:.78rem;">diet_planner_agent(age, gender, …, goal)</code>
          </div>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card h-100">
          <div class="card-body p-4">
            <span class="agent-badge mb-2 d-inline-block"><i class="bi bi-heart-pulse"></i> Agent 3</span>
            <h6 class="fw-700">Health Advisory Agent</h6>
            <p style="font-size:.84rem;color:#555;">
              Takes a list of health conditions (diabetes, PCOS, etc.) and generates tailored foods-to-include,
              foods-to-avoid, habits, and lifestyle tips. Always appends a medical disclaimer.
            </p>
            <code style="font-size:.78rem;">health_advisory_agent(conditions)</code>
          </div>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card h-100">
          <div class="card-body p-4">
            <span class="agent-badge mb-2 d-inline-block"><i class="bi bi-clipboard2-data"></i> Agent 4</span>
            <h6 class="fw-700">Meal Analysis Agent</h6>
            <p style="font-size:.84rem;color:#555;">
              Accepts a free-text meal log and returns quality score, nutritional strengths, deficiencies,
              healthier swaps, and macro balance estimates — all AI-generated by IBM Granite.
            </p>
            <code style="font-size:.78rem;">meal_analysis_agent(meal_description)</code>
          </div>
        </div>
      </div>
    </div>

    <!-- Orchestrator -->
    <div class="card mb-4" style="border-left:4px solid #1a7f4b;">
      <div class="card-body p-4">
        <h6 class="fw-700"><i class="bi bi-diagram-2 me-2 text-success"></i>Agent Orchestrator</h6>
        <p style="font-size:.88rem;line-height:1.75;">
          The <code>orchestrate(agent_name, payload)</code> function acts as the central router.
          Each Flask API endpoint calls the orchestrator with the target agent name and the
          request payload — keeping routing logic cleanly separated from agent logic.
        </p>
      </div>
    </div>

    <!-- Tech stack -->
    <h5 class="fw-700 mb-3"><i class="bi bi-stack me-2 text-success"></i>Technology Stack</h5>
    <div class="row g-3">
      <div class="col-6 col-md-3">
        <div class="card text-center p-3"><i class="bi bi-filetype-py fs-3 text-primary mb-1"></i><br><small class="fw-600">Python 3</small></div>
      </div>
      <div class="col-6 col-md-3">
        <div class="card text-center p-3"><i class="bi bi-server fs-3 text-success mb-1"></i><br><small class="fw-600">Flask</small></div>
      </div>
      <div class="col-6 col-md-3">
        <div class="card text-center p-3"><i class="bi bi-bootstrap fs-3 text-purple mb-1" style="color:#7952b3;"></i><br><small class="fw-600">Bootstrap 5</small></div>
      </div>
      <div class="col-6 col-md-3">
        <div class="card text-center p-3"><i class="bi bi-cpu fs-3 text-info mb-1" style="color:#0f62fe;"></i><br><small class="fw-600">IBM Granite</small></div>
      </div>
    </div>
  </div>
</div>
""").replace("{% block scripts %}{% endblock %}", "")


# =============================================================================
# FLASK ROUTES – Page Views
# =============================================================================

@app.route("/")
def home():
    return render_template_string(HOME_HTML, page_title="Home", active="home")

@app.route("/nutrition-chat")
def nutrition_chat():
    return render_template_string(CHAT_HTML, page_title="Nutrition Chat", active="chat")

@app.route("/diet-planner")
def diet_planner():
    return render_template_string(PLANNER_HTML, page_title="Diet Planner", active="planner")

@app.route("/health-advisor")
def health_advisor():
    return render_template_string(ADVISOR_HTML, page_title="Health Advisor", active="advisor")

@app.route("/meal-analyzer")
def meal_analyzer():
    return render_template_string(ANALYZER_HTML, page_title="Meal Analyzer", active="analyzer")

@app.route("/about")
def about():
    return render_template_string(ABOUT_HTML, page_title="About", active="about")


# =============================================================================
# FLASK ROUTES – JSON API Endpoints (called by JavaScript fetch)
# =============================================================================

@app.route("/api/nutrition-chat", methods=["POST"])
def api_nutrition_chat():
    """Agent 1 endpoint — routes to Nutrition Knowledge Agent via orchestrator."""
    data = request.get_json()
    response = orchestrate("nutrition_knowledge", {"question": data.get("question", "")})
    return jsonify({"response": response})


@app.route("/api/diet-planner", methods=["POST"])
def api_diet_planner():
    """Agent 2 endpoint — routes to Diet Planner Agent via orchestrator."""
    data = request.get_json()
    response = orchestrate("diet_planner", data)
    return jsonify({"response": response})


@app.route("/api/health-advisor", methods=["POST"])
def api_health_advisor():
    """Agent 3 endpoint — routes to Health Advisory Agent via orchestrator."""
    data = request.get_json()
    response = orchestrate("health_advisory", {"conditions": data.get("conditions", [])})
    return jsonify({"response": response})


@app.route("/api/meal-analyzer", methods=["POST"])
def api_meal_analyzer():
    """Agent 4 endpoint — routes to Meal Analysis Agent via orchestrator."""
    data = request.get_json()
    response = orchestrate("meal_analysis", {"meal_description": data.get("meal_description", "")})
    return jsonify({"response": response})


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  NutriWise AI – Personalized Nutrition Coach")
    print("  Powered by IBM watsonx.ai Granite Models")
    print("=" * 60)
    print()
    print("  Required environment variables:")
    print("    WATSONX_API_KEY      :", "✓ set" if WATSONX_API_KEY else "✗ NOT SET")
    print("    WATSONX_PROJECT_ID   :", "✓ set" if WATSONX_PROJECT_ID else "✗ NOT SET")
    print("    WATSONX_URL          :", WATSONX_URL)
    print()
    print("  Starting Flask on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
