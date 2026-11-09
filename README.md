# Promo Finder

An LLM-powered pipeline for discovering, generating, ranking, and auditing promotional clips from timestamped micro-drama screenplays.

The project consists of three tasks:

- **Task 1 – Promo Generation:** Identifies potential promotional moments from screenplays, generates 20–60 second promo candidates, validates timestamps and segment continuity, optionally explores cross-episode stitching, and evaluates candidates using an LLM judge.
- **Task 2 – Promo Ranking:** Ranks generated promos based on acquisition potential, premise clarity, spoiler risk, and diversity.
- **Task 3 – Quality Audit:** Analyzes candidate quality, identifies potential weaknesses, evaluates judge consistency, and provides a human-review mechanism.

## 1. Project Architecture

The overall pipeline follows this workflow:

**Task 1: Promo Generation**

1. Load and parse timestamped screenplays.
2. Analyze individual episodes using LLM-based candidate generation.
3. Validate timestamps, duration, chronology, and segment gaps.
4. Optionally generate cross-episode promo candidates.
5. Evaluate candidates on hook strength, cliffhanger strength, and standalone clarity.
6. Filter and deduplicate candidates.

**Task 2: Promo Ranking**

1. Evaluate promos from a growth and acquisition perspective.
2. Score lead potential and premise clarity.
3. Account for spoiler risk and promo duration.
4. Apply diversity-aware ranking to avoid selecting repetitive promos.
5. Generate explanations for the top-ranked candidates.

**Task 3: Quality Audit**

1. Analyze accepted and rejected candidates.
2. Identify structural issues and potential quality problems.
3. Evaluate judge-score consistency.
4. Run an adversarial LLM critic on a sample of promos.
5. Identify potentially missed promotional moments using a heuristic.
6. Generate a human-evaluation template.

## 2. Repository Structure

The repository should contain:

- `promo_finder.py` – Task 1: Promo generation and validation.
- `rank_and_audit.py` – Task 2: Ranking and Task 3: Quality auditing.
- `Input/` – Directory containing screenplay files.
- `out/` – Directory containing generated results.

**Important:** Both Python files must remain in the same directory because `rank_and_audit.py` imports `promo_finder.py`.

## 3. Requirements

The project was developed and run using:

- Python 3.14
- LangChain 1.4.9
- LangGraph 1.2.20
- LangChain Google GenAI 4.4.3
- Google GenAI 3.0.0
- python-dotenv 1.2.4

An LLM API key is required to execute the generation, judging, ranking, and LLM-based auditing stages.

The current configuration uses Google Gemini through LangChain.

## 4. Installation

Clone the repository:

```bash
git clone https://github.com/Digiiiii/KUKU_Promo_Finder.git
cd KUKU_Promo_Finder
```

Create and activate a virtual environment (recommended):

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the required dependencies:

```bash
python -m pip install langchain==1.4.9 langgraph==1.2.20 python-dotenv==1.2.4 langchain-google-genai==4.4.3
```

These are the direct dependencies used by the current Google Gemini configuration. Their transitive dependencies will be installed automatically. Install the following if using different api: 
```bash
pip install langchain-openai
pip install langchain-mistralai
pip install langchain-openrouter
pip install langchain-anthropic
pip install langchain-groq
pip install langchain-ollama
```

## 5. Configuration

The project uses environment variables to configure the LLM.

Edit the provided demo `.env` file in the project root and insert api keys:

```env
LLM_PROVIDER=google_genai
LLM_MODEL=your-gemini-model-id
GOOGLE_API_KEY=your-google-api-key
```

Replace the model identifier with a Gemini model currently available to your API account.

The implementation uses LangChain's `init_chat_model()` interface, allowing the LLM provider and model to be configured without changing the pipeline code.

**Security:** Never commit your `.env` file or API keys to GitHub.

## 6. Input Format

The application expects timestamped screenplay files in the `Input/` directory.

The parser recognizes:

- Episode number.
- Episode logline.
- Continuity notes.
- Scene headings and timestamps.
- Dialogue blocks.
- Action blocks.
- Sound effects and music blocks.
- Optional camera-direction blocks.

The expected screenplay structure is:

- `EPISODE NUMBER:`
- `EPISODE LOGLINE:`
- `CONTINUITY NOTES:`
- `--- SCRIPT START ---`
- Scene information.
- Labelled screenplay blocks.
- `Start:` and `End:` timestamps for each timed block.

Timestamps use the `MM:SS.ss` format and are interpreted relative to the beginning of the episode.

A single screenplay file can contain one or multiple episodes.

Episode numbering is extracted from the episode header when available. Otherwise, the application attempts to infer it from the filename.

Place the screenplay files inside `Input/`, or provide another directory through the `--input` argument.

## 7. Execution

All commands should be executed from the project root directory.

### Task 1 – Promo Generation

First, verify that the screenplay parser recognizes the input correctly:

```bash
python promo_finder.py --parse-only
```

This checks screenplay parsing without making any LLM API calls.

Once parsing is verified, execute promo generation:

```bash
python promo_finder.py
```

The pipeline performs candidate generation, validation, optional cross-episode stitching, judging, and final filtering.

**Generated files:**

| File | Description |
|---|---|
| `out/promos.json` | Accepted promo candidates and their metadata |
| `out/promos.md` | Human-readable list of generated promos |
| `out/rejected.json` | Candidates rejected during validation or judging |

### Task 2 – Promo Ranking and Quality Audit

Execute only ranking after Task 1:

```bash
python rank_and_audit.py --rank-only
```

The ranking process evaluates promotional potential, combines relevant scores, applies adjustments, and produces a diversity-aware ranking.

**Generated files:**

| File | Description |
|---|---|
| `out/ranking.json` | Ranked promo candidates with scoring information |
| `out/ranking.md` | Human-readable ranking and explanations |


Execute ranking and audit both :

```bash
python rank_and_audit.py
```

To execute the audit without the additional LLM critic and judge-stability calls:

```bash
python rank_and_audit.py --audit-only --no-llm-audit
```

**Generated files:**

| File | Description |
|---|---|
| `out/audit.json` | Structured quality-audit results |
| `out/audit.md` | Human-readable audit report |
| `out/audit_labels.csv` | Template for manual human evaluation |

### Human Evaluation

The audit generates a CSV template containing selected promo candidates.

Fill the `human_verdict` column with one of:

- `good`
- `ok`
- `bad`

Optionally provide a note explaining the evaluation.

Run the following command to analyze the completed labels:

```bash
python rank_and_audit.py --input Input --out out --audit-only --labels out/audit_labels.csv --no-llm-audit
```

The current human-label analysis compares human verdicts against the original Task 1 judge scores.

## 8. Configuration and Customization

The pipeline supports configurable parameters through environment variables.

### Task 1 – Promo Generation

| Variable | Default | Description |
|---|---:|---|
| `MIN_PROMO_DURATION` | 20 | Minimum promo duration in seconds |
| `MAX_PROMO_DURATION` | 60 | Maximum promo duration in seconds |
| `WINDOW_CHARS` | 20000 | Maximum screenplay characters per analysis window |
| `MIN_SECONDS_BETWEEN_CALLS` | 2.5 | Minimum delay between LLM calls |
| `MAX_CONCURRENCY` | 2 | Maximum concurrent LangGraph operations |
| `ENABLE_CROSS_EPISODE` | true | Enable or disable cross-episode stitching |
| `INCLUDE_CAMERA` | false | Include camera-direction blocks |
| `MAX_SEGMENT_GAP` | 15 | Maximum permitted gap between segments |
| `EPISODE_EDGE_SECONDS` | 15 | Episode-boundary proximity constraint |
| `STITCH_TAIL_SECONDS` | 90 | Amount of ending screenplay text provided to the stitcher |
| `STITCH_HEAD_SECONDS` | 60 | Amount of beginning screenplay text provided to the stitcher |
| `JUDGE_BATCH` | 4 | Maximum promos per judge request |
| `JUDGE_MAX_CHARS` | 9000 | Maximum transcript characters per judge request |
| `MIN_HOOK_SCORE` | 7 | Minimum hook score |
| `MIN_CLIFF_SCORE` | 7 | Minimum cliffhanger score |
| `MIN_STANDALONE_SCORE` | 6 | Minimum standalone-clarity score |
| `KEEP_UNSCORED` | false | Whether to retain candidates that could not be scored |
| `LLM_RETRIES` | 4 | Number of LLM retries |
| `RETRY_BASE_DELAY` | 3 | Base retry delay in seconds |

### Task 2 and Task 3 – Ranking and Auditing

| Variable | Default | Description |
|---|---:|---|
| `W_HOOK` | 0.4 | Hook score weight |
| `W_CLIFF` | 0.3 | Cliffhanger score weight |
| `W_STANDALONE` | 0.3 | Standalone score weight |
| `EARLY_EPISODE` | 3 | Episode threshold for the early-entry bonus |
| `RANK_TOP_K` | 10 | Number of top promos receiving written explanations |
| `CRITIC_SAMPLE` | 12 | Number of promos sampled by the adversarial critic |
| `CRITIC_BATCH` | 3 | Number of promos per critic request |

## 9. Known Limitations

- The system operates on screenplay text and does not analyze actual video footage.
- Acting quality, visual composition, music, sound design, and real video pacing cannot be evaluated.
- Cross-episode stitching proposes text-based combinations and cannot establish actual visual or narrative continuity.
- LLM-generated scores are subjective and should not be treated as measured audience engagement.
- The possible-misses analysis uses a heuristic based on screenplay characteristics and is not a ground-truth recall measurement.
- Deterministic audit flags are warnings rather than definitive indicators of poor promotional quality.
- The same LLM family may be used for generation, judging, ranking, and criticism, so shared model biases can remain.
- Actual promotional effectiveness requires human evaluation and real audience-performance data.
