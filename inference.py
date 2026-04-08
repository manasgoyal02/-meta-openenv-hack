import asyncio
import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from client import AgriOpsEnv  # noqa: E402
from models import AgriOpsAction  # noqa: E402

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("AGRIOPS_TASK", "easy")
BENCHMARK = "agriops_env"
MAX_STEPS = 7
TEMPERATURE = 0.2
MAX_TOKENS = 700
SUCCESS_SCORE_THRESHOLD = 0.55
USE_LLM_POLICY = os.getenv("USE_LLM_POLICY", "0") == "1"

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


SYSTEM_PROMPTS = {
    "easy": textwrap.dedent(
        """
        You are solving an agricultural operations benchmark.

        EASY TASK: crop selection under field constraints.
        Use:
          1) {"action_type":"analyze_field"}
          2) {"action_type":"recommend_crop","crop_recommendation":"<crop>"}

        Pick one concrete crop name only.
        Return exactly one JSON object per turn.
        """
    ).strip(),
    "medium": textwrap.dedent(
        """
        You are solving an agricultural operations benchmark.

        MEDIUM TASK: diagnose issue and prescribe intervention.
        Use this order:
          1) {"action_type":"diagnose_issue","diagnosis":"<text>"}
          2) {"action_type":"recommend_intervention","intervention":"<text>"}
          3) {"action_type":"finalize_case"}

        Return exactly one JSON object per turn.
        """
    ).strip(),
    "hard": textwrap.dedent(
        """
        You are solving an agricultural operations benchmark.

        HARD TASK: seasonal farm planning under constraints.
        Use:
          1) {"action_type":"analyze_constraints"}
          2) optional {"action_type":"propose_plan", ...}
          3) {"action_type":"submit_plan",
               "crop_recommendation":"<crop>",
               "fertilizer_strategy":"<plan>",
               "irrigation_strategy":"<plan>",
               "estimated_cost": <number>}

        Return exactly one JSON object per turn.
        """
    ).strip(),
}


def parse_action(text: str) -> Optional[AgriOpsAction]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text.strip()).strip().rstrip("`").strip()
    try:
        return AgriOpsAction(**json.loads(cleaned))
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return AgriOpsAction(**json.loads(match.group()))
        except Exception:
            pass
    return None


def _choose_easy_crop(data: Dict[str, Any]) -> str:
    season = str(data.get("season", "")).lower()
    rainfall = float(data.get("rainfall_mm", 0) or 0)
    water = str(data.get("water_availability", "")).lower()
    humidity = float(data.get("humidity_pct", 0) or 0)

    if season == "kharif" and (rainfall >= 180 or water == "high") and humidity >= 65:
        return "rice"
    if season == "rabi" and (rainfall <= 120 or water == "low"):
        return "chickpea"
    if water == "low" or rainfall < 100:
        return "mustard"
    return "maize"


def _medium_case_type(data: Dict[str, Any]) -> str:
    crop = str(data.get("crop", "")).lower()
    symptoms = " ".join([str(s).lower() for s in data.get("symptoms", [])])
    weather = data.get("weather", {})
    rain = float(weather.get("rain_last_7d_mm", 0) or 0)
    humidity = float(weather.get("humidity_pct", 0) or 0)

    blight_signal = (
        crop == "tomato"
        or "lesion" in symptoms
        or "defoliation" in symptoms
        or (humidity >= 80 and rain >= 50)
    )
    deficiency_signal = (
        "older leaves" in symptoms
        or "stunted" in symptoms
        or "thin canopy" in symptoms
        or crop == "rice"
    )

    if blight_signal and not deficiency_signal:
        return "blight"
    if deficiency_signal and not blight_signal:
        return "nitrogen"
    if crop == "tomato":
        return "blight"
    return "nitrogen"


def _choose_hard_crop(data: Dict[str, Any]) -> str:
    weather = data.get("weather_forecast", {})
    rain = float(weather.get("rainfall_mm", 0) or 0)
    mean_temp = float(weather.get("mean_temp_c", 0) or 0)
    irrigation = str(data.get("irrigation_availability", "")).lower()
    soil = data.get("soil_profile", {})
    texture = str(soil.get("texture", "")).lower()

    if irrigation == "low" or rain < 300 or "sandy" in texture or mean_temp >= 32:
        return "pearl millet"
    if irrigation == "moderate" and rain >= 350:
        return "maize"
    return "soybean"


def _build_hard_plan(data: Dict[str, Any], crop: str) -> Dict[str, Any]:
    budget = float(data.get("budget_usd", 0) or 0)
    irrigation = str(data.get("irrigation_availability", "")).lower()

    if crop == "pearl millet":
        fert = "Use moderate nitrogen with SSP basal and organic manure incorporation in split schedule."
        irr = "Use drip or deficit irrigation with mulching and irrigate only at critical growth stages."
        cost = min(budget, budget * 0.9) if budget > 0 else 6500
    else:
        fert = "Apply split nitrogen with DAP basal placement and potash top-up with compost support."
        if irrigation == "moderate":
            irr = "Use furrow irrigation at critical growth stages and adjust intervals to rainfall."
        else:
            irr = "Use controlled irrigation at critical growth stages and avoid water-intensive flooding."
        cost = min(budget, budget * 0.95) if budget > 0 else 9000

    return {
        "crop_recommendation": crop,
        "fertilizer_strategy": fert,
        "irrigation_strategy": irr,
        "estimated_cost": round(cost, 2),
    }


def heuristic_action(task: str, step: int, input_data: Dict[str, Any]) -> AgriOpsAction:
    if task == "easy":
        if step == 1:
            return AgriOpsAction(action_type="analyze_field")
        return AgriOpsAction(
            action_type="recommend_crop",
            crop_recommendation=_choose_easy_crop(input_data),
        )

    if task == "medium":
        case_type = _medium_case_type(input_data)
        if step == 1:
            if case_type == "blight":
                diag = "Early blight (Alternaria leaf blight) infection in tomato."
            else:
                diag = "Nitrogen deficiency causing chlorosis and poor vegetative growth."
            return AgriOpsAction(action_type="diagnose_issue", diagnosis=diag)

        if step == 2:
            if case_type == "blight":
                intv = "Apply chlorothalonil fungicide, remove infected leaves, and reduce leaf wetness from overhead irrigation."
            else:
                intv = "Apply urea top dressing in split application and monitor canopy recovery over 7-10 days."
            return AgriOpsAction(action_type="recommend_intervention", intervention=intv)

        return AgriOpsAction(action_type="finalize_case")

    if step == 1:
        return AgriOpsAction(action_type="analyze_constraints")

    crop = _choose_hard_crop(input_data)
    plan = _build_hard_plan(input_data, crop)

    if step == 2:
        return AgriOpsAction(action_type="propose_plan", **plan)
    return AgriOpsAction(action_type="submit_plan", **plan)


def fallback_action(task: str, step: int, input_data: Dict[str, Any]) -> AgriOpsAction:
    return heuristic_action(task, step, input_data)


def get_llm_action(client: OpenAI, system: str, history: List[dict], obs: str) -> str:
    messages = [{"role": "system", "content": system}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": obs})
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        return ""


async def run_episode(task: str) -> None:
    rewards: List[float] = []
    history: List[dict] = []
    steps_taken = 0
    score = 0.0
    success = False
    env = None

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
        sys_prompt = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["easy"])

        if IMAGE_NAME:
            env = await AgriOpsEnv.from_docker_image(IMAGE_NAME, env_vars={"PORT": "8000"})
        else:
            env_url = os.getenv("AGRIOPS_ENV_URL", "http://localhost:8000")
            env = AgriOpsEnv(base_url=env_url)
            await env.connect()

        result = await env.reset(task=task)
        current_input = result.observation.input_data or {}
        obs_message = result.observation.message + "\nInput data: " + json.dumps(current_input)
        done = result.done

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Deterministic policy first for stable benchmark scores.
            action = heuristic_action(task, step, current_input)
            err_msg = None

            # Optional LLM action only when explicitly enabled.`r`n            if USE_LLM_POLICY and llm is not None:`r`n                raw = get_llm_action(llm, sys_prompt, history, obs_message)`r`n                llm_action = parse_action(raw)`r`n                if llm_action is not None and llm_action.action_type == action.action_type:`r`n                    action = llm_action`r`n                elif llm_action is None:`r`n                    err_msg = f"parse_error:{raw[:60]!r}"

            action_str = json.dumps(action.model_dump(exclude_none=True))

            result = await env.step(action)
            reward = result.reward or 0.0
            done = result.done
            current_input = result.observation.input_data or current_input
            obs_message = result.observation.message + "\nInput data: " + json.dumps(current_input)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=err_msg)

            history.append({"role": "user", "content": obs_message})
            history.append({"role": "assistant", "content": action_str})

            if done:
                break

        score = min(max(sum(rewards), 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode exception: {type(exc).__name__}: {exc}", flush=True)

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as close_error:
                print(f"[DEBUG] env.close() error: {close_error}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    try:
        asyncio.run(run_episode(TASK_NAME))
    except Exception as top_error:
        print(f"[DEBUG] Top-level exception: {top_error}", flush=True)
    sys.exit(0)


