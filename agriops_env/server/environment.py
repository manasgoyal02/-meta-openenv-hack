import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server import Environment

from ..models import AgriOpsAction, AgriOpsObservation, AgriOpsState


TASK_BANK: Dict[str, List[Dict[str, Any]]] = {
    "easy": [
        {
            "task_id": "easy-01",
            "difficulty": "easy",
            "instruction": "Recommend the most suitable crop for this field.",
            "input_data": {
                "N": 96,
                "P": 42,
                "K": 44,
                "pH": 6.3,
                "temperature_c": 27,
                "humidity_pct": 78,
                "rainfall_mm": 230,
                "season": "kharif",
                "water_availability": "high",
            },
            "expected_output": {
                "primary_crop": "rice",
                "acceptable_alternatives": ["maize"],
                "unsafe_crops": ["chickpea", "lentil"],
            },
        },
        {
            "task_id": "easy-02",
            "difficulty": "easy",
            "instruction": "Recommend the most suitable crop for this field.",
            "input_data": {
                "N": 34,
                "P": 22,
                "K": 28,
                "pH": 7.4,
                "temperature_c": 23,
                "humidity_pct": 46,
                "rainfall_mm": 62,
                "season": "rabi",
                "water_availability": "low",
            },
            "expected_output": {
                "primary_crop": "chickpea",
                "acceptable_alternatives": ["mustard", "barley"],
                "unsafe_crops": ["rice", "sugarcane"],
            },
        },
    ],
    "medium": [
        {
            "task_id": "medium-01",
            "difficulty": "medium",
            "instruction": "Diagnose the main field issue and recommend intervention.",
            "input_data": {
                "crop": "tomato",
                "soil": {"pH": 6.1, "drainage": "poor", "organic_matter_pct": 1.2},
                "symptoms": [
                    "concentric brown lesions on lower leaves",
                    "yellowing around lesion margins",
                    "defoliation progressing upward",
                ],
                "weather": {
                    "temperature_c": 25,
                    "humidity_pct": 89,
                    "rain_last_7d_mm": 74,
                },
            },
            "expected_output": {
                "diagnosis": ["early blight", "alternaria leaf blight"],
                "diagnosis_keywords": ["blight", "alternaria"],
                "intervention_keywords": ["fungicide", "chlorothalonil", "remove infected leaves"],
                "unsafe_interventions": ["increase overhead irrigation", "no action"],
            },
        },
        {
            "task_id": "medium-02",
            "difficulty": "medium",
            "instruction": "Diagnose the main field issue and recommend intervention.",
            "input_data": {
                "crop": "rice",
                "soil": {"pH": 5.4, "drainage": "moderate", "organic_matter_pct": 0.9},
                "symptoms": [
                    "uniform yellowing on older leaves",
                    "stunted tillering",
                    "thin canopy",
                ],
                "weather": {
                    "temperature_c": 31,
                    "humidity_pct": 67,
                    "rain_last_7d_mm": 12,
                },
            },
            "expected_output": {
                "diagnosis": ["nitrogen deficiency", "n deficiency"],
                "diagnosis_keywords": ["nitrogen", "deficiency"],
                "intervention_keywords": ["urea", "split application", "top dressing"],
                "unsafe_interventions": ["apply fungicide only", "stop fertilization"],
            },
        },
    ],
    "hard": [
        {
            "task_id": "hard-01",
            "difficulty": "hard",
            "instruction": "Build a seasonal farm plan under resource and budget constraints.",
            "input_data": {
                "soil_profile": {"texture": "clay loam", "pH": 6.5, "organic_matter_pct": 1.5},
                "weather_forecast": {"season": "kharif", "rainfall_mm": 540, "mean_temp_c": 28},
                "irrigation_availability": "moderate",
                "fertilizer_availability": ["urea", "DAP", "MOP", "compost"],
                "farm_size_ha": 12,
                "budget_usd": 9800,
            },
            "expected_output": {
                "primary_crop": "maize",
                "acceptable_alternatives": ["soybean"],
                "fertilizer_keywords": ["split nitrogen", "dap basal", "potash"],
                "irrigation_keywords": ["furrow", "critical growth stage", "rainfall adjusted"],
                "max_feasible_cost_usd": 9800,
                "unsafe_keywords": ["continuous flooding", "single heavy urea dose"],
            },
        },
        {
            "task_id": "hard-02",
            "difficulty": "hard",
            "instruction": "Build a seasonal farm plan under resource and budget constraints.",
            "input_data": {
                "soil_profile": {"texture": "sandy loam", "pH": 7.7, "organic_matter_pct": 0.7},
                "weather_forecast": {"season": "kharif", "rainfall_mm": 210, "mean_temp_c": 33},
                "irrigation_availability": "low",
                "fertilizer_availability": ["urea", "SSP", "MOP", "farmyard manure"],
                "farm_size_ha": 18,
                "budget_usd": 7200,
            },
            "expected_output": {
                "primary_crop": "pearl millet",
                "acceptable_alternatives": ["sorghum", "pigeon pea"],
                "fertilizer_keywords": ["moderate nitrogen", "ssp basal", "organic manure"],
                "irrigation_keywords": ["drip", "deficit irrigation", "mulching"],
                "max_feasible_cost_usd": 7200,
                "unsafe_keywords": ["water intensive paddy", "flood irrigation"],
            },
        },
    ],
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()


def _contains_any(text: str, phrases: List[str]) -> bool:
    ntext = _norm(text)
    return any(_norm(p) in ntext for p in phrases)


def _keyword_fraction(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    ntext = _norm(text)
    hits = sum(1 for k in keywords if _norm(k) in ntext)
    return hits / len(keywords)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


class AgriOpsEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._state = AgriOpsState()

    def _pick_instance(self, task_name: str, seed: Optional[int]) -> Dict[str, Any]:
        instances = TASK_BANK[task_name]
        idx = (seed or 0) % len(instances)
        return instances[idx]

    def _record_action(self, action_type: str) -> float:
        count = self._state.repeated_actions.get(action_type, 0)
        self._state.repeated_actions[action_type] = count + 1
        if count == 0:
            return 0.0
        return -min(0.12, 0.03 * count)

    def _build_obs(
        self,
        reward: Optional[float],
        done: bool,
        message: str,
        diagnosis_feedback: Optional[str] = None,
        intervention_feedback: Optional[str] = None,
        plan_component_scores: Optional[Dict[str, float]] = None,
    ) -> AgriOpsObservation:
        return AgriOpsObservation(
            done=done,
            reward=reward,
            task=self._state.task_name,
            task_id=self._state.task_id,
            instruction=self._state.instruction,
            input_data=self._state.input_data,
            message=message,
            diagnosis_feedback=diagnosis_feedback,
            intervention_feedback=intervention_feedback,
            plan_component_scores=plan_component_scores,
            checked_so_far={
                "repeated_actions": dict(self._state.repeated_actions),
                "diagnosis_submitted": self._state.diagnosis_submitted,
                "intervention_submitted": self._state.intervention_submitted,
                "cumulative_reward": round(self._state.cumulative_reward, 4),
            },
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> AgriOpsObservation:
        import os

        task_name = (task or os.getenv("AGRIOPS_TASK") or "easy").lower()
        if task_name not in TASK_BANK:
            task_name = "easy"

        instance = self._pick_instance(task_name, seed)
        self._state = AgriOpsState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
            task_id=instance["task_id"],
            instruction=instance["instruction"],
            input_data=instance["input_data"],
            expected_output=instance["expected_output"],
            cumulative_reward=0.0,
            repeated_actions={},
            diagnosis_submitted=False,
            intervention_submitted=False,
            diagnosis_score=0.0,
            intervention_score=0.0,
            latest_plan_score=0.0,
        )

        if task_name == "easy":
            message = (
                "Task easy: Crop Selection Under Field Constraints. "
                "Use action_type='analyze_field' for context, then action_type='recommend_crop' "
                "with crop_recommendation='<crop_name>'."
            )
        elif task_name == "medium":
            message = (
                "Task medium: Field Issue Diagnosis and Intervention. "
                "Use action_type='diagnose_issue' with diagnosis text, "
                "action_type='recommend_intervention' with intervention text, "
                "then action_type='finalize_case'."
            )
        else:
            message = (
                "Task hard: Seasonal Farm Planning Under Constraints. "
                "Use action_type='analyze_constraints', optional action_type='propose_plan', "
                "and finish with action_type='submit_plan' including crop_recommendation, "
                "fertilizer_strategy, irrigation_strategy, and estimated_cost."
            )

        return self._build_obs(reward=None, done=False, message=message)

    def step(
        self,
        action: AgriOpsAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AgriOpsObservation:
        self._state.step_count += 1
        if self._state.task_name == "easy":
            return self._step_easy(action)
        if self._state.task_name == "medium":
            return self._step_medium(action)
        return self._step_hard(action)

    def _step_easy(self, action: AgriOpsAction) -> AgriOpsObservation:
        repeat_penalty = self._record_action(action.action_type)

        if action.action_type == "analyze_field":
            novelty_bonus = 0.1 if self._state.repeated_actions[action.action_type] == 1 else 0.0
            reward = round(novelty_bonus + repeat_penalty, 4)
            self._state.cumulative_reward += reward
            return self._build_obs(
                reward=reward,
                done=False,
                message=(
                    "Field analysis acknowledged. Prioritize crop-season-water fit and nutrient demand. "
                    "Next: recommend_crop."
                ),
            )

        if action.action_type != "recommend_crop":
            reward = round(-0.1 + repeat_penalty, 4)
            self._state.cumulative_reward += reward
            return self._build_obs(
                reward=reward,
                done=False,
                message="Invalid action for easy task. Use analyze_field or recommend_crop.",
            )

        crop = (action.crop_recommendation or "").strip().lower()
        gold = self._state.expected_output
        primary = gold["primary_crop"]
        alternatives = [c.lower() for c in gold["acceptable_alternatives"]]
        unsafe = [c.lower() for c in gold.get("unsafe_crops", [])]

        if not crop:
            reward = round(-0.12 + repeat_penalty, 4)
            self._state.cumulative_reward += reward
            return self._build_obs(
                reward=reward,
                done=False,
                message="recommend_crop requires crop_recommendation.",
            )

        if crop == primary.lower():
            score = 1.0
        elif crop in alternatives:
            score = 0.8
        elif crop in unsafe:
            score = 0.1
        else:
            score = 0.35

        reward = round(_clip01(0.9 * score) + repeat_penalty, 4)
        self._state.cumulative_reward += reward
        return self._build_obs(
            reward=reward,
            done=True,
            message=(
                f"Crop graded. recommended={crop}, primary={primary}, alternatives={alternatives}. "
                f"score={score:.2f}, step_reward={reward:.2f}, cumulative={self._state.cumulative_reward:.2f}."
            ),
        )

    def _diagnosis_score(self, text: str) -> Tuple[float, str]:
        expected = self._state.expected_output
        diagnosis_list = expected["diagnosis"]
        if _contains_any(text, diagnosis_list):
            return 1.0, "Diagnosis matches expected issue."
        kw = _keyword_fraction(text, expected.get("diagnosis_keywords", []))
        if kw >= 0.5:
            return 0.6, "Diagnosis is partially correct."
        return 0.0, "Diagnosis does not match expected issue."

    def _intervention_score(self, text: str) -> Tuple[float, str, float]:
        expected = self._state.expected_output
        unsafe_penalty = -0.25 if _contains_any(text, expected.get("unsafe_interventions", [])) else 0.0
        kw = _keyword_fraction(text, expected.get("intervention_keywords", []))
        score = _clip01(kw + (0.2 if kw >= 0.67 else 0.0))
        if score >= 0.8:
            feedback = "Intervention aligns with expected treatment plan."
        elif score >= 0.4:
            feedback = "Intervention is partially aligned; key specifics are missing."
        else:
            feedback = "Intervention is not aligned with expected treatment."
        return score, feedback, unsafe_penalty

    def _step_medium(self, action: AgriOpsAction) -> AgriOpsObservation:
        repeat_penalty = self._record_action(action.action_type)

        if action.action_type == "analyze_case":
            novelty_bonus = 0.1 if self._state.repeated_actions[action.action_type] == 1 else 0.0
            reward = round(novelty_bonus + repeat_penalty, 4)
            self._state.cumulative_reward += reward
            return self._build_obs(
                reward=reward,
                done=False,
                message="Case analysis noted. Submit diagnosis, intervention, then finalize_case.",
            )

        if action.action_type == "diagnose_issue":
            text = action.diagnosis or ""
            if not text.strip():
                reward = round(-0.1 + repeat_penalty, 4)
                self._state.cumulative_reward += reward
                return self._build_obs(reward=reward, done=False, message="diagnose_issue requires diagnosis text.")

            score, feedback = self._diagnosis_score(text)
            self._state.diagnosis_submitted = True
            self._state.diagnosis_score = score
            reward = round(0.45 * score + repeat_penalty, 4)
            self._state.cumulative_reward += reward
            return self._build_obs(
                reward=reward,
                done=False,
                message="Diagnosis received. Next: recommend_intervention and finalize_case.",
                diagnosis_feedback=feedback,
            )

        if action.action_type == "recommend_intervention":
            text = action.intervention or ""
            if not text.strip():
                reward = round(-0.1 + repeat_penalty, 4)
                self._state.cumulative_reward += reward
                return self._build_obs(
                    reward=reward,
                    done=False,
                    message="recommend_intervention requires intervention text.",
                )

            score, feedback, unsafe_penalty = self._intervention_score(text)
            self._state.intervention_submitted = True
            self._state.intervention_score = score
            precheck_penalty = -0.05 if not self._state.diagnosis_submitted else 0.0
            reward = round(0.45 * score + unsafe_penalty + precheck_penalty + repeat_penalty, 4)
            self._state.cumulative_reward += reward
            return self._build_obs(
                reward=reward,
                done=False,
                message="Intervention received. Finalize with action_type='finalize_case'.",
                intervention_feedback=feedback,
            )

        if action.action_type == "finalize_case":
            if not (self._state.diagnosis_submitted and self._state.intervention_submitted):
                reward = round(-0.12 + repeat_penalty, 4)
                self._state.cumulative_reward += reward
                return self._build_obs(
                    reward=reward,
                    done=False,
                    message="Finalize requires both diagnosis and intervention submissions.",
                )

            combined = 0.5 * self._state.diagnosis_score + 0.5 * self._state.intervention_score
            bonus = 0.1 if combined >= 0.8 else 0.0
            reward = round(0.1 + bonus + repeat_penalty, 4)
            self._state.cumulative_reward += reward
            return self._build_obs(
                reward=reward,
                done=True,
                message=(
                    f"Case closed. diagnosis_score={self._state.diagnosis_score:.2f}, "
                    f"intervention_score={self._state.intervention_score:.2f}, "
                    f"final_step_reward={reward:.2f}, cumulative={self._state.cumulative_reward:.2f}."
                ),
            )

        reward = round(-0.1 + repeat_penalty, 4)
        self._state.cumulative_reward += reward
        return self._build_obs(
            reward=reward,
            done=False,
            message="Invalid action for medium task. Use analyze_case, diagnose_issue, recommend_intervention, finalize_case.",
        )

    def _score_plan(self, action: AgriOpsAction) -> Tuple[Dict[str, float], float, float, str]:
        expected = self._state.expected_output
        crop = (action.crop_recommendation or "").strip().lower()
        fert = action.fertilizer_strategy or ""
        irr = action.irrigation_strategy or ""
        cost = action.estimated_cost

        primary = expected["primary_crop"].lower()
        alternatives = [c.lower() for c in expected.get("acceptable_alternatives", [])]

        if crop == primary:
            crop_score = 1.0
        elif crop in alternatives:
            crop_score = 0.7
        elif crop:
            crop_score = 0.2
        else:
            crop_score = 0.0

        fert_score = _keyword_fraction(fert, expected.get("fertilizer_keywords", []))
        irr_score = _keyword_fraction(irr, expected.get("irrigation_keywords", []))

        max_cost = float(expected.get("max_feasible_cost_usd", 0))
        if cost is None:
            budget_score = 0.0
        elif cost <= max_cost:
            budget_score = 1.0
        else:
            over = (cost - max_cost) / max(max_cost, 1.0)
            budget_score = _clip01(1.0 - over)

        unsafe_hit = _contains_any(
            " ".join([crop, fert, irr]),
            expected.get("unsafe_keywords", []),
        )
        unsafe_penalty = -0.2 if unsafe_hit else 0.0

        weighted = (
            0.35 * crop_score
            + 0.25 * fert_score
            + 0.20 * irr_score
            + 0.20 * budget_score
        )
        msg = "Plan evaluated against crop, fertilizer, irrigation, and budget constraints."
        return (
            {
                "crop_suitability": round(crop_score, 4),
                "fertilizer_suitability": round(fert_score, 4),
                "irrigation_feasibility": round(irr_score, 4),
                "budget_compliance": round(budget_score, 4),
            },
            round(weighted, 4),
            unsafe_penalty,
            msg,
        )

    def _step_hard(self, action: AgriOpsAction) -> AgriOpsObservation:
        repeat_penalty = self._record_action(action.action_type)

        if action.action_type == "analyze_constraints":
            novelty_bonus = 0.1 if self._state.repeated_actions[action.action_type] == 1 else 0.0
            reward = round(novelty_bonus + repeat_penalty, 4)
            self._state.cumulative_reward += reward
            return self._build_obs(
                reward=reward,
                done=False,
                message="Constraints reviewed. Provide plan via propose_plan or submit_plan.",
            )

        if action.action_type in ("propose_plan", "submit_plan"):
            components, plan_score, unsafe_penalty, detail = self._score_plan(action)
            self._state.latest_plan_score = plan_score

            if action.action_type == "propose_plan":
                reward = round(0.5 * plan_score + unsafe_penalty + repeat_penalty, 4)
                self._state.cumulative_reward += reward
                return self._build_obs(
                    reward=reward,
                    done=False,
                    message=(
                        f"Draft plan scored. plan_score={plan_score:.2f}. Improve weak components and submit final plan."
                    ),
                    plan_component_scores=components,
                )

            completion_bonus = 0.15 if components["budget_compliance"] >= 1.0 and components["irrigation_feasibility"] >= 0.6 else 0.0
            infeasible_penalty = -0.15 if components["budget_compliance"] < 0.4 else 0.0
            reward = round(0.7 * plan_score + completion_bonus + unsafe_penalty + infeasible_penalty + repeat_penalty, 4)
            self._state.cumulative_reward += reward
            return self._build_obs(
                reward=reward,
                done=True,
                message=(
                    f"Final plan submitted. {detail} plan_score={plan_score:.2f}, "
                    f"step_reward={reward:.2f}, cumulative={self._state.cumulative_reward:.2f}."
                ),
                plan_component_scores=components,
            )

        reward = round(-0.1 + repeat_penalty, 4)
        self._state.cumulative_reward += reward
        return self._build_obs(
            reward=reward,
            done=False,
            message="Invalid action for hard task. Use analyze_constraints, propose_plan, or submit_plan.",
        )

    @property
    def state(self) -> AgriOpsState:
        return self._state
