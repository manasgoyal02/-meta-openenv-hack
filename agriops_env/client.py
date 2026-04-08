from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import AgriOpsAction, AgriOpsObservation, AgriOpsState


class AgriOpsEnv(EnvClient[AgriOpsAction, AgriOpsObservation, AgriOpsState]):
    def _step_payload(self, action: AgriOpsAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        obs_data = payload.get("observation", {})
        obs = AgriOpsObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            task=obs_data.get("task", "easy"),
            task_id=obs_data.get("task_id", ""),
            instruction=obs_data.get("instruction", ""),
            input_data=obs_data.get("input_data", {}),
            message=obs_data.get("message", ""),
            diagnosis_feedback=obs_data.get("diagnosis_feedback"),
            intervention_feedback=obs_data.get("intervention_feedback"),
            plan_component_scores=obs_data.get("plan_component_scores"),
            checked_so_far=obs_data.get("checked_so_far"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AgriOpsState:
        return AgriOpsState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", "easy"),
            task_id=payload.get("task_id", ""),
            instruction=payload.get("instruction", ""),
            input_data=payload.get("input_data", {}),
            expected_output=payload.get("expected_output", {}),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            repeated_actions=payload.get("repeated_actions", {}),
            diagnosis_submitted=payload.get("diagnosis_submitted", False),
            intervention_submitted=payload.get("intervention_submitted", False),
            diagnosis_score=payload.get("diagnosis_score", 0.0),
            intervention_score=payload.get("intervention_score", 0.0),
            latest_plan_score=payload.get("latest_plan_score", 0.0),
        )
