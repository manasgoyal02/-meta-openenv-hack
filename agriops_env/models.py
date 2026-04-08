from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State


class AgriOpsAction(Action):
    action_type: str

    # Shared optional fields across tasks
    task_id: Optional[str] = None
    notes: Optional[str] = None

    # Easy task
    crop_recommendation: Optional[str] = None

    # Medium task
    diagnosis: Optional[str] = None
    intervention: Optional[str] = None

    # Hard task
    candidate_crops: Optional[List[str]] = None
    fertilizer_strategy: Optional[str] = None
    irrigation_strategy: Optional[str] = None
    estimated_cost: Optional[float] = None


class AgriOpsObservation(Observation):
    task: str
    task_id: str
    instruction: str
    input_data: Dict[str, Any]
    message: str

    # Optional feedback fields
    diagnosis_feedback: Optional[str] = None
    intervention_feedback: Optional[str] = None
    plan_component_scores: Optional[Dict[str, float]] = None
    checked_so_far: Optional[Dict[str, Any]] = None


class AgriOpsState(State):
    task_name: str = "easy"
    task_id: str = ""
    instruction: str = ""
    input_data: Dict[str, Any] = {}
    expected_output: Dict[str, Any] = {}

    # Progress and grading
    cumulative_reward: float = 0.0
    repeated_actions: Dict[str, int] = {}
    diagnosis_submitted: bool = False
    intervention_submitted: bool = False
    diagnosis_score: float = 0.0
    intervention_score: float = 0.0
    latest_plan_score: float = 0.0
