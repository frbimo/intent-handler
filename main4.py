from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import json
from datetime import datetime, timezone, timedelta # Ensure timezone aware datetimes
from uuid import uuid4
from enum import Enum
# from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles # Not used in provided snippet but kept if needed elsewhere
from src.netconf_cli import NETCONFCLIENT
from src.pm_data import get_measurement_data

import re
import logging
import os
import time # Added for sleep functionality

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from openai import OpenAI
# for debugging
import inspect

# --- Configuration Constants for Closed Loop ---
MIN_WAIT_FIRST_PM_CHECK_SECONDS = 65
PM_CHECK_INTERVAL_SECONDS = 60
OBSERVATION_PERIOD_SECONDS = 125
MAX_PM_CHECK_CYCLES_PER_STRATEGY = 10
MAX_TOTAL_REFINEMENT_ATTEMPTS_FOR_INTENT = 7

# Condition and fulfillment enums
class ConditionEnum(str, Enum):
    IS_EQUAL_TO = "IS_EQUAL_TO"
    IS_LESS_THAN = "IS_LESS_THAN"
    IS_GREATER_THAN = "IS_GREATER_THAN"
    IS_WITHIN_RANGE = "IS_WITHIN_RANGE"
    IS_OUTSIDE_RANGE = "IS_OUTSIDE_RANGE"
    IS_ONE_OF = "IS_ONE_OF"
    IS_NOT_ONE_OF = "IS_NOT_ONE_OF"
    IS_EQUAL_TO_OR_LESS_THAN = "IS_EQUAL_TO_OR_LESS_THAN"
    IS_EQUAL_TO_OR_GREATER_THAN = "IS_EQUAL_TO_OR_GREATER_THAN"
    IS_ALL_OF = "IS_ALL_OF"

class FulfilmentStatusEnum(str, Enum):
    FULFILLED = "FULFILLED"
    NOT_FULFILLED = "NOT_FULFILLED"

class NotFulfilledStateEnum(str, Enum):
    ACKNOWLEDGED = "ACKNOWLEDGED"
    COMPLIANT = "COMPLIANT"
    DEGRADED = "DEGRADED"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    FULFILMENTFAILED = "FULFILMENTFAILED"

class IntentState(BaseModel):
    initialized: bool = False # Indicates if the state has been initialized
    intent: Dict
    pm_data: Dict = {}
    history_summary: Dict = {} # Still useful for the *initial* strategy proposal and context for refinements
    
    # --- Details for the SINGLE strategy type being worked on for this intent ---
    current_strategy_type: Optional[str] = None     # The chosen strategy type (e.g., "PowerOptimization") - SET ONCE
    current_config_parameters: Optional[Dict] = None # The latest parameters applied/to be applied for this type
    strategies: Dict = {}                           # Holds the full strategy structure with current_config_parameters
    patches: Dict = {}                              # Derived NETCONF patches

    # --- Iteration Tracking for THIS INTENT'S LIFECYCLE ---
    total_refinement_attempts_for_intent: int = 0 # Counts how many times we've tried to apply a config (initial or refined)

    # --- Outcome and Reporting ---
    outcomes: List[Dict] = []
    report: Dict = {}
    fulfilmentStatus: str = "NONE"

    # --- State for Closed-Loop Monitoring of a SINGLE APPLIED CONFIGURATION ---
    config_applied_successfully: bool = False
    pm_evaluation_pending: bool = False
    current_pm_check_cycle: int = 0 # For PM checks within one config's observation
    last_config_application_time: Optional[datetime] = None
    applied_strategy_id: str = "NONE"
    # --- Calculated Metrics & Deadline ---
    calculated_ee_metrics: Optional[Dict[str, Any]] = None
    intent_processing_deadline: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True

# --- Mocked or Example Data (Replace with your actual data sources) ---
CELL_CAPABILITIES = {
    "1": {"configurableParameters": {"parameters": {"ChannelBandwidthUL": {}, "ChannelBandwidthDL": {}}}},
    "2": {"configurableParameters": {"parameters": {"configuredMaxTxPower": {}}}},
    "3": {"configurableParameters": {"parameters": {"configuredMaxTxPower": {}}}},
    "7": {"configurableParameters": {"parameters": {"configuredMaxTxPower": {}}}}

}
STRATEGY_DATABASE = {
    "PowerOptimization": {"configType": "EnergySavingConfig"},
    "ThroughputOptimization": {"configType": "BandwidthConfig"}
}
CONFIG_TEMPLATES = {"ThroughputOptimization": {"parameters": {}}}
STRATEGY_ATTEMPTS = {}

# --- Helper Functions ---
def store_attempt(state: IntentState, intent_id: str) -> None:
    """
    Stores the details of the current intent processing attempt, both locally
    and potentially by updating an external Intent Management API.
    This is typically called when a definitive FULFILLED or NOT_FULFILLED status
    is reached for the intent, or for a significant intermediate failure.
    """
    if not intent_id:
        logger.error("store_attempt called with no intent_id. Cannot store.")
        # Depending on desired robustness, could raise ValueError or just return
        return # raise ValueError("Invalid input for store_attempt: intent_id missing")

    logger.info(f"Storing attempt for intent {intent_id}. Current fulfilmentStatus: {state.fulfilmentStatus}, "
                f"TotalRefinements: {state.total_refinement_attempts_for_intent}, "
                f"StrategyType: '{state.current_strategy_type}'")

    attempt_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Prepare PM data snapshot for storage (ensure it's serializable)
    serializable_pm_data = {}
    if state.pm_data:
        if state.pm_data.get("error"):
            serializable_pm_data = {"error_in_pm_data_collection": state.pm_data.get("error")}
        else:
            try:
                # Attempt to serialize to catch any non-serializable types early
                json.dumps(state.pm_data)
                serializable_pm_data = state.pm_data
            except TypeError:
                logger.warning(f"PM data for intent {intent_id} (attempt {attempt_id}) is not fully JSON serializable. Storing summary/error.")
                serializable_pm_data = {"summary": "PM data present but contained non-serializable elements.",
                                        "keys_present": list(state.pm_data.keys())}
    else:
        serializable_pm_data = {"info": "No PM data available at the time of storing this attempt."}

    # Extract failure analysis details from the report
    failure_analysis_detail = {}
    if state.report and state.report.get("expectationFulfilmentResult"):
        # Assuming the first entry in expectationFulfilmentResult holds the most relevant overall failure
        first_result = state.report["expectationFulfilmentResult"][0] if state.report["expectationFulfilmentResult"] else {}
        failure_analysis_detail = first_result.get("failureAnalysis", {})
    elif state.report and state.report.get("failureAnalysis"): # If report has direct failureAnalysis
        failure_analysis_detail = state.report.get("failureAnalysis", {})
    elif state.report and state.report.get("reason"): # Simpler report structure
         failure_analysis_detail = {"reason": state.report.get("reason")}


    # Context about the closed-loop monitoring for the *last applied configuration*
    # This provides insight into why a specific configuration succeeded or failed its observation.
    closed_loop_context_of_last_config = {}
    if state.last_config_application_time: # Indicates a configuration was applied and potentially monitored
        closed_loop_context_of_last_config = {
            "parameters_evaluated": state.current_config_parameters, # Parameters that led to current outcomes/pm_data
            "pm_checks_done_for_this_config": state.current_pm_check_cycle,
            "observation_duration_seconds": (datetime.now(timezone.utc) - state.last_config_application_time).total_seconds()
                                            if state.fulfilmentStatus != "FULFILLED" and state.fulfilmentStatus != "NOT_FULFILLED" else None, # Duration if it didn't complete full observation
            "reason_if_not_fulfilled_this_config": failure_analysis_detail.get("reason") if state.fulfilmentStatus != "FULFILLED" else None
        }


    attempt_data = {
        "intent_id": intent_id,
        "attempt_id": attempt_id, # Good to have its own ID
        "timestamp": timestamp,
        "fulfilment_status_at_storage": state.fulfilmentStatus,

        # Details about the strategy and parameters at the point of this stored attempt
        "strategy_type_attempted": state.current_strategy_type,
        "config_parameters_at_storage": state.current_config_parameters, # The last set of params tried
        "strategy_details_json": state.strategies if state.strategies else {"info": "No detailed strategy structure at storage."},
        
        # Iteration counts
        "total_refinement_attempts_for_intent_at_storage": state.total_refinement_attempts_for_intent,
        
        # Observational data
        "outcomes_at_storage": state.outcomes if state.outcomes else [{"info": "No specific outcomes recorded."}],
        "pm_data_snapshot_at_decision": serializable_pm_data,
        "calculated_ee_metrics_at_storage": state.calculated_ee_metrics,

        # Context from the last monitoring cycle
        "closed_loop_context_of_last_config": closed_loop_context_of_last_config,
        
        # Overall failure reason for this stored state
        "overall_failure_analysis": failure_analysis_detail,
        
        # Static context (consider if needed for every attempt, can be large)
        # "cell_capabilities_snapshot": CELL_CAPABILITIES, # Assuming CELL_CAPABILITIES is global
    }

    # Store locally in STRATEGY_ATTEMPTS
    # Assuming STRATEGY_ATTEMPTS is a global dictionary
    global STRATEGY_ATTEMPTS
    STRATEGY_ATTEMPTS[attempt_id] = attempt_data
    logger.info(f"Stored attempt {attempt_id} locally for intent {intent_id}. Status: {state.fulfilmentStatus}.")
    logger.debug(f"Stored attempt data: {json.dumps(attempt_data, indent=2, default=str)}") # default=str for datetime

    # --- Update the external Intent Management API ---
    # The report payload for the API should align with what the API expects.
    # It typically includes the overall fulfilmentStatus and a structured report (like state.report).
    
    # Construct the report that route_main_agent should have built
    api_report_content = state.report if state.report else {}
    if not api_report_content.get("intentId"):
        api_report_content["intentId"] = intent_id # Ensure intentId is in the report
    if not api_report_content.get("expectationFulfilmentResult") and state.outcomes:
        # If a full report structure wasn't built but we have outcomes, create a basic one
        results = []
        for out_eval in state.outcomes:
            orig_target = next((t for t in state.intent["intent"]["intentExpectation"]["expectationTargets"] if t["targetName"] == out_eval.get("targetName")), None)
            results.append({
                "targetName": out_eval.get("targetName"),
                "targetStatus": out_eval.get("fulfilmentStatus", "NOT_FULFILLED"),
                "targetAchievedValue": out_eval.get("targetAchievedValue"),
                "targetValue": orig_target.get("targetValue") if orig_target else "N/A",
                "failureAnalysis": {"reason": out_eval.get("evaluation_details")} if out_eval.get("fulfilmentStatus") != "FULFILLED" else {}
            })
        api_report_content["expectationFulfilmentResult"] = results
    
    # If even after that, report is minimal, ensure a basic failure reason is there for NOT_FULFILLED
    if state.fulfilmentStatus == "NOT_FULFILLED" and not api_report_content.get("expectationFulfilmentResult"):
        reason = "Processing concluded with NOT_FULFILLED status without detailed outcome breakdown."
        if failure_analysis_detail.get("reason"):
            reason = failure_analysis_detail.get("reason")
        elif state.report.get("reason"): # For very simple error reports
             reason = state.report.get("reason")

        api_report_content["expectationFulfilmentResult"] = [{
            "targetStatus": "NOT_FULFILLED",
            "failureAnalysis": {"reason": reason }
        }]


    api_payload_for_update = {
        "fulfilmentStatus": state.fulfilmentStatus,
        "report": api_report_content # state.report should contain expectationFulfilmentResult etc.
    }

    # Assuming intent_client is globally available or passed appropriately
    # from src.intent_api_client import intent_client
    if intent_id and api_payload_for_update.get("report"):
        logger.info(f"Attempting to update report via API for intent {intent_id} with status {state.fulfilmentStatus}")
        # success = intent_client.update_intent_report(intent_id, api_payload_for_update) # UNCOMMENT WHEN API CLIENT IS READY
        success = True # MOCKING API UPDATE SUCCESS
        if success:
            logger.info(f"Successfully updated intent {intent_id} report via API.")
        else:
            logger.warning(f"Failed to update intent {intent_id} report via API. Report stored locally only.")
    else:
        logger.warning(f"Skipping API report update for intent {intent_id}: missing intent_id or essential report data in state.")

base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")

if not base_url or not api_key:
    logger.warning("OPENAI_BASE_URL or OPENAI_API_KEY environment variable not set. LLM calls will fail.")
    # raise ValueError("OPENAI_BASE_URL or OPENAI_API_KEY environment variable not set.") # Uncomment to make it fatal

client = OpenAI(base_url=base_url, api_key=api_key) if base_url and api_key else None

def call_llm(messages: List[Dict]) -> Dict:
    if not client:
        logger.error("OpenAI client not initialized. Cannot call LLM.")
        return {"error": "OpenAI client not initialized."}
    logger.info(f"Calling LLM. First message content: {messages[0].get('content', '')[:100]}...")
    try:
        response = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct", messages=messages, max_tokens=4000, temperature=0.2, top_p=0.7
        )
        llm_resp = response.choices[0].message.content
        logger.debug(f"LLM raw response: {llm_resp}")
        match = re.search(r"```json\s*(\{.*?\})\s*```", llm_resp, re.DOTALL)
        if not match: match = re.search(r"(\{.*?\})", llm_resp, re.DOTALL)
        if match:
            json_string = match.group(1)
            print(json_string)
            try:
                data = json.loads(json_string)
                logger.info("LLM call successful, JSON parsed.")
                return data
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from LLM: {e}. String: {json_string}")
                return {"error": "Error decoding JSON", "details": str(e), "problematic_string": json_string}
        else:
            logger.warning("No JSON block found in the LLM response.")
            return {"explanation_only": llm_resp, "error": "No JSON block found"}
    except Exception as e:
        logger.error(f"Exception during LLM call: {e}")
        return {"error": f"LLM API call failed: {str(e)}"}

def calculate_energy_efficiency(pm_data: Dict) -> Optional[float]:
    pdcp_ul_key = "QosFlow.TotPdcpPduVolumeUl" #Mbits 3000000
    pdcp_dl_key = "QosFlow.TotPdcpPduVolumeDl" #Mbits 400000
    # power_key = "PEE.AvgPower"
    power_key = "PEE.Energy" # kWh 0.02 

    if pdcp_ul_key in pm_data and pdcp_dl_key in pm_data and power_key in pm_data:
        try:
            total_pdcp_volume_kbit = (float(pm_data[pdcp_ul_key]) + float(pm_data[pdcp_dl_key]))*1000 # Convert Mbits to kbits
            total_power = float(pm_data[power_key])
            if total_power > 0:
                ee = total_pdcp_volume_kbit / total_power
                logger.info(f"Calculated EE: {total_pdcp_volume_kbit} / {total_power} = {ee}")
                return ee
            else:
                logger.warning("Cannot calculate EE: Total power consumption is zero or not positive.")
                return 0.0 if total_pdcp_volume_kbit == 0 else None # Or a marker for very high EE
        except ValueError:
            logger.error(f"Could not convert PM data for EE calculation to float.")
            return None
    else:
        missing = [k for k in [pdcp_ul_key, pdcp_dl_key, power_key] if k not in pm_data]
        logger.warning(f"Cannot calculate EE: Missing PM data keys: {', '.join(missing)}")
        return None

# --- Agent Nodes ---
def history_agent(state: IntentState) -> IntentState:
    logger.info(f"Executing {inspect.currentframe().f_code.co_name}")
    intent = state.intent
    current_intent_id = intent.get("intent", {}).get("id", "UNKNOWN_INTENT_ID")
    expectation_targets = intent.get("intent", {}).get("intentExpectation", {}).get("expectationTargets", [])
    intent_type_description = [t["targetName"] for t in expectation_targets if "targetName" in t]

    recent_attempts_current_intent = []
    for attempt_id, attempt_data in reversed(list(STRATEGY_ATTEMPTS.items())):
        if attempt_data.get("intent_id") == current_intent_id:
            recent_attempts_current_intent.append({
                "attempt_id": attempt_id,
                "strategy_details": attempt_data.get("strategy_json"),
                "final_outcomes": attempt_data.get("outcome_json"),
                "failure_analysis": attempt_data.get("failure_analysis"),
                "fulfilment_status": attempt_data.get("fulfilment_status_at_storage"),
                "closed_loop_context": attempt_data.get("closed_loop_context", {}),
                "timestamp": attempt_data.get("timestamp")
            })
            if len(recent_attempts_current_intent) >= 3: break

    general_recent_attempts = []
    current_intent_attempt_ids = {att["attempt_id"] for att in recent_attempts_current_intent}
    for attempt_id, attempt_data in reversed(list(STRATEGY_ATTEMPTS.items())):
        if attempt_id not in current_intent_attempt_ids:
            general_recent_attempts.append({
                "attempt_id": attempt_id, "strategy_details": attempt_data.get("strategy_json"),
                "final_outcomes": attempt_data.get("outcome_json"), "failure_analysis": attempt_data.get("failure_analysis"),
                "fulfilment_status": attempt_data.get("fulfilment_status_at_storage"), "timestamp": attempt_data.get("timestamp")
            })
            if len(general_recent_attempts) >= 5: break

    history_prompt_context = {
        "recent_attempts_for_this_intent": recent_attempts_current_intent,
        "other_recent_general_attempts": general_recent_attempts[:3]
    }

    user_prompt = f"""
As a 5G network analysis expert, analyze historical strategy attempts to inform new strategy generation for the current intent.
Current Intent ID: {current_intent_id}
Current Intent Type/Targets: {json.dumps(intent_type_description)}
Historical Context: {json.dumps(history_prompt_context, indent=2)}

Instructions for LLM:
1. Focus on "recent_attempts_for_this_intent". Analyze strategies, failures ("failure_analysis", "closed_loop_context"), and successes.
2. If a strategy for *this intent* recently failed (status NOT_FULFILLED, especially with `closed_loop_context` showing `pm_checks_done > 0`), the new strategy should try a *different approach*.
3. "failure_analysis" and "closed_loop_context.reason_if_not_fulfilled" are key to understanding *why* a monitored strategy failed.
4. "other_recent_general_attempts" provide broader context.
5. Summarize key learnings and suggest directions or warnings for the *next* strategy.

Output JSON with:
- result: {{
    "key_learnings": ["string summary of what was learned"],
    "suggested_strategy_directions": ["string, e.g., 'Consider bandwidth adjustment.'"],
    "warnings": ["string, e.g., 'Strategy type X consistently failed.'"]
  }}
- explanation: "String briefly explaining your derivation."
- error: Null or error message.
"""
    messages = [{"role": "system", "content": "You provide historical insights for 5G strategy. Output structured JSON."}, {"role": "user", "content": user_prompt}]
    response = call_llm(messages)
    if response.get("result") and not response.get("error"):
        state.history_summary = response
    else:
        logger.warning(f"History agent LLM error: {response.get('error', 'No result')}")
        state.history_summary = {"result": {"key_learnings": [], "suggested_strategy_directions": [], "warnings": ["Failed to process history via LLM."]},
                               "explanation": "LLM processing failed.", "error": response.get('error', 'Unknown')}
    return state

def strategy_agent(state: IntentState) -> IntentState:
    logger.info(f"Executing strategy_agent. CurrentStrategyType='{state.current_strategy_type}', "
                f"TotalRefinementAttemptsSoFar={state.total_refinement_attempts_for_intent}")
    
   
    intent = state.intent
    intent_id = intent.get("intent", {}).get("id", "UNKNOWN_INTENT_ID")
    
    
     # mark intent initialized
    if not state.initialized:
        state.initialized = True
        logger.info(f"StrategyAgent initialized for intent {state.intent.get('intent', {}).get('id', 'UNKNOWN_INTENT_ID')}.")


    elif not state.pm_data or state.pm_data.get("error"):
        logger.error(f"StrategyAgent for {intent_id}: PM data error: {state.pm_data.get('error', 'Missing')}. Cannot proceed.")
        state.fulfilmentStatus = "NOT_FULFILLED" # Fatal for this path
        state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"reason": f"PM data error for strategy: {state.pm_data.get('error', 'Missing')}"}}]}
        return state

    # --- Phase 1: Propose INITIAL Strategy Type and Parameters ---
    if state.current_strategy_type is None:
        logger.info(f"StrategyAgent for {intent_id}: Phase 1 - Proposing INITIAL strategy type and parameters.")
        if state.total_refinement_attempts_for_intent >= MAX_TOTAL_REFINEMENT_ATTEMPTS_FOR_INTENT: # Should be 0 here
            logger.warning(f"Max total refinement attempts reached for {intent_id} even before initial proposal. This is unusual.")
            state.fulfilmentStatus = "NOT_FULFILLED"
            state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"reason": "Max total attempts reached pre-proposal."}}]}
            return state



        state.total_refinement_attempts_for_intent += 1 # Count this initial proposal as the first attempt

        history_insights = state.history_summary.get("result", {})
        targets = intent.get("intent", {}).get("intentExpectation", {}).get("expectationTargets", [])
        object_target = intent.get("intent", {}).get("intentExpectation", {}).get("expectationObject", {}).get("ObjectTarget", [])
        
        # Obtain runtime configuration parameters
        for cell_id_str in object_target:
            netconf_client = NETCONFCLIENT(cell_id_str, cell_id_str)
            state.current_config_parameters={"configuredMaxTxPower": netconf_client.get_current_tx_power()} 

        user_prompt_initial = f"""
Propose an INITIAL strategy type and configuration parameters for intent {intent_id}.
Intent Targets: {json.dumps(targets)}
Object Target: {object_target}
Obtained runtime configuration parameters: {json.dumps(state.current_config_parameters)}
Historical Insights: {json.dumps(history_insights, indent=2)}
Current PM Data: {json.dumps(state.pm_data)}
STRATEGY_DATABASE: {json.dumps(STRATEGY_DATABASE)}
CELL_CAPABILITIES: {json.dumps(CELL_CAPABILITIES)}

Instruction:
- Select ONE `strategyType` from STRATEGY_DATABASE that seems most appropriate.
- Propose initial `parameters` for this strategyType, considering PM data and CELL_CAPABILITIES. Parameters to be adjusted should be within the capabilities of the target cells.
- This will be the ONLY strategy type used and refined for this intent. Choose wisely.
output format: 
```json
{{ "result": {{ "chosen_strategy_type": "...", "initial_parameters": {{...}} }}, "explanation": "...", "error": null }}
```

example output:
```json
{{
  "result": {{
    "chosen_strategy_type": "PowerOptimization",
    "initial_parameters": {{
      "configuredMaxTxPower": 16
    }}
  }},
  "explanation": "...",
  "error": null
}}
```
"""
        messages = [{"role": "system", "content": "You are an AI proposing an initial 5G configuration strategy. Output JSON."}, {"role": "user", "content": user_prompt_initial}]
        response = call_llm(messages)

        if response.get("error") or not response.get("result") or \
           not response["result"].get("chosen_strategy_type") or not response["result"].get("initial_parameters"):
            logger.error(f"StrategyAgent (Initial) LLM error or invalid proposal for {intent_id}: {response.get('error', 'Invalid structure')}")
            state.fulfilmentStatus = "NOT_FULFILLED"
            state.total_refinement_attempts_for_intent -=1 # Revert count
            state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"reason": f"LLM failed to provide initial strategy: {response.get('error', 'Invalid')}"}}]}
            return state

        chosen_strategy_type = response["result"]["chosen_strategy_type"]
        initial_parameters = response["result"]["initial_parameters"]
        explanation = response.get("explanation", "N/A")
        logger.info(f"LLM Initial Strategy Explanation for {intent_id}: {explanation}")

        user_input = input(f"LLM proposed initial strategy type: '{chosen_strategy_type}' with params {initial_parameters}. Use this type for all refinements? ('yes' to confirm): ")
        if user_input.lower() == "yes":
            state.current_strategy_type = chosen_strategy_type
            state.current_config_parameters = initial_parameters
            # Construct the state.strategies and state.patches
            strategy_id = f"S_{chosen_strategy_type}_{str(uuid4())[:4]}"
            config_id = f"C_{chosen_strategy_type}_init"
            # Assuming strategy DB gives config type, or LLM includes it if not standard
            config_type_from_db = STRATEGY_DATABASE.get(chosen_strategy_type, {}).get("configType", "DefaultConfigType")

            state.strategies = {
                "appliedStrategies": [{
                    "strategyId": strategy_id,
                    "strategyType": chosen_strategy_type,
                    "configurations": [{
                        "configId": config_id,
                        "configType": config_type_from_db, # Or LLM can suggest if more flexible
                        "parameters": initial_parameters,
                        "affectedCells": object_target # Simplification, ensure these are valid
                    }]
                }]
            }
            state.patches = {} # Derive patches
            for cell in object_target: # Simplification
                if cell in CELL_CAPABILITIES:
                    state.patches[cell] = {config_type_from_db: initial_parameters}
            
            state.applied_strategy_id = strategy_id
            state.fulfilmentStatus = "ACKNOWLEDGED"
            logger.info(f"Initial strategy '{chosen_strategy_type}' for {intent_id} approved.")
        else:
            logger.info(f"User rejected initial strategy proposal for {intent_id}.")
            state.fulfilmentStatus = "NOT_FULFILLED" # User terminated
            state.total_refinement_attempts_for_intent -=1 # Revert
            state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"notFulfilledState": "TERMINATED", "reason": "User rejected initial strategy proposal"}}]}
        return state

    # --- Phase 2: REFINE Parameters for the EXISTING Strategy Type ---
    elif state.current_strategy_type:
        logger.info(f"StrategyAgent for {intent_id}: Phase 2 - Refining strategy type '{state.current_strategy_type}'. "
                    f"Next total refinement attempt for intent: {state.total_refinement_attempts_for_intent + 1}.")

        if state.total_refinement_attempts_for_intent >= MAX_TOTAL_REFINEMENT_ATTEMPTS_FOR_INTENT:
            logger.warning(f"Max total refinement attempts ({MAX_TOTAL_REFINEMENT_ATTEMPTS_FOR_INTENT}) reached for intent {intent_id} while trying to refine '{state.current_strategy_type}'.")
            state.fulfilmentStatus = "NOT_FULFILLED"
            state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"reason": f"Max total refinement attempts reached for strategy type '{state.current_strategy_type}'."}}]}
            return state # Router will handle final storage and END

        state.total_refinement_attempts_for_intent += 1

        # state.current_config_parameters holds the *last applied* parameters
        # state.pm_data and state.outcomes hold the results of *that last application*
        user_prompt_refine = f"""
Targets: {json.dumps(state.intent["intent"]["intentExpectation"]["expectationTargets"])}
Previous Parameters: {json.dumps(state.current_config_parameters)}
PM Data (post-application): {json.dumps(state.pm_data)}
Outcomes: {json.dumps(state.outcomes)}
Energy Efficiency Metrics: {json.dumps(state.calculated_ee_metrics)}
Parameter Ranges (CELL_CAPABILITIES): {json.dumps(CELL_CAPABILITIES)}
Strategy Reference (STRATEGY_DATABASE): {json.dumps(STRATEGY_DATABASE)}

Instructions:
- Analyze why the previous parameters underperformed.
- Propose new parameters (within CELL_CAPABILITIES) to improve target fulfillment.
- If no further improvement is feasible, set `give_up_on_this_strategy_type: true`.

Output format:
```json
{{ "result": {{ "refined_parameters": {{...}}, "give_up_on_this_strategy_type": false/true }}, "explanation": "...", "error": null }}
```
"""
        messages = [{"role": "system", "content": "You are an AI optimizing 5G strategies based on observed performance. Generate response in JSON."}, {"role": "user", "content": user_prompt_refine}]
        response = call_llm(messages)

        if response.get("error") or not response.get("result"):
            logger.error(f"StrategyAgent (Refine) LLM error for {intent_id}: {response.get('error', 'No result')}")
            state.fulfilmentStatus = "NOT_FULFILLED" # Refinement step failed
            state.total_refinement_attempts_for_intent -=1 # Revert count
            state.pm_evaluation_pending = False
            state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"reason": f"LLM failed to refine strategy '{state.current_strategy_type}'"}}]}
            return state

        explanation = response.get("explanation", "N/A")
        logger.info(f"LLM Refinement Explanation for {intent_id}: {explanation}")
        
        refined_parameters = response["result"].get("refined_parameters")
        give_up = response["result"].get("give_up_on_this_strategy_type", False)

        if give_up or not refined_parameters:
            logger.info(f"LLM indicated to give up on '{state.current_strategy_type}' or no refinement for {intent_id}.")
            state.fulfilmentStatus = "NOT_FULFILLED" # This strategy type is exhausted for this intent
            state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"reason": f"LLM gave up on/failed to refine '{state.current_strategy_type}' after {state.total_refinement_attempts_for_intent} total attempts."}}]}
            state.current_strategy_type = None # Mark as exhausted
            state.pm_evaluation_pending = False
            return state

        user_input = input(f"LLM proposed refined params {refined_parameters} for '{state.current_strategy_type}' (Total Attempt {state.total_refinement_attempts_for_intent}). Apply? ('yes' to confirm): ")
        if user_input.lower() == "yes":
            state.current_config_parameters = refined_parameters # Update with new params to be applied
            
            # Reconstruct state.strategies and state.patches with new parameters
            strategy_id = f"{state.applied_strategy_id}_iter{state.total_refinement_attempts_for_intent}" if state.applied_strategy_id else f"S_{state.current_strategy_type}_iter{state.total_refinement_attempts_for_intent}"
            config_id = f"C_{state.current_strategy_type}_iter{state.total_refinement_attempts_for_intent}"
            config_type_from_db = STRATEGY_DATABASE.get(state.current_strategy_type, {}).get("configType", "DefaultConfigType")
            object_target = state.intent.get("intent", {}).get("intentExpectation", {}).get("expectationObject", {}).get("ObjectTarget", [])

            state.strategies = {
                "appliedStrategies": [{
                    "strategyId": strategy_id,
                    "strategyType": state.current_strategy_type,
                    "configurations": [{
                        "configId": config_id,
                        "configType": config_type_from_db,
                        "parameters": refined_parameters,
                        "affectedCells": object_target
                    }]
                }]
            }
            state.patches = {} # Derive patches
            for cell in object_target:
                if cell in CELL_CAPABILITIES:
                    state.patches[cell] = {config_type_from_db: refined_parameters}
            
            state.applied_strategy_id = strategy_id # Update if changed
            state.fulfilmentStatus = "ACKNOWLEDGED" # Ready for orchestration of refined config
            state.config_applied_successfully = False # Reset for this application
            state.pm_evaluation_pending = False
            state.current_pm_check_cycle = 0
            state.last_config_application_time = None
            state.outcomes = [] # Clear outcomes from previous config's observation
            logger.info(f"Refined parameters for '{state.current_strategy_type}' approved for {intent_id}.")
        else:
            logger.info(f"User rejected refined parameters for {intent_id}.")
            state.fulfilmentStatus = "NOT_FULFILLED" # User terminated this refinement path for the intent
            state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"notFulfilledState": "TERMINATED", "reason": f"User rejected refined parameters for '{state.current_strategy_type}'"}}]}
            state.pm_evaluation_pending = False
        return state
    
    logger.error(f"StrategyAgent for {intent_id}: Invalid state (e.g. current_strategy_type not None or string).")
    state.fulfilmentStatus = "NOT_FULFILLED"
    return state


def data_agent_node(state: IntentState) -> IntentState:
    logger.info(f"Executing {inspect.currentframe().f_code.co_name}")
    cell_id_to_query = None
    object_targets = state.intent.get("intent", {}).get("intentExpectation", {}).get("expectationObject", {}).get("ObjectTarget", [])
    if object_targets:
        for target_cell_id in object_targets:
            if target_cell_id in CELL_CAPABILITIES: cell_id_to_query = target_cell_id; break
    if not cell_id_to_query:
        logger.error(f"No valid cell ID in ObjectTarget {object_targets} found in CELL_CAPABILITIES. Cannot fetch PM.")
        state.pm_data = {"error": "No matching/valid cell ID for PM query."}
        return state

    logger.info(f"Fetching PM data for cell: {cell_id_to_query}")
    try:
        raw_pm_data_str = get_measurement_data(cell_id_to_query, cell_id_to_query)
        
        pm_metrics = json.loads(raw_pm_data_str)
        pm_metrics["queried_cell_id"] = cell_id_to_query
        pm_metrics["fetch_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        state.pm_data = pm_metrics

        # obtain calculated metrics
        calculated_ee_value = calculate_energy_efficiency(state.pm_data)
        logger.info(f"Debug: pm_data: {state.pm_data}")
        state.calculated_ee_metrics = {"RANEnergyEfficiency": calculated_ee_value if calculated_ee_value is not None else "N/A_CALCULATION_FAILED"}
        logger.info(f"Debug: Calculated EE value: {calculated_ee_value}")
        logger.info(f"Debug: AvgPower: {state.pm_data.get('PEE.AvgPower', 'N/A')}")
        logger.info(f"Debug: TotPdcpPduVolumeUl: {state.pm_data.get('QosFlow.TotPdcpPduVolumeUl', 'N/A')}")
        logger.info(f"Debug: TotPdcpPduVolumeDl: {state.pm_data.get('QosFlow.TotPdcpPduVolumeDl', 'N/A')}")

        logger.info(f"Fetched PM for {cell_id_to_query}: {list(pm_metrics.keys())}") # Log keys to verify content
    except Exception as e:
        logger.error(f"Error fetching/parsing PM for {cell_id_to_query}: {e}")
        state.pm_data = {"error": f"Failed to fetch/parse PM data: {str(e)}"}
    return state

def orchestrator_agent_node(state: IntentState) -> IntentState:
    logger.info(f"Executing orchestrator_agent_node for strategy type '{state.current_strategy_type}'")
    intent_id = state.intent.get("intent", {}).get("id", "UNKNOWN_INTENT_ID")

    if not state.strategies or not state.patches or state.current_config_parameters is None:
        logger.error(f"Orchestrator for {intent_id}: Missing strategies, patches, or current_config_parameters.")
        state.fulfilmentStatus = "NOT_FULFILLED"
        state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"reason": "Orchestration error: Missing data for application."}}]}
        state.pm_evaluation_pending = False # Stop if this critical error occurs
        return state

    logger.info(f"Orchestrator for {intent_id}: Applying config with parameters '{state.current_config_parameters}' "
                f"derived from strategy '{state.current_strategy_type}'. Patches: {json.dumps(state.patches)}")
    config_application_success_overall = True
    try:
        # --- Your NETCONF Application Logic using state.patches ---
        for cell_id_str, config_types_dict in state.patches.items():
            if cell_id_str not in state.intent["intent"]["intentExpectation"]["expectationObject"]["ObjectTarget"]:
                logger.warning(f"Skipping patch for cell {cell_id_str} (not in ObjectTarget).")
                continue
            for config_type, parameters_in_patch in config_types_dict.items():
                if 'configuredMaxTxPower' in parameters_in_patch:
                    max_tx_power = parameters_in_patch['configuredMaxTxPower']
                    netconf_client = NETCONFCLIENT(cell_id_str, cell_id_str)
                    netconf_client.configure_tx_power(str(max_tx_power))
                else:
                    logger.info(f"no target parameters available from: {config_types_dict}")
                 # Ensure parameters_in_patch match state.current_config_parameters for consistency if only one set is applied
                logger.info(f"NETCONF: Apply {parameters_in_patch} to cell {cell_id_str} (type: {config_type})")
        # --- End of NETCONF Logic ---
        logger.info(f"NETCONF: Applied config for {intent_id} (parameters: {state.current_config_parameters})") # MOCK
    except Exception as e:
        logger.error(f"NETCONF configuration failed for intent {intent_id}: {e}")
        config_application_success_overall = False
        state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"notFulfilledState": "FULFILMENTFAILED", "reason": f"NETCONF apply error: {str(e)}" }}]}

    if config_application_success_overall:
        state.config_applied_successfully = True
        state.last_config_application_time = datetime.now(timezone.utc)
        state.pm_evaluation_pending = True  # Signal that monitoring phase begins
        state.current_pm_check_cycle = 0    # Reset for this newly applied config
        state.outcomes = []                 # Clear any old outcomes
        state.pm_data = {}                  # Clear PM data to trigger fresh fetch

        logger.info(f"Intent {intent_id}: Config with params '{state.current_config_parameters}' applied. "
                    f"Waiting {MIN_WAIT_FIRST_PM_CHECK_SECONDS}s for first PM check.")
        time.sleep(MIN_WAIT_FIRST_PM_CHECK_SECONDS)

    else:
        state.config_applied_successfully = False
        state.pm_evaluation_pending = False # No point in monitoring if config failed
        state.fulfilmentStatus = "NOT_FULFILLED" # Set overall status
    return state

# def pm_evaluation_agent_node(state: IntentState) -> IntentState:
#     logger.info(f"Executing {inspect.currentframe().f_code.co_name}")
#     intent_id = state.intent.get("intent", {}).get("id", "UNKNOWN_INTENT_ID")
#     evaluated_outcomes = []

#     if not state.pm_data or state.pm_data.get("error"):
#         logger.error(f"PM Evaluator for {intent_id}: PM data error: {state.pm_data.get('error', 'Missing')}")
#         state.outcomes = [{"error": "PM data error for evaluation", "targetName": "PM_Evaluation", "fulfilmentStatus": "NOT_FULFILLED"}]
#         state.calculated_ee_metrics = None
#         return state

#     intent_targets = state.intent.get("intent", {}).get("intentExpectation", {}).get("expectationTargets", [])
#     enriched_pm_data_for_llm = {**state.pm_data} # Start with a copy

#     # Calculate and add EE to the enriched PM data if it's a target or for LLM context
#     calculated_ee_value = calculate_energy_efficiency(state.pm_data)

#     # debugging output
#     logger.info(f"Intent {intent_id}: Calculated EE value: {calculated_ee_value}")
#     logger.info(f"Intent {intent_id}: AvgPower: {state.pm_data.get('PEE.AvgPower', 'N/A')}")
#     logger.info(f"Intent {intent_id}: TotPdcpPduVolumeUl: {state.pm_data.get('QosFlow.TotPdcpPduVolumeUl', 'N/A')}")
#     logger.info(f"Intent {intent_id}: TotPdcpPduVolumeDl: {state.pm_data.get('QosFlow.TotPdcpPduVolumeDl', 'N/A')}")

#     state.calculated_ee_metrics = {"RANEnergyEfficiencyLive": calculated_ee_value if calculated_ee_value is not None else "N/A_CALCULATION_FAILED"}
#     if calculated_ee_value is not None:
#         enriched_pm_data_for_llm["calculatedRANEnergyEfficiency"] = calculated_ee_value
#         logger.info(f"Intent {intent_id}: Calculated EE = {calculated_ee_value}. Added to PM snapshot for LLM.")
#     else:
#         logger.warning(f"Intent {intent_id}: Failed to calculate EE for PM evaluation.")

#     # Explicitly evaluate RANEnergyEfficiency if it's a target
#     for target in intent_targets:
#         target_name = target.get("targetName")
#         if target_name == "RANEnergyEfficiency": # Assuming this is the name used in intents for EE
#             target_value_str = target.get("targetValue")
#             target_condition = target.get("targetCondition")
#             achieved_ee = calculated_ee_value
#             is_ee_fulfilled = False
#             ee_eval_details = "EE target evaluation. "

#             if achieved_ee is not None:
#                 ee_eval_details += f"Calculated EE = {achieved_ee:.4f}. "
#                 try:
#                     target_ee_float = float(target_value_str)
#                     if target_condition == ConditionEnum.IS_GREATER_THAN: is_ee_fulfilled = achieved_ee > target_ee_float
#                     elif target_condition == ConditionEnum.IS_EQUAL_TO_OR_GREATER_THAN: is_ee_fulfilled = achieved_ee >= target_ee_float
#                     # Add other relevant conditions for EE
#                     else: ee_eval_details += f"Unsupported condition '{target_condition}' for EE. "
#                     ee_eval_details += f"Met target ({target_condition} {target_ee_float})? {is_ee_fulfilled}."
#                 except ValueError:
#                     ee_eval_details += f"Target value '{target_value_str}' not a float. "
#             else:
#                 ee_eval_details += "Could not calculate EE. "
            
#             evaluated_outcomes.append({
#                 "targetName": target_name,
#                 "targetAchievedValue": round(achieved_ee, 4) if isinstance(achieved_ee, float) else achieved_ee,
#                 "targetValue": target_value_str, "targetCondition": target_condition,
#                 "fulfilmentStatus": "FULFILLED" if is_ee_fulfilled else "NOT_FULFILLED",
#                 "evaluation_details": ee_eval_details
#             })
#             break # Assuming only one RANEnergyEfficiency target

#     # LLM for other targets or overall summary
#     remaining_targets_for_llm = [t for t in intent_targets if t.get("targetName") != "RANEnergyEfficiency"]
#     if remaining_targets_for_llm:
#         user_prompt = f"""
# Analyze PM data (enriched with `calculatedRANEnergyEfficiency` if available) for the *remaining* intent targets.
# Intent ID: {intent_id}
# Remaining Intent Targets: {json.dumps(remaining_targets_for_llm)}
# Current PM Data Snapshot (may include calculated EE for context): {json.dumps(enriched_pm_data_for_llm, indent=2)}
# Instructions: For each target in "Remaining Intent Targets", provide evaluation. Prioritize `calculatedRANEnergyEfficiency` if relevant.
# Output JSON with "outcomes_llm": [{{targetName, targetAchievedValue, fulfilmentStatus, evaluation_details,...}}]
# """
#         messages = [{"role": "system", "content": "5G Analyst for PM data. Output JSON."}, {"role": "user", "content": user_prompt}]
#         response = call_llm(messages)
#         if response.get("error") or "outcomes_llm" not in response:
#             logger.error(f"PM Evaluator LLM error for remaining targets in {intent_id}: {response.get('error', 'No outcomes_llm')}")
#             for rem_target in remaining_targets_for_llm:
#                  evaluated_outcomes.append({"error": "LLM failed for this target", "targetName": rem_target.get("targetName"), "fulfilmentStatus": "NOT_FULFILLED"})
#         else:
#             evaluated_outcomes.extend(response["outcomes_llm"])
    
#     if not evaluated_outcomes and intent_targets:
#          logger.warning(f"PM Evaluator for {intent_id}: No targets evaluated.")
#          evaluated_outcomes.append({"error": "No targets processed.", "targetName": "PM_Evaluation", "fulfilmentStatus": "NOT_FULFILLED"})

#     state.outcomes = evaluated_outcomes
#     logger.info(f"PM Evaluation for {intent_id} complete. Final Outcomes: {json.dumps(state.outcomes, indent=2)}")
#     return state


def route_main_agent(state: IntentState) -> str:
    logger.info(f"Router: Status='{state.fulfilmentStatus}', "
                f"CurrentStrategyType='{state.current_strategy_type}', "
                f"TotalRefinementAttempts={state.total_refinement_attempts_for_intent}, "
                f"PMCycleForConfig={state.current_pm_check_cycle}, PMEvalPending={state.pm_evaluation_pending}")

    intent_id = state.intent.get("intent", {}).get("id", "UNKNOWN_INTENT_ID")

    # --- 1. Deadline Check (Highest Priority) ---
    if state.intent_processing_deadline and datetime.now(timezone.utc) > state.intent_processing_deadline:
        logger.warning(f"Router for {intent_id}: Intent DEADLINE EXCEEDED. Marking as NOT_FULFILLED.")
        state.fulfilmentStatus = "NOT_FULFILLED"
        state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"notFulfilledState": "TERMINATED", "reason": "Intent processing deadline exceeded."}}]}
        state.pm_evaluation_pending = False # Stop any active monitoring
        # store_attempt will be called by the NOT_FULFILLED final block below

    # --- 2. Final FULFILLED state for the intent ---
    if state.fulfilmentStatus == "FULFILLED":
        logger.info(f"Router for {intent_id}: Intent FULFILLED. Storing and ENDING.")
        store_attempt(state, intent_id)
        return END

    # --- 3. Handling definitive NOT_FULFILLED state (when not actively monitoring a config) ---
    # This is hit if:
    #   a. Deadline exceeded (set above).
    #   b. Max total refinement attempts for the intent reached (set by strategy_agent or checked here).
    #   c. User rejected an initial or refined strategy (strategy_agent sets NOT_FULFILLED).
    #   d. LLM gave up on refining the chosen strategy type (strategy_agent sets NOT_FULFILLED).
    #   e. Orchestrator failed NETCONF (orchestrator_agent sets NOT_FULFILLED).
    #   f. Strategy_agent failed to produce any strategy/refinement.
    if state.fulfilmentStatus == "NOT_FULFILLED" and not state.pm_evaluation_pending:
        logger.info(f"Router for {intent_id}: Intent definitively NOT_FULFILLED (and not in active PM monitoring). Storing and ENDING.")
        store_attempt(state, intent_id)
        return END

    # --- 4. Active Closed-Loop Monitoring for an APPLIED configuration ---
    if state.pm_evaluation_pending:
        if not state.pm_data or state.pm_data.get("error"):
            logger.info(f"Router for {intent_id}: Monitoring. PM data needed. To data_agent.")
            return "data_agent"
        elif state.pm_data and not state.outcomes: # Fresh PM data, needs evaluation
            logger.info(f"Router for {intent_id}: Monitoring. Fresh PM. To pm_evaluation_agent.")
            return "pm_evaluation_agent_node"
        elif state.outcomes: # Outcomes from pm_evaluation_agent are available
            state.current_pm_check_cycle += 1
            logger.info(f"Router for {intent_id}: Monitoring. Outcomes from PM cycle {state.current_pm_check_cycle} "
                        f"for config params '{state.current_config_parameters}' of strategy type '{state.current_strategy_type}'.")

            all_targets_met = not any(o.get("fulfilmentStatus") != "FULFILLED" for o in state.outcomes)
            
            current_report_results = [] # Update report for this observation cycle
            for out_eval in state.outcomes:
                orig_target = next((t for t in state.intent["intent"]["intentExpectation"]["expectationTargets"] if t["targetName"] == out_eval.get("targetName")), None)
                current_report_results.append({
                    "targetName": out_eval.get("targetName"), "targetStatus": out_eval.get("fulfilmentStatus", "NOT_FULFILLED"),
                    "targetAchievedValue": out_eval.get("targetAchievedValue"), "targetValue": orig_target.get("targetValue") if orig_target else "N/A",
                    "appliedConfigParameters": state.current_config_parameters,
                    "failureAnalysis": {"reason": out_eval.get("evaluation_details")} if out_eval.get("fulfilmentStatus") != "FULFILLED" else {}
                })
            state.report = {"intentId": intent_id, "expectationFulfilmentResult": current_report_results,
                            "strategyTypeAttempted": state.current_strategy_type,
                            "totalRefinementAttemptForIntent": state.total_refinement_attempts_for_intent,
                            "pmCheckCycleForThisConfig": state.current_pm_check_cycle}

            if all_targets_met:
                logger.info(f"Router for {intent_id}: All targets FULFILLED with params '{state.current_config_parameters}' for strategy '{state.current_strategy_type}'.")
                state.fulfilmentStatus = "FULFILLED"
                state.pm_evaluation_pending = False
                return "strategy_agent" # Go to hub, will route to END (handled by section 2)
            else: # Not all targets met for this config's PM check
                elapsed_monitoring_time = (datetime.now(timezone.utc) - state.last_config_application_time).total_seconds()
                end_of_observation_for_this_config = (state.current_pm_check_cycle >= MAX_PM_CHECK_CYCLES_PER_STRATEGY or \
                                                     elapsed_monitoring_time >= OBSERVATION_PERIOD_SECONDS)

                if end_of_observation_for_this_config:
                    logger.info(f"Router for {intent_id}: Config params '{state.current_config_parameters}' for strategy '{state.current_strategy_type}' "
                                f"did NOT meet targets after full observation (PM Cycle {state.current_pm_check_cycle}, Elapsed {elapsed_monitoring_time:.0f}s).")
                    state.pm_evaluation_pending = False # Observation for *this config* is over.
                    # The intent is not yet NOT_FULFILLED. We need to try refining.
                    # state.fulfilmentStatus remains "NONE" or previous ACK.
                    # The state.pm_data and state.outcomes from this failed observation are preserved
                    # for the strategy_agent's refinement prompt.
                    logger.info(f"Router for {intent_id}: Will attempt to REFINE strategy type '{state.current_strategy_type}'.")
                    state.history_summary = {} # Get fresh history (may include this recent failure if store_attempt was intermediate)
                    state.outcomes = [] # Clear outcomes before next PM eval (for the *refined* config)
                    # state.pm_data is KEPT as it's input for strategy_agent's refinement
                    return "history_agent" # -> then strategy_agent (Phase 2 - Refine)
                else: # Continue monitoring *this specific applied configuration*
                    logger.info(f"Router for {intent_id}: Targets not met. Continuing monitoring current config "
                                f"(Cycle {state.current_pm_check_cycle + 1} next, Elapsed {elapsed_monitoring_time:.0f}s). Waiting {PM_CHECK_INTERVAL_SECONDS}s.")
                    time.sleep(PM_CHECK_INTERVAL_SECONDS)
                    state.pm_data = {} # Clear for fresh PM fetch for *this same config's next check*
                    state.outcomes = [] # Clear outcomes before next PM eval for *this same config*
                    return "data_agent"

    # --- 5. Path for Orchestration or Getting Initial/Refined Strategy ---
    if state.fulfilmentStatus == "ACKNOWLEDGED": # Strategy (initial or refined) approved by user
        logger.info(f"Router for {intent_id}: Status ACKNOWLEDGED. Routing to orchestrator_agent.")
        # state.current_config_parameters should have been set by strategy_agent with the params to be applied
        return "orchestrator_agent"

    # --- 6. Default path to get context and then generate/refine strategy ---
    # This is hit:
    #   - At the very start of intent processing.
    #   - After a config's observation failed, and we need to refine (flow is: route -> data -> history -> strategy).
    #   - If any other path leads here needing a strategy decision.

    # Ensure PM data is available first if a strategy decision is upcoming
    if (state.current_strategy_type is None or \
        (state.current_strategy_type and not state.pm_evaluation_pending and state.fulfilmentStatus != "ACKNOWLEDGED")) and \
       (not state.pm_data or state.pm_data.get("error")):
        # Condition breakdown:
        # (state.current_strategy_type is None) -> We need to propose an initial strategy.
        # OR
        # (state.current_strategy_type and not state.pm_evaluation_pending and state.fulfilmentStatus != "ACKNOWLEDGED")
        #   -> We have a strategy type, we are NOT currently observing its application (pm_evaluation_pending=False),
        #      and it's not just been ACKNOWLEDGED (waiting for orchestrator). This implies a previous
        #      observation cycle for this strategy type might have completed (and possibly failed),
        #      and we are now heading towards refining it.
        # AND in either of these cases, if PM data is missing/invalid...
        logger.info(f"Router for {intent_id}: PM data needed before strategy decision. To data_agent.")
        return "data_agent"

    # Ensure history summary is available if PM data is present and strategy decision is upcoming
    if (state.current_strategy_type is None or \
        (state.current_strategy_type and not state.pm_evaluation_pending and state.fulfilmentStatus != "ACKNOWLEDGED")) and \
       (state.pm_data and not state.pm_data.get("error")) and \
       (not state.history_summary or not state.history_summary.get("result")): # Check if history is valid
        logger.info(f"Router for {intent_id}: History summary needed before strategy decision. To history_agent.")
        return "history_agent"
    
    # If context (PM, History) is ready, and not in other specific flows (like ACKNOWLEDGED or PM_EVAL_PENDING),
    # then it's time to call strategy_agent.
    # strategy_agent will determine if it's Phase 1 (initial proposal) or Phase 2 (refinement)
    # based on state.current_strategy_type.
    # This path is reached if:
    #   1. Initial call, PM and History fetched.
    #   2. A refinement is needed (observation of previous params for current_strategy_type is complete,
    #      and it wasn't FULFILLED), PM and History (updated with last outcome) are ready.
    if (state.pm_data and not state.pm_data.get("error") and \
        state.history_summary and state.history_summary.get("result")) and \
        (state.fulfilmentStatus not in ["FULFILLED", "NOT_FULFILLED", "ACKNOWLEDGED"] or \
         (state.fulfilmentStatus == "NONE" and state.current_strategy_type and not state.pm_evaluation_pending) ): 
         # The last condition (NONE + current_strategy_type + not pm_eval_pending) specifically targets 
         # the case where a refinement cycle just finished its observation unsuccessfully.
        logger.info(f"Router for {intent_id}: Context (PM, History) ready. Routing to strategy_agent.")
        return "strategy_agent"

    # Fallback if other specific conditions in the router (ACKNOWLEDGED, PM_EVAL_PENDING) are met
    # This part should be reached if, for example, status is ACKNOWLEDGED, or PM_EVALUATION_PENDING is true,
    # and those earlier blocks in the router will handle the routing.
    # If it falls through all specific handlers, it means something is unexpected.

    logger.warning(f"Router for {intent_id}: Unhandled state or waiting for specific phase. "
                   f"Status='{state.fulfilmentStatus}', PMDataValid={state.pm_data and not state.pm_data.get('error')}, "
                   f"HistoryValid={state.history_summary and state.history_summary.get('result')}, "
                   f"PMEvalPending={state.pm_evaluation_pending}. Re-routing to strategy_agent as a convergence point.")
    # This fallback is a bit of a catch-all. Ideally, the conditions above should perfectly guide.
    # If it reaches here, `strategy_agent` might be called without perfect prerequisites,
    # and its internal checks would then matter more. The goal is to avoid this.
    # A safer fallback if prerequisites are NOT met and no other path is clear might be to error or end.
    # However, since strategy_agent is the entry point of the graph after each node,
    # sending it back to strategy_agent allows route_main_agent to re-evaluate the state cleanly.
    return "strategy_agent"

def build_graph():
    graph = StateGraph(IntentState)
    # nodes = ["strategy_agent", "data_agent", "history_agent", "orchestrator_agent", "pm_evaluation_agent_node"]
    # for node_name in nodes: graph.add_node(node_name, globals()[node_name])
    graph.add_node("strategy_agent", strategy_agent)
    graph.add_node("data_agent", data_agent_node)
    graph.add_node("history_agent", history_agent)
    graph.add_node("orchestrator_agent", orchestrator_agent_node)
    # graph.add_node("pm_evaluation_agent_node", pm_evaluation_agent_node)
    graph.set_entry_point("strategy_agent")
    graph.add_conditional_edges("strategy_agent", route_main_agent, {
        "data_agent": "data_agent", 
        "history_agent": "history_agent",
        "orchestrator_agent": "orchestrator_agent", 
        "strategy_agent": "strategy_agent", END: END,
    })
    fixed_edges = [("data_agent", "strategy_agent"), 
                   ("history_agent", "strategy_agent"),
                   ("orchestrator_agent", "strategy_agent")]
    for src, dest in fixed_edges: graph.add_edge(src, dest)
    return graph.compile()

if __name__ == "__main__":
    graph = build_graph() # Assuming build_graph uses the modified agents
    intent_input = {
        "intent": {
            "id": "INTENT_EE_Test_335",
            "intentExpectation": {
                "expectationObject": {"objectInstance": "SubNetwork_1", "ObjectTarget": ["2"]},
                "expectationTargets": [{"targetName": "RANEnergyEfficiency", "targetCondition": "IS_GREATER_THAN", "targetValue": "10", "targetUnit": "percentage"}] # Example target value
            },
            "startTime": datetime.now(timezone.utc).isoformat(),
            "endTime": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
        }
    }
    initial_state = IntentState(intent=intent_input) # Corrected structure
    # initial_state = IntentState(**initial_state_dict) # Use the dict directly if it matches IntentState fields

    # Set deadline for the intent processing
    processing_start_time = datetime.now(timezone.utc)
    initial_state.intent_processing_deadline = processing_start_time + timedelta(seconds=600)
    
    # Initialize fields for the simplified single-strategy refinement loop
    initial_state.current_strategy_type = None # strategy_agent will set this on first call
    initial_state.current_config_parameters = None
    initial_state.total_refinement_attempts_for_intent = 0 # Will be incremented by strategy_agent

    logger.info(f"Starting graph for intent: {initial_state.intent.get('id', 'N/A')}, "
                f"Deadline: {initial_state.intent_processing_deadline.isoformat() if initial_state.intent_processing_deadline else 'N/A'}")
    
    # graph = build_graph()
    # intent_input = {
    #     "intent": {
    #         "id": f"INTENT_EE_Test_{str(uuid4())[:4]}",
    #         "intentExpectation": {
    #             "expectationObject": {"objectInstance": "SubNetwork_1", "ObjectTarget": ["2"]},
    #             "expectationTargets": [{"targetName": "RANEnergyEfficiency", "targetCondition": "IS_GREATER_THAN", "targetValue": "0.00005", "targetUnit": "BytesPerJoule"}] # Example target value
    #         },
    #         "startTime": datetime.now(timezone.utc).isoformat(),
    #         "endTime": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    #     }
    # }
    # initial_state = IntentState(intent=intent_input)
    # logger.info(f"Starting graph with intent: {json.dumps(intent_input, indent=2)}")
    
    # # --- To run the actual graph (requires OPENAI and NETCONF setup) ---
    final_state = graph.invoke(initial_state)
    # logger.info(f"Graph execution complete. Final state: {final_state.model_dump_json(indent=2)}")
    # logger.info(f"All STRATEGY_ATTEMPTS: {json.dumps(STRATEGY_ATTEMPTS, indent=2)}")
    # # --- End of actual graph run block ---
