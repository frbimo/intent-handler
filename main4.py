from typing import Dict, List, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_core.runnables.config import RunnableConfig

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
MAX_TOTAL_REFINEMENT_ATTEMPTS_FOR_INTENT = 30

# COLOR CODE For Print
RESET = '\033[0m'
CYAN= '\033[36m' 
RED = '\033[31m'
GREEN = '\033[32m'
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

# class PMData(BaseModel):
#     """
#     Represents the PM data structure expected from the PM Data API.
#     This is a simplified version; adjust fields as per your actual PM data structure.
#     """
#     DRBUEThpDl_Mbps: Optional[float] = None
#     DRBUEThpUl_Mbps: Optional[float] = None
#     TotPdcpPduVolumeUl: Optional[float] = None
#     TotPdcpPduVolumeDl: Optional[float] = None
#     PEEAvgPower_: Optional[float] = None # Average Power Consumption in Watts
#     # Add other fields as necessary based on your PM data structure

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
    observation_deadline: Optional[datetime] = None

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
    # pdcp_ul_key = "QosFlow.TotPdcpPduVolumeUl" #Mbits 
    # pdcp_dl_key = "QosFlow.TotPdcpPduVolumeDl" #Mbits 
    power_key = "PEE.AvgPower" # Watt 
    # power_key = "PEE.Energy" # kWh
    ue_thp_ul_key = "DRB.UEThpUl" # bps
    ue_thp_dl_key = "DRB.UEThpDl" # bps

    logger.info(f"{GREEN}Calculating Energy Efficiency from PM data{RESET}")

    # # --- 3GPP uses PDCPU SDU --- #
    # if pdcp_ul_key in pm_data and pdcp_dl_key in pm_data and power_key in pm_data:
    #     try:
    #         total_pdcp_volume_kbit = (float(pm_data[pdcp_ul_key]) + float(pm_data[pdcp_dl_key]))
    #         # total_power_kWh = float(pm_data[power_key]) / 1000 # Convert Watts to kW (1 kW = 1000 W)
    #         total_power_kWh = float(pm_data[power_key]) # Convert Watts to kW (1 kW = 1000 W)
    #         logger.info(f"{GREEN}Total PDCP Volume (UL: {float(pm_data[pdcp_ul_key])} + DL: {float(pm_data[pdcp_dl_key])}): {total_pdcp_volume_kbit} kbit, Total Power Consumption: {total_power_kWh} kWh{RESET}")
    #         if total_power_kWh > 0:
    #             ee = total_pdcp_volume_kbit / total_power_kWh
    #             logger.info(f"Calculated EE: {total_pdcp_volume_kbit} / {total_power_kWh} = {ee}")
    #             return ee
    #         else:
    #             logger.warning("Cannot calculate EE: Total power consumption is zero or not positive.")
    #             return 0.0 if total_pdcp_volume_kbit == 0 else None # Or a marker for very high EE
    #     except ValueError:
    #         logger.error(f"Could not convert PM data for EE calculation to float.")
    #         return None
    # else:
    #     missing = [k for k in [pdcp_ul_key, pdcp_dl_key, power_key] if k not in pm_data]
    #     logger.warning(f"Cannot calculate EE: Missing PM data keys: {', '.join(missing)}")
    #     return None

    # --- IEEE uses Throughput --- #
    if ue_thp_ul_key in pm_data and ue_thp_dl_key in pm_data and power_key in pm_data:
        try:
            total_throughput_bits = (float(pm_data[ue_thp_ul_key]) + float(pm_data[ue_thp_dl_key]))
            # total_power_kWh = float(pm_data[power_key]) / 1000 # Convert Watts to kW (1 kW = 1000 W)
            total_power_kWh = float(pm_data[power_key]) # Convert Watts to kW (1 kW = 1000 W)
            logger.info(f"{GREEN}Total Throughput Volume (UL: {float(pm_data[ue_thp_ul_key])} + DL: {float(pm_data[ue_thp_dl_key])}): {total_throughput_bits} kbit, Total Power Consumption: {total_power_kWh} kWh{RESET}")
            if total_power_kWh > 0:
                ee = total_throughput_bits / total_power_kWh
                logger.info(f"Calculated EE: {total_throughput_bits} / {total_power_kWh} = {ee}")
                return ee
            else:
                logger.warning("Cannot calculate EE: Total power consumption is zero or not positive.")
                return 0.0 if total_throughput_bits == 0 else None # Or a marker for very high EE
        except ValueError:
            logger.error(f"Could not convert PM data for EE calculation to float.")
            return None
    else:
        missing = [k for k in [ue_thp_ul_key, ue_thp_dl_key, power_key] if k not in pm_data]
        logger.warning(f"Cannot calculate EE: Missing PM data keys: {', '.join(missing)}")
        return None

# --- Agent Nodes ---
def history_agent(state: IntentState) -> IntentState:
    logger.info(f"{CYAN}Executing {inspect.currentframe().f_code.co_name}{RESET}")

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

def filter_and_rename_metrics_for_json(pm_metrics: dict) -> dict:
    """
    Renames specified metric keys and converts numeric string values to actual numbers.
    Only includes the metrics defined in the 'keys_to_rename_map', effectively
    removing all other keys from the original dictionary.

    Args:
        pm_metrics (dict): The dictionary containing raw PM data.

    Returns:
        dict: A new dictionary containing only the specified keys (renamed)
              and their numeric values (if converted), ready for JSON dumping.
    """
    # This dictionary defines the exact keys you want to keep and their new names
    keys_to_rename_map = {
        "DRB.UEThpDl": "DRB.UEThpDl_bps",
        "DRB.UEThpUl": "DRB.UEThpUl_bps",
        # "QosFlow.TotPdcpPduVolumeUl": "QosFlow.TotPdcpPduVolumeUl_Mbits",
        # "QosFlow.TotPdcpPduVolumeDl": "QosFlow.TotPdcpPduVolumeDl_Mbits",
        # "PEE.Energy": "PEE.Energy_kWh"
        "PEE.AvgPower": "PEE.AvgPower_Watts", # Assuming this is the average power consumption in Watts

    }

    filtered_and_renamed_metrics = {}

    # Iterate through the 'keys_to_rename_map' to ensure we only process desired keys
    for original_key_in_map, new_key_name in keys_to_rename_map.items():
        if original_key_in_map in pm_metrics:
            # If the original key exists in the raw pm_metrics, process its value
            value = pm_metrics[original_key_in_map]
            
            processed_value = value
            # Attempt to convert string values to int or float for better JSON representation
            if isinstance(value, str):
                try:
                    processed_value = int(value)
                except ValueError:
                    try:
                        processed_value = float(value)
                    except ValueError:
                        # Value is a string but not a number, keep as string
                        pass
            
            # Add the item to the new dictionary with the new key name
            filtered_and_renamed_metrics[new_key_name] = processed_value
        else:
            # Log a warning if a key you expected to rename wasn't found in the input data
            logger.warning(f"Metric '{original_key_in_map}' (to be renamed to '{new_key_name}') not found in the input PM data. Skipping.")
            
    return filtered_and_renamed_metrics

def strategy_agent(state: IntentState) -> IntentState:
    logger.info(f"{CYAN}Executing {inspect.currentframe().f_code.co_name}{RESET}")
    logger.info(f"CurrentStrategyType='{state.current_strategy_type}', "
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
            netconf_client = NETCONFCLIENT(1, cell_id_str)
            state.current_config_parameters={"configuredMaxTxPower": netconf_client.get_current_tx_power()} 

        logger.info(f"Obtained runtime Energy Efficiency Metrics {intent_id}: {json.dumps(state.calculated_ee_metrics)}")
        user_prompt_initial = f"""
Propose an INITIAL strategy type and configuration parameters for intent {intent_id}.
Intent Targets: {json.dumps(targets)}
Object Target: {object_target}
Current Energy Efficiency Metrics in bit_per_joule: {json.dumps(state.calculated_ee_metrics)}
Obtained runtime configuration parameters: {json.dumps(state.current_config_parameters)}
Historical Insights: {json.dumps(history_insights, indent=2)}
Current PM Data: {json.dumps(filter_and_rename_metrics_for_json(state.pm_data))}
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

        # user_input = input(f"LLM proposed initial strategy type: '{chosen_strategy_type}' with params {initial_parameters}. Use this type for all refinements? ('yes' to confirm): ")
        user_input = "yes" # For testing, assume user always approves the initial strategy type
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
            # --- IMPORTANT CHANGE FOR INITIAL REJECTION ---
            logger.info(f"User rejected initial strategy proposal for {intent_id}. Re-evaluating.")
            # DO NOT set fulfilmentStatus to NOT_FULFILLED here.
            # Instead, clear the proposed strategy and ensure pm_evaluation_pending is False
            # The router will then re-check the RAN state.
            state.current_strategy_type = None # Clear this so it can be proposed again (or fail if attempts exhausted)
            state.current_config_parameters = {} # Clear any temp proposed params
            state.pm_evaluation_pending = False # Not pending application of a strategy
            # Do not decrement total_refinement_attempts_for_intent, as this was a valid attempt.
            # Router will take over and lead to data_agent or eventually terminate if MAX_TOTAL_REFINEMENT_ATTEMPTS_FOR_INTENT is met.
        return state

    # --- Phase 2: REFINE Parameters for the EXISTING Strategy Type ---
    elif state.current_strategy_type:
        logger.info(f"StrategyAgent for {intent_id}: Phase 2 - Refining strategy type '{state.current_strategy_type}'. "
                    f"Next total refinement attempt for intent: {state.total_refinement_attempts_for_intent + 1}.")

        if state.total_refinement_attempts_for_intent >= MAX_TOTAL_REFINEMENT_ATTEMPTS_FOR_INTENT:
            logger.warning(f"Max total refinement attempts ({MAX_TOTAL_REFINEMENT_ATTEMPTS_FOR_INTENT}) reached for intent {intent_id} while trying to refine '{state.current_strategy_type}'. Giving up.")
            state.fulfilmentStatus = "NOT_FULFILLED" # Max attempts reached, definitive termination
            state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"reason": f"Max total refinement attempts reached for strategy type '{state.current_strategy_type}'."}}]}
            return state

        state.total_refinement_attempts_for_intent += 1 # Count this refinement attempt


        # state.current_config_parameters holds the *last applied* parameters
        # state.pm_data and state.outcomes hold the results of *that last application*
        user_prompt_refine = f"""
Targets: {json.dumps(state.intent["intent"]["intentExpectation"]["expectationTargets"])}
Previous Parameters: {json.dumps(state.current_config_parameters)}
PM Data (post-application): {json.dumps(filter_and_rename_metrics_for_json(state.pm_data))}
Outcomes: {json.dumps(state.outcomes)}
Current Energy Efficiency Metrics in bit_per_joule: {json.dumps(state.calculated_ee_metrics)}
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
            # LLM failed to refine, so this attempt is flawed. Mark as NOT_FULFILLED.
            state.fulfilmentStatus = "NOT_FULFILLED"
            state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"reason": f"LLM failed to refine strategy '{state.current_strategy_type}'"}}]}
            state.pm_evaluation_pending = False # Not pending any application
            return state

        explanation = response.get("explanation", "N/A")
        logger.info(f"LLM Refinement Explanation for {intent_id}: {explanation}")

        refined_parameters = response["result"].get("refined_parameters")
        give_up = response["result"].get("give_up_on_this_strategy_type", False)

        if give_up or not refined_parameters:
            logger.info(f"LLM indicated to give up on '{state.current_strategy_type}' or no refinement for {intent_id}. Re-evaluating.")
            # --- IMPORTANT CHANGE FOR LLM GIVE UP ---
            # DO NOT set fulfilmentStatus to NOT_FULFILLED here.
            # Allow the router to re-evaluate the current RAN state.
            state.pm_evaluation_pending = False # Not pending application of a new refined strategy
            # Current_strategy_type remains, but this 'give up' counts as an attempt.
            # The router's max_attempts check will handle final termination.
            # state.outcomes might still be populated from the *last* PM evaluation
            # The router will pick it up and go to data_agent if targets not met.
            # Clear outcomes if you want a fresh PM evaluation cycle.
            state.outcomes = []
            return state # Return state, let router decide

        # user_input = input(f"LLM proposed refined params {refined_parameters} for '{state.current_strategy_type}' (Total Attempt {state.total_refinement_attempts_for_intent}). Apply? ('yes' to confirm): ")
        user_input = "yes" # For testing, assume user always approves the refined parameters
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
            # --- IMPORTANT CHANGE FOR REFINED REJECTION ---
            logger.info(f"User rejected refined parameters for {intent_id}. Re-evaluating.")
            # DO NOT set fulfilmentStatus to NOT_FULFILLED here.
            # Allow the router to re-evaluate the current RAN state.
            state.pm_evaluation_pending = False # Not pending application of a new refined strategy
            # Current_strategy_type remains, as we are still working on this type.
            # The router will pick it up and go to data_agent if targets not met.
            # Clear outcomes if you want a fresh PM evaluation cycle.
            state.outcomes = [] # Clear outcomes from previous config's observation
        return state
    
    logger.error(f"StrategyAgent for {intent_id}: Invalid state (e.g. current_strategy_type not None or string).")
    state.fulfilmentStatus = "NOT_FULFILLED"
    return state


def data_agent_node(state: IntentState) -> IntentState:
    logger.info(f"{CYAN}Executing {inspect.currentframe().f_code.co_name}{RESET}")
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
        raw_pm_data_str = get_measurement_data(cell_id_to_query, 1)
        
        pm_metrics = json.loads(raw_pm_data_str)
        pm_metrics["queried_cell_id"] = cell_id_to_query
        pm_metrics["fetch_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        state.pm_data = pm_metrics

        # obtain calculated metrics
        calculated_ee_value = calculate_energy_efficiency(state.pm_data)
        logger.info(f"Debug: pm_data: {state.pm_data}")
        state.calculated_ee_metrics = {"RANEnergyEfficiency": calculated_ee_value if calculated_ee_value is not None else "N/A_CALCULATION_FAILED"}
        logger.info(f"Debug: Calculated EE value: {calculated_ee_value}")
        logger.info(f"Debug: AvgPower: {state.pm_data.get('PEE.Energy', 'N/A')}")
        logger.info(f"Debug: TotPdcpPduVolumeUl: {state.pm_data.get('QosFlow.TotPdcpPduVolumeUl', 'N/A')}")
        logger.info(f"Debug: TotPdcpPduVolumeDl: {state.pm_data.get('QosFlow.TotPdcpPduVolumeDl', 'N/A')}")

        logger.info(f"Fetched PM for {cell_id_to_query}: {list(pm_metrics.keys())}") # Log keys to verify content
    except Exception as e:
        logger.error(f"Error fetching/parsing PM for {cell_id_to_query}: {e}")
        state.pm_data = {"error": f"Failed to fetch/parse PM data: {str(e)}"}
    return state

def orchestrator_agent_node(state: IntentState) -> IntentState:
    logger.info(f"{CYAN}Executing {inspect.currentframe().f_code.co_name}{RESET}")

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
                    netconf_client = NETCONFCLIENT(1, cell_id_str)
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


def pm_evaluation_agent_node(state: IntentState) -> IntentState:
    logger.info(f"{CYAN}Executing {inspect.currentframe().f_code.co_name}{RESET}")
    targets = state.intent["intent"]["intentExpectation"]["expectationTargets"]
    
    # Initialize outcomes and assume all targets are met until proven otherwise
    state.outcomes = []
    overall_all_targets_met = True

    for target in targets:
        target_name = target["targetName"]
        target_condition = target["targetCondition"]
        target_value_str = target["targetValue"]
        
        current_value = None
        target_status = "NOT_FULFILLED" # Default status for this target
        failure_reason = None

        # --- Extract current value based on target_name ---
        if target_name == "RANEnergyEfficiency":
            current_value_raw = state.calculated_ee_metrics.get(target_name)
            logger.debug(f"PM Evaluation: Current RANEnergyEfficiency raw value: {current_value_raw}")
            
            try:
                current_value = float(current_value_raw) if current_value_raw not in [None, "N/A_CALCULATION_FAILED"] else None
            except ValueError:
                logger.error(f"Invalid RANEnergyEfficiency value for target '{target_name}': '{current_value_raw}'. Cannot evaluate.")
                current_value = None
                failure_reason = "Invalid_Current_Value_Format"
        # Add additional `elif` blocks here for other target_names if they exist in your PM data

        # --- Perform comparison ---
        if current_value is None:
            target_status = "NOT_FULFILLED"
            overall_all_targets_met = False
            if not failure_reason: # Set reason if not already set by value parsing
                failure_reason = "Current_Value_Unavailable_or_Invalid"
        else:
            try:
                target_value = float(target_value_str)
            except ValueError:
                logger.error(f"Invalid targetValue format for target '{target_name}': '{target_value_str}'. Cannot evaluate.")
                target_status = "NOT_FULFILLED"
                overall_all_targets_met = False
                failure_reason = "Invalid_Target_Value_Format"
            else:
                is_met = False
                if target_condition == "IS_GREATER_THAN":
                    is_met = current_value > target_value
                elif target_condition == "IS_LESS_THAN":
                    is_met = current_value < target_value
                elif target_condition == "IS_EQUAL_TO":
                    is_met = current_value == target_value
                # Add more conditions as needed for your intent types

                if is_met:
                    target_status = "FULFILLED"
                else:
                    target_status = "NOT_FULFILLED"
                    overall_all_targets_met = False
                    failure_reason = f"Condition_Not_Met: Current {current_value} vs Target {target_condition} {target_value}"
        
        # Add the outcome for this specific target
        state.outcomes.append({
            "targetName": target_name,
            "currentValue": current_value,
            "targetCondition": target_condition,
            "targetValue": target_value_str,
            "fulfilmentStatus": target_status,
            "failureAnalysis": {"reason": failure_reason} if failure_reason else None
        })
        
        logger.info(f"Target '{target_name}' Status: {target_status}. Current: {current_value}, Target: {target_condition} {target_value_str}")
    
    # Set the overall intent fulfillment status based on all targets
    if overall_all_targets_met:
        state.fulfilmentStatus = "FULFILLED"
        logger.info(f"Overall Intent {state.intent.get('intent', {}).get('id', 'UNKNOWN')} Status: FULFILLED.")
        # Report is set at the end by the router if needed, or by this agent if specific failure.
        # For now, let's keep the report setting in the router as it's the central decision point.
    else:
        # If not all targets are met, the status remains as is (or `None`), and the router decides next.
        # It's better to NOT set NOT_FULFILLED here directly, as it might be an interim step.
        # The router will determine if it needs further refinement or definitive NOT_FULFILLED.
        logger.info(f"Overall Intent {state.intent.get('intent', {}).get('id', 'UNKNOWN')} Status: NOT FULFILLED (targets not met yet).")

    return state

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

def route_main_agent(state: IntentState) -> IntentState:
    """
    This function now acts as the 'router node'. It's responsible for
    updating state based on routing conditions (e.g., setting FULFILLED status)
    and returning the updated state. The actual next node decision (str)
    is handled by _get_next_route_from_state which is used in add_conditional_edges.
    """
    logger.info(f"{RED}Executing {inspect.currentframe().f_code.co_name}{RESET}")
    intent_id = state.intent.get("intent", {}).get("id", "UNKNOWN_INTENT_ID")

    # IMPORTANT: Any state modifications that depend on routing conditions
    # must happen here, before returning the state.

    # Example: If a deadline is exceeded, update state before it's passed to the conditional edge
    if state.observation_deadline and datetime.now(timezone.utc) > state.observation_deadline:
        logger.warning(f"Router Node for {intent_id}: Intent DEADLINE EXCEEDED. Marking as NOT_FULFILLED.")
        state.fulfilmentStatus = "NOT_FULFILLED"
        state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"notFulfilledState": "TERMINATED", "reason": "Intent processing deadline exceeded."}}]}
        state.pm_evaluation_pending = False
        # The _get_next_route_from_state will then see NOT_FULFILLED and return END.
        return state

    # Example: If all targets were just fulfilled in the PM eval, update state.
    # This logic assumes outcomes are fresh from pm_evaluation_agent_node
    if state.pm_evaluation_pending and state.outcomes:
        all_targets_met = not any(o.get("fulfilmentStatus") != "FULFILLED" for o in state.outcomes)
        if all_targets_met and state.fulfilmentStatus != "FULFILLED": # Prevent re-setting if already FULFILLED
            logger.info(f"Router Node for {intent_id}: All targets FULFILLED. Setting state.fulfilmentStatus.")
            state.fulfilmentStatus = "FULFILLED"
            state.pm_evaluation_pending = False # Stop active monitoring
            # Report generation related to fulfillment
            current_report_results = []
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

    # If observation period for current config ended unsuccessfully, reset relevant flags for refinement
    if state.pm_evaluation_pending and state.outcomes: # Only check if outcomes are fresh from PM eval
        all_targets_met_current_cycle = not any(o.get("fulfilmentStatus") != "FULFILLED" for o in state.outcomes)
        if not all_targets_met_current_cycle:
            elapsed_monitoring_time = (datetime.now(timezone.utc) - state.last_config_application_time).total_seconds()
            end_of_observation_for_this_config = (elapsed_monitoring_time >= state.observation_deadline.timestamp())
                                                 
            if end_of_observation_for_this_config:
                logger.info(f"Router Node for {intent_id}: Config params '{state.current_config_parameters}' did NOT meet targets after full observation. Clearing for refinement.")
                state.pm_evaluation_pending = False # Observation for *this config* is over.
                state.history_summary = {} # Clear history to force refresh for refinement
                state.outcomes = [] # Clear outcomes for next PM eval (for refined config)
                # state.pm_data is KEPT as it's input for strategy_agent's refinement

    # Store attempt needs to be called *after* final status is set, but before END.
    # It might be best handled by the main loop/orchestrator outside the graph,
    # or by specific "finalizer" nodes that lead to END.
    # For now, let's keep it here if it's convenient for report generation.
    if state.fulfilmentStatus in ["FULFILLED", "NOT_FULFILLED"] and not state.pm_evaluation_pending:
        logger.info(f"Router Node for {intent_id}: Finalizing and storing attempt.")
        # store_attempt(state, intent_id) # Call your actual store_attempt function here

    # Always return the (potentially updated) state.
    return state

def _get_next_route_from_state(state: IntentState) -> str | Literal[END]:
    logger.info(f"{RED}Executing {inspect.currentframe().f_code.co_name}{RESET}")
    
    intent_id = state.intent.get("intent", {}).get("id", "UNKNOWN_INTENT_ID")

    # --- 1. Deadline Check (Highest Priority for Early Exit) ---
    # If the processing deadline has passed, mark as NOT_FULFILLED and end.
    if state.observation_deadline and datetime.now(timezone.utc) > state.observation_deadline:
        logger.warning(f"Router for {intent_id}: Intent DEADLINE EXCEEDED. Marking as NOT_FULFILLED.")
        state.fulfilmentStatus = "NOT_FULFILLED" # Set status for final report
        return END

    # --- 2. Final FULFILLED State (Immediate Stop) ---
    # If the intent is already marked FULFILLED (by PM evaluation or other means), end the graph.
    if state.fulfilmentStatus == "FULFILLED":
        logger.info(f"Router for {intent_id}: Intent FULFILLED. ENDING.")
        return END

    # --- 3. Overall Attempt Limit Check (Ultimate Failure) ---
    # If we've exhausted all allowed strategy proposal/refinement attempts, mark as NOT_FULFILLED and end.
    if state.total_refinement_attempts_for_intent >= MAX_TOTAL_REFINEMENT_ATTEMPTS_FOR_INTENT:
        logger.warning(f"Router for {intent_id}: Max total refinement attempts ({MAX_TOTAL_REFINEMENT_ATTEMPTS_FOR_INTENT}) reached. Marking NOT_FULFILLED.")
        state.fulfilmentStatus = "NOT_FULFILLED"
        # Only set report if not already done by strategy_agent for a specific LLM failure reason
        if not state.report:
            state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"reason": "Max total refinement attempts reached for intent."}}]}
        return END

    # --- 4. Path for Orchestration (Apply the Acknowledged Strategy) ---
    # If a strategy has been approved by the user (ACKNOWLEDGED status), proceed to apply it.
    if state.fulfilmentStatus == "ACKNOWLEDGED" and state.config_applied_successfully is False:
        logger.info(f"Router for {intent_id}: Status ACKNOWLEDGED. Routing to orchestrator_agent.")
        return "orchestrator_agent"

    # --- 5. Core Decision Logic: Prioritize PM Data & Evaluation ---
    # This section handles all decisions based on current PM data and its evaluation.
    # This covers initial intent, post-orchestration, post-user-rejection, and post-LLM-giveup scenarios.

    # 5.1. Need PM Data? Go fetch it first.
    # If PM data is missing or has an error, we need to get it from the data_agent.
    # Clear outcomes to ensure a fresh evaluation after fetching new data.
    if not state.pm_data or state.pm_data.get("error"):
        logger.info(f"Router for {intent_id}: PM data needed for evaluation. Routing to data_agent.")
        state.outcomes = [] # Clear old outcomes to force a fresh evaluation next.
        return "data_agent"

    # 5.2. Have PM Data, but need to evaluate it?
    # This path is taken right after `data_agent` fetches new data and returns to the router.
    # It ensures that `pm_evaluation_agent_node` is always called when new PM data is available.
    if not state.outcomes: # This means PM data is present, but outcomes aren't computed for it.
        logger.info(f"Router for {intent_id}: PM data available, evaluating outcomes. Routing to pm_evaluation_agent_node.")
        return "pm_evaluation_agent_node"

    # 5.3. Have PM Data AND it's been evaluated (outcomes are present). Now make decisions.
    # This is the central decision point after PM has been evaluated.
    if state.outcomes:
        all_targets_met = not any(o.get("fulfilmentStatus") != "FULFILLED" for o in state.outcomes)

        if all_targets_met:
            logger.info(f"Router for {intent_id}: All targets FULFILLED by current RAN state. Setting FULFILLED status and Ending.")
            state.fulfilmentStatus = "FULFILLED" # Explicitly set to FULFILLED
            return END # Immediate stop when fulfilled.
        else:
            # Targets not met. Decide next step based on whether we're actively monitoring an applied config.
            logger.info(f"Router for {intent_id}: Targets NOT met by current RAN state.")

            # Check if we are in an active monitoring phase (i.e., a config was applied and we're observing it).
            # `state.pm_evaluation_pending` is set by `orchestrator_agent` after applying a config.
            # `state.last_config_application_time` is also set by `orchestrator_agent`.
            if state.pm_evaluation_pending and state.last_config_application_time:
                elapsed_monitoring_time = (datetime.now(timezone.utc) - state.last_config_application_time).total_seconds()
                remaining_time = (state.observation_deadline - datetime.now(timezone.utc)).total_seconds()
                end_of_observation_for_this_config = (elapsed_monitoring_time >= remaining_time)

                logger.info(f"{GREEN}Router for {intent_id}: Elapsed monitoring time: {elapsed_monitoring_time} seconds. and {remaining_time} {RESET}")
                logger.info(f"{GREEN}Router for {intent_id}: end_of_observation_for_this_config: {end_of_observation_for_this_config} {RESET}")
                            
                if not end_of_observation_for_this_config:
                    logger.info(f"Router for {intent_id}: Targets not met.  Will attempt REFINE.")
                    state.pm_evaluation_pending = False # Reset for new strategy generation.
                    return "strategy_agent" # Go to strategy_agent for refinement.
                else:
                    logger.info(f"Router for {intent_id}: Targets not met during observation.")
                    state.outcomes = [] # Clear outcomes to force new PM data and re-evaluation cycle.
                    state.fulfilmentStatus = "NOT_FULFILLED"
                    if not state.report:
                        state.report = {"intentId": intent_id, "expectationFulfilmentResult": [{"targetStatus": "NOT_FULFILLED", "failureAnalysis": {"notFulfilledState": "TERMINATED", "reason": "Intent processing deadline exceeded."}}]}
                    state.pm_evaluation_pending = False
                    return END # Continue the monitoring loop.
            else:
                # This branch is for cases where targets are not met, but we are NOT in an active monitoring loop.
                # This includes:
                #   a) Initial intent processing (no strategy applied yet).
                #   b) After a user rejects a proposed strategy (initial or refined).
                #   c) After the LLM indicated it "gave up" on a refinement.
                logger.info(f"Router for {intent_id}: Targets not met, no active monitoring loop. Routing to strategy_agent for proposal/refinement.")
                state.pm_evaluation_pending = False # Not actively monitoring a specific applied config.
                return "strategy_agent" # Go to strategy_agent for initial proposal or general refinement.

    # --- 6. Fallback/Error State ---
    # This section should ideally not be reached if all states are covered.
    logger.error(f"Router for {intent_id}: Reached an unexpected state. Returning to strategy_agent as fallback.")
    state.pm_evaluation_pending = False # Don't assume monitoring in an error state.
    return "strategy_agent"

def build_graph():
    graph = StateGraph(IntentState)

    graph.add_node("strategy_agent", strategy_agent)
    graph.add_node("data_agent", data_agent_node)
    # graph.add_node("history_agent", history_agent) # <--- REMOVED
    graph.add_node("orchestrator_agent", orchestrator_agent_node)
    graph.add_node("pm_evaluation_agent_node", pm_evaluation_agent_node)
    graph.add_node("router_node", route_main_agent) # route_main_agent is the actual node function

    graph.set_entry_point("router_node")

    # Fixed edge from strategy_agent to the router_node
    graph.add_edge("strategy_agent", "router_node")

    # Conditional edges originate FROM THE DEDICATED ROUTER_NODE
    graph.add_conditional_edges(
        "router_node",
        _get_next_route_from_state,
        {
            "data_agent": "data_agent",
            # "history_agent": "history_agent", # <--- REMOVED
            "orchestrator_agent": "orchestrator_agent",
            "pm_evaluation_agent_node": "pm_evaluation_agent_node",
            "strategy_agent": "strategy_agent",
            END: END,
        }
    )

    # Fixed edges from all other 'worker' nodes back to the ROUTER_NODE
    fixed_edges = [
        ("strategy_agent", "router_node"),
        ("data_agent", "router_node"),
        # ("history_agent", "router_node"), # <--- REMOVED
        ("orchestrator_agent", "router_node"),
        ("pm_evaluation_agent_node", "router_node")
    ]
    for src, dest in fixed_edges:
        graph.add_edge(src, dest)

    return graph.compile()

if __name__ == "__main__":
    graph = build_graph() # Assuming build_graph uses the modified agents
    
    intent_input = {
        "intent": {
            "id": "INTENT_EE_Test_335",
            "intentExpectation": {
                "expectationObject": {"objectInstance": "SubNetwork_1", "ObjectTarget": ["7"]},
                "expectationTargets": [{"targetName": "RANEnergyEfficiency", "targetCondition": "IS_GREATER_THAN", "targetValue": "1200000", "bit": "bit/joule"}] # Example target value
            },
            "observationPeriod": 20
        }
    }
    
    initial_state = IntentState(intent=intent_input) # Corrected structure
    # initial_state = IntentState(**initial_state_dict) # Use the dict directly if it matches IntentState fields

    # Set deadline for the intent processing
    processing_start_time = datetime.now(timezone.utc)
    
    initial_state.observation_deadline = processing_start_time + timedelta(minutes=intent_input["intent"]["observationPeriod"])
    print(f"Intent processing deadline set to: {initial_state.observation_deadline.isoformat()}")
    # Initialize fields for the simplified single-strategy refinement loop
    initial_state.current_strategy_type = None # strategy_agent will set this on first call
    initial_state.current_config_parameters = None
    initial_state.total_refinement_attempts_for_intent = 0 # Will be incremented by strategy_agent

    logger.info(f"Starting graph for intent: {initial_state.intent.get('id', 'N/A')}, "
                f"Deadline: {initial_state.observation_deadline.isoformat() if initial_state.observation_deadline else 'N/A'}")
    
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
    config = RunnableConfig(recursion_limit=1000)
    print(config)
    final_state = graph.invoke(initial_state,config)
    # logger.info(f"Graph execution complete. Final state: {final_state.model_dump_json(indent=2)}")
    # logger.info(f"All STRATEGY_ATTEMPTS: {json.dumps(STRATEGY_ATTEMPTS, indent=2)}")
    # # --- End of actual graph run block ---
