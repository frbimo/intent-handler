
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
import json
from datetime import datetime
from uuid import uuid4
from enum import Enum
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from src.netconf_cli import NETCONFCLIENT
from src.pm_data import get_measurement_data

import re
import logging
import os
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from openai import OpenAI
# for debugging
import inspect

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


# Pydantic model for standardized LLM output
class LLMOutput(BaseModel):
    result: Any
    explanation: str
    error: Optional[str] = None

# State definition
class IntentState(BaseModel):
    intent: Dict  # Intent details, including targets (e.g., AveDLRANUEThpt > 300 Mbps) and subnetwork
    pm_data: Dict = {}  # Raw performance management data (e.g., activeUsers, interferenceLevel) for analysis
    # pm_summary: Dict = {}  # Summarized key metrics and constraints from PM Data Agent (e.g., activeUsers: 50)
    history_summary: Dict = {}  # Filtered relevant past attempts from History Agent (e.g., successful strategies)
    strategies: Dict = {}  # Generated strategies and configurations (e.g., ThroughputOptimization for Cell_1)
    outcomes: List[Dict] = []  # Results of applied strategies (e.g., targetName, targetAchievedValue)
    report: Dict = {}  # Fulfillment report with status and failure analysis (e.g., NOT_FULFILLED reason)
    fulfilmentStatus: str = "NONE"  # Intent fulfillment status (PENDING, FULFILLED, NOT_FULFILLED)
    attempt_count: int = 0  # Number of strategy application attempts (max 3)
    applied: bool = False  # Indicates if strategies have been applied to the network
    patches: Dict = {}  # Configuration patches for affected cells (e.g., Cell_1: {BandwidthConfig: {}})

"""# New Section"""

# CELL_CAPABILITIES = {
#     "Cell_1": {
#         "maxBandwidth": 100,  # MHz
#         "defaultBandwidth": 40,
#         "supportedARFCN": [636000, 638000],  # n78 band
#         "supportedSSBFrequency": [3500000],  # kHz,
#         "supportedConfigs": {
#             "BandwidthConfig": ["ChannelBandwidthUL", "ChannelBandwidthDL"]
#         }
#     },
#     "Cell_3": {
#         "maxBandwidth": 40,
#         "defaultBandwidth": 20,
#         "supportedARFCN": [636000],
#         "supportedSSBFrequency": [3500000],
#         "supportedConfigs": {
#             "BandwidthConfig": ["ChannelBandwidthUL", "ChannelBandwidthDL",  "ARFCNUL", "ARFCNDL"]
#         }
#     },
#     "Cell_4": {
#         "maxBandwidth": 40,
#         "defaultBandwidth": 20,
#         "supportedARFCN": [636000],
#         "supportedSSBFrequency": [3500000],
#         "supportedConfigs": {
#             "BandwidthConfig": ["ChannelBandwidthUL", "ChannelBandwidthDL",  "ARFCNUL", "ARFCNDL"]
#         }
#     }
# }

CELL_CAPABILITIES = {
    "1": {
        "maxBandwidth_MHz": 100,
        "defaultBandwidth_MHz": 40,
        "supportedARFCNs": [636000, 638000],
        "supportedSSBFrequency_kHz": [3500000],
        "configurableParameters": {
            "strategy": "bandwidthOptimization",  # Strategy name
            "description": "Parameters to adjust for bandwidth optimization (e.g., maximizing throughput).",
            "parameters": {
                "ChannelBandwidthUL": {"units": "MHz", "min": 5, "max": 100},
                "ChannelBandwidthDL": {"units": "MHz", "min": 5, "max": 100}
            }
        }
    },
    "2": {
        "maxBandwidth_MHz": 40,
        "defaultBandwidth_MHz": 20,
        "supportedARFCNs": [636000],
        "supportedSSBFrequency_kHz": [3500000],
        "configurableParameters": {
            "strategy": "interferenceMitigation",
            "description": "Parameters to adjust to reduce interference (e.g., changing frequency).",
            "parameters": {
                "configuredMaxTxPower": {"units": "dBm", "min": 0, "max": 40},
                # "ChannelBandwidthUL": {"units": "MHz", "min": 5, "max": 40},
                # "ChannelBandwidthDL": {"units": "MHz", "min": 5, "max": 40},
                # "ARFCNUL": {"units": "absolute frequency channel number"},
                # "ARFCNDL": {"units": "absolute frequency channel number"}
            }
        }
    },
    "3": {
        "maxBandwidth_MHz": 40,
        "defaultBandwidth_MHz": 20,
        "supportedARFCNs": [636000],
        "supportedSSBFrequency_kHz": [3500000],
        "configurableParameters": {
            "strategy": "coverageExtension",
            "description": "Parameters to adjust to extend the cell's coverage area (e.g., reducing bandwidth).",
            "parameters": {
                "configuredMaxTxPower": {"units": "dBm", "min": 0, "max": 40},
                # "ChannelBandwidthUL": {"units": "MHz", "min": 5, "max": 40},
                # "ChannelBandwidthDL": {"units": "MHz", "min": 5, "max": 40},
                # "ARFCNUL": {"units": "absolute frequency channel number"},
                # "ARFCNDL": {"units": "absolute frequency channel number"}
            }
        }
    }
}

# Strategy and configuration templates
STRATEGY_DATABASE = {
    "PowerOptimization": {"configType": "EnergySavingConfig"},
    "ThroughputOptimization": {"configType": "BandwidthConfig"},
    "LatencyOptimization": {"configType": "SchedulingConfig"},
    "InterferenceMitigation": {"configType": "FrequencyAdjustmentConfig"}
    # Users can add new strategies, e.g., "CoverageOptimization": {"configType": "BeamformingConfig"}
}

CONFIG_TEMPLATES = {
   
    "CoverageOptimization": {
        "parameters": {
            "ARFCNUL": {"default": 636000},
            "ARFCNDL": {"default": 636000},
            "ChannelBandwidthUL": {"default": 5, "options": [5, 10, 20]},
            "ChannelBandwidthDL": {"default": 5, "options": [5, 10, 20]},
            "SSBFrequency": {"default": 3500000}
        },
        "affectedCells": []
    },
    "ThroughputOptimization": {
        "parameters": {
            "ARFCNUL": {"default": 636000},
            "ARFCNDL": {"default": 636000},
            "ChannelBandwidthUL": {"default": 100, "options": [5, 10, 20, 40,  60,  80,  100]},
            "ChannelBandwidthDL": {"default": 100, "options": [5, 10, 20, 40,  60,  80,  100]},
            "SSBFrequency": {"default": 3500000},
        },
        "affectedCells": []
    },
}



# Dictionary-based store_attempt (artifact ID: 78dda6c9-d805-4c0b-a050-4fb7254dff47)
STRATEGY_ATTEMPTS = {}

# Function to record attempt for history or data analytics
def store_attempt(intent_id: str, strategies: Dict, outcomes: List[Dict], failure_analysis: Dict, pm_data: Dict, cell_capabilities: Dict, fulfilment_status: str) -> None:
    print(f"""##preview store_attempt##
    intent_id: {intent_id}
    strategies: {strategies}
    outcomes: {outcomes}
    failure_analysis: {failure_analysis}
    pm_data: {pm_data}
    cell_capabilities: {cell_capabilities}
    """)
    """Store strategy attempt with context for reuse and learning."""
    # if not intent_id or not strategies or not isinstance(strategies, list) or not outcomes or not isinstance(outcomes, list):
    if not intent_id or not strategies:
        raise ValueError("Invalid input for store_attempt")
    attempt_id = str(uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"
    # Add metadata for strategy context
    metadata = {
        # "user_load": "high" if pm_data.get("activeUsers", 0) > 30 else "low",
        # "interference": "high" if pm_data.get("interferenceLevel", -85) > -80 else "low",
        "intent_type": [t["targetName"] for t in intent["intent"]["intentExpectation"]["expectationTargets"]]
    }
    attempt_data = {
        "intent_id": intent_id,
        "strategy_json": strategies,
        "outcome_json": outcomes,
        "failure_analysis": failure_analysis,
        "pm_data": pm_data,
        "cell_capabilities": cell_capabilities,
        "metadata": metadata,
        "fulfilment_status": fulfilment_status,
        "timestamp": timestamp
    }
    STRATEGY_ATTEMPTS[attempt_id] = attempt_data
    logger.info(f"Stored attempt {attempt_id} for intent {intent_id} with status {fulfilment_status}")


# # Mock tools (replace with actual implementations)
# def fetch_pm_data(subnetwork: str, cell_id:str, start_time: str, end_time: str) -> Dict:
#     """Mock PM data fetch."""
#     # return {
#     #     "dlPRBUtilization": 10,
#     #     "activeUsers": 5,
#     #     "channelCorrelation": 0.6,
#     #     "interferenceLevel": -85,
#     #     "pmDataReference": {"subnetwork": subnetwork, "start_time": start_time, "end_time": end_time}
#     # }
#     return {"txPower": 30}

# def fetch_outcome(subnetwork: str, start_time: str, end_time: str) -> List[Dict]:
#     """Mock outcome fetch."""
#     return [
#         {"targetName": "RANEnergyConsumption", "targetAchievedValue": 1000, "targetValue": 1000},
#         {"targetName": "AveDLRANUEThpt", "targetAchievedValue": 350, "targetValue": 300}
#     ]

# def apply_patch(patches: Dict) -> bool:
#     """Mock patch application."""
#     print(f"Applying patches: {json.dumps(patches, indent=2)}")
#     return True

base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")

if not base_url:
    raise ValueError("OPENAI_BASE_URL environment variable not set.")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(
    base_url = base_url,
    api_key = api_key
)


def call_llm(messages: str) -> Dict:
    """Call API to generate strategies and configurations."""
        # Send prompt to Grok 3
    response = client.chat.completions.create(
        model= "meta/llama-3.1-70b-instruct",
        messages=messages,
        max_tokens=4000,
        temperature=0.2,
        top_p=0.7,
        stream=True
    #     response_format={
    #     'type': 'json_object'
    # }
    )

    collected_messages=[]
    print("\n-- Streaming response --")
    for chunk in response:
      if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
        collected_messages.append(chunk.choices[0].delta.content)
    print("\n-- End Streaming response --")

    llm_resp = "".join(collected_messages)
    json_match = re.search(r'```json(.*?)```', llm_resp, re.DOTALL)
    json_match_noword = re.search(r'```(.*?)```', llm_resp, re.DOTALL)

    if json_match:
      json_string = json_match.group(1)

      try:
        data = json.loads(json_string)
        return data
      except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Problematic JSON string: {json_string}")
        return {"error": "Error decoding JSON"}

    elif json_match_noword:
      json_string = json_match_noword.group(1)
      try:
        data = json.loads(json_string)
        print("call_llm: {0}".format(data))
        return data
      except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Problematic JSON string: {json_string}")
        return {"error": "Error decoding JSON"}

    else:
      print("No JSON block found in the LLM response.")
      return {"error": "No JSON block found in the LLM response"}


    print("\n-- End of streaming response --")

def history_agent(state: IntentState) -> IntentState:
    """Filter STRATEGY_ATTEMPTS to relevant past attempts."""
    print(f"Inside function: {inspect.currentframe().f_code.co_name}")

    # Process PM data from pm_data_analyzer_agent
    intent = state.intent
    intent_type = [t["targetName"] for t in intent["intent"]["intentExpectation"]["expectationTargets"]]

    user_prompt = f"""
Filter past strategy attempts for relevance to the current intent and PM data.

Intent Type: {json.dumps(intent_type)}

Past Attempts: {json.dumps([a for a in STRATEGY_ATTEMPTS.values() if a["metadata"]["intent_type"] == intent_type][-10:])}

Instructions:
- Select attempts matching Intent Type and similar PM conditions.
- Rank by fulfilment_status (FULFILLED > NOT_FULFILLED) and recency.
- Output JSON with:
  - result: {{relevant_attempts: [{{attempt_id, strategy_json, outcome_json, fulfilment_status}}]}}
  - explanation: String explaining the filtering
  - error: Null or error message
Example:
```json
{{
  "result": {{
    "relevant_attempts": [
      {{
        "attempt_id": "123",
        "strategy_json": {{...}},
        "outcome_json": [{{...}}],
        "fulfilment_status": "FULFILLED"
      }}
    ]
  }},
  "explanation": "Selected recent successful attempts for AveDLRANUEThpt",
  "error": null
}}
```
    """

    # debug
    print("\n-- DEBUG user_prompt --")
    print(user_prompt)
    print("\n-- End of DEBUG user_prompt --")
    messages=[
      {
        "role": "system",
        "content": "You are a 5G network analysis agent. Your task is to analyze past strategy attempts. Based on analysis you need to generate a structured JSON output. You will be given Performance Management (PM) data and CELL_CAPABILITIES data for analysis. You also be given example of structure JSON for output."
      },
      {
        "role": "user",
        "content": user_prompt  }
    ]
    response = call_llm(messages)

    if response.get("result") is not None:
      state.history_summary = response
      return state
    else:
      print("No JSON block found in the LLM response.")
      state.history_summary = {}
      return state


def strategy_agent(state: IntentState) -> IntentState:
    """Generate strategies based on intent, PM summary, history, and STRATEGY_DATABASE."""
    print(f"Inside function: {inspect.currentframe().f_code.co_name}")
    intent = state.intent
    intent_id = intent["intent"]["id"]
    MAX_ATTEMPTS = 3

    # Check max attempts
    if state.attempt_count >= MAX_ATTEMPTS:
        state.fulfilmentStatus = "NOT_FULFILLED"
        state.report = {
            "intentId": intent_id,
            "expectationFulfilmentResult": [{
                "targetStatus": "NOT_FULFILLED",
                "failureAnalysis": {
                    "notFulfilledState": "FULFILMENTFAILED",
                    "reason": "Maximum attempts reached"
                }
            }]
        }
        return state

    # Increment attempt count
    attempt_count = state.attempt_count + 1
    state.attempt_count = attempt_count

    # Parse intent
    subnetwork = intent["intent"]["intentExpectation"]["expectationObject"]["objectInstance"]
    object_target = intent["intent"]["intentExpectation"]["expectationObject"]["ObjectTarget"]
    targets = intent["intent"]["intentExpectation"]["expectationTargets"]
    start_time = intent["intent"].get("startTime", datetime.now())
    end_time = intent["intent"].get("endTime", datetime.now())


    # Determine if PM data is needed
    pm_data_needed = not state.pm_data and attempt_count == 1
    if attempt_count > 1:
        prev_failure = state.report.get("expectationFulfilmentResult", [{}])[0].get("failureAnalysis", {})
        if prev_failure.get("reason", "").lower().startswith("stale data"):
            pm_data_needed = True
            state.pm_data = {}
            # state.pm_summary = {}
            state.history_summary = {}
    if pm_data_needed:
        state.pm_data = {"subnetwork": subnetwork, "start_time": start_time, "end_time": end_time}
        
        # print("i need pm_data_needed")
        return state

    current_tx_power = {}
    try: 
        for cell_id in object_target:
            if cell_id in CELL_CAPABILITIES:
                netconf_client = NETCONFCLIENT(cell_id, cell_id)
                tx_power_value = netconf_client.get_current_tx_power()
                current_tx_power[cell_id] = tx_power_value
    except Exception as e:
        logger.warning(f"Failed to get current tx power: {e}")
        current_tx_power = {"error": "Failed to retrieve tx power"}

    # Generate strategies
    # pm_summary = state.pm_summary["result"]
    history_summary = state.history_summary["result"]
    prev_failure = state.report["expectationFulfilmentResult"][0]["failureAnalysis"] if state.report.get("expectationFulfilmentResult") else {}
    user_prompt = f"""
Generate appliedStrategies for intent {intent_id} targeting {subnetwork} {object_target}.

Intent Targets: {json.dumps(targets)}
Relevant Past Attempts: {json.dumps(history_summary["relevant_attempts"])}
Previous Failure: {json.dumps(prev_failure)}
Current Network State: {json.dumps(state.pm_data)}
Current Tx Power: {json.dumps(current_tx_power)}

STRATEGY_DATABASE: {json.dumps(STRATEGY_DATABASE)}
CONFIG_TEMPLATES: {json.dumps(CONFIG_TEMPLATES)}
CELL_CAPABILITIES: {json.dumps(CELL_CAPABILITIES)}

Instructions:
- If Current Network State is not present, you cannot generate appliedStrategies and should return error.
- STRATEGY_DATABASE lists available strategyType and configType. Use these or propose new strategyType if needed.
- CONFIG_TEMPLATES defines configuration templates. Use these or propose new configType if justified.
- Proposed new configType should only contain supportedConfigs within CELL_CAPABILITIES. CELL_CAPABILITIES categorized by cell id.
- Relevant Past Attempts include successful or failed strategies. Reuse successful ones or learn from failures.
- For each intent target, select or propose a strategy based on PM Summary and intent requirements.
- Generate one configuration per affected cell.
- If object target cell id is not specified or is not within cell capabilities database, reject the intent.
- Output JSON with:
  - result: {{appliedStrategies: [{{strategyId, strategyType, configurations: [{{configId, configType, parameters, affectedCells, appliedAt}}]}}]}}
  - explanation: String detailing decisions, trade-offs, risk mitigation
  - error: Null or error message

The following are examples for the output JSON:
Examples#1:
```json
{{
  "result": {{
    "appliedStrategies": [
      {{
        "strategyId": "S1",
        "strategyType": "RANEnergyEfficiency",
        "configurations": [
          {{
            "configId": "C1",
            "configType": "TxControl",
            "parameters": {{
              "antennaMaskName": "S7/NR1/C1:12x8",
              "antennaMask": "1111111111111111111111111111111111111111111111111111000000000000",
            }},
            "affectedCells": ["1"],
            "appliedAt": "2025-04-28T10:00:00Z"
          }}
        ]
      }}
    ]
  }},
  "explanation": "Parameters value serve to turn off array of mask to save energy",
  "error": null
}}
```
Examples#2:
```json
{{
  "result": {{
    "appliedStrategies": [
      {{
        "strategyId": "S2",
        "strategyType": "RANEnergyEfficiency",
        "configurations": [
          {{
            "configId": "C1",
            "configType": "TxConfig",
            "parameters": {{
              "configuredMaxTxPower": 20
            }},
            "affectedCells": ["2"],
            "appliedAt": "2025-04-28T10:00:00Z"
          }}
        ]
      }}
    ]
  }},
  "explanation": "Selected 20 dbm tx power to reduce energy consumption",
  "error": null
}}
```
    """
    print("\n-- DEBUG user_prompt --")
    print(user_prompt)
    print("\n-- End of DEBUG user_prompt --")
    messages=[
      {
        "role": "system",
        "content": "You are an AI assistant for 5G network optimization. Generate strategies and configurations based on the provided intent, PM data, and constraints. Output JSON with 'explanation' (string) and 'configuration' (appliedStrategies JSON)."
      },
      {
        "role": "user",
        "content": user_prompt  }
    ]
    response = call_llm(messages)

    # Handle model response
    # explanation = response.get("explanation", "No explanation provided.")
    # print(f"Strategy Explanation: {explanation}")
    user_input = input("Proceed with configuration? Type 'yes' to confirm: ")

    if user_input.lower() == "yes":
        state.strategies = response.get("result", {}).get("appliedStrategies", [])
        state.fulfilmentStatus = "ACKNOWLEDGED"
    else:
        print("Configuration not approved. Retrying or terminating.")
        state.strategies = {}
        if state.attempt_count < MAX_ATTEMPTS:
            state.pm_data = {}
            # state.pm_summary = {}
            state.history_summary = {}
        else:
            state.fulfilmentStatus = "NOT_FULFILLED"
            state.report = {
                "intentId": intent_id,
                "expectationFulfilmentResult": [{
                    "targetStatus": "NOT_FULFILLED",
                    "failureAnalysis": {
                        "notFulfilledState": "TERMINATED",
                        "reason": "User did not approve configuration"
                    }
                }]
            }
        return state

    # Generate patches for all affected cells
    patches = {}
    for strategy in state.strategies:
        for config in strategy.get("configurations", []):
            for cell in config["affectedCells"]:
                patches[cell] = patches.get(cell, {})
                patches[cell][config["configType"]] = config["parameters"]
    state.patches = patches
    return state

def route_main_agent(state: IntentState) -> str:
    print(f"Inside function: {inspect.currentframe().f_code.co_name}")
    print("## current state: ", state.fulfilmentStatus )
    print("## current strategies: ", state.strategies )

    if state.fulfilmentStatus != "NONE" and state.attempt_count <=3:
        print("go to orchestrator_agent")
        return "orchestrator_agent"

    if state.fulfilmentStatus in ["FULFILLED", "NOT_FULFILLED"]:

        store_attempt(
            intent_id=state.intent["intent"]["id"],
            strategies=state.strategies,
            outcomes=state.outcomes,
            failure_analysis=state.report.get("expectationFulfilmentResult", [{}])[0].get("failureAnalysis", {}),
            pm_data=state.pm_data,
            cell_capabilities=CELL_CAPABILITIES,
            fulfilment_status=state.fulfilmentStatus
        )

        return END
    if state.pm_data.get("subnetwork") and not state.pm_data.get("activeUsers"):  # Metadata only

        return "data_agent"
    # if state.pm_data.get("activeUsers") and not state.pm_summary:  # Real metrics ready
    #     return "pm_data_agent"
    # if state.pm_summary and not state.history_summary:
    #     return "history_agent"
    if not state.history_summary:
        return "history_agent"
    if state.history_summary and not state.strategies:
        return "strategy_agent"
    if state.patches and not state.applied:
        return "orchestrator_agent"
    if state.outcomes:
        print("state.outcomes : ", state.outcomes )
        # Evaluate outcomes
        report = {
            "intentId": state.intent["intent"]["id"],
            "expectationFulfilmentResult": []
        }
        fulfilled = True
        for outcome in state.outcomes:
            target_name = outcome["targetName"]
            achieved = outcome["targetAchievedValue"]
            target_value = next(t["targetValue"] for t in state.intent["intent"]["expectationTargets"] if t["targetName"] == target_name)
            target_condition = next(t["targetCondition"] for t in state.intent["intent"]["expectationTargets"] if t["targetName"] == target_name)

            status = "NOT_FULFILLED"
            if target_condition == "IS_GREATER_THAN":
                status = "FULFILLED" if achieved > target_value else "NOT_FULFILLED"
            elif target_condition == "IS_LESS_THAN":
                status = "FULFILLED" if achieved < target_value else "NOT_FULFILLED"

            report["expectationFulfilmentResult"].append({
                "targetName": target_name,
                "targetStatus": status,
                "targetAchievedValue": achieved,
                "appliedStrategies": state.strategies.get("appliedStrategies", []),
                "failureAnalysis": {"reason": f"{target_name} not met"} if status == "NOT_FULFILLED" else {}
            })
            if status == "NOT_FULFILLED":
                fulfilled = False

        state.report = report
        state.fulfilmentStatus = "FULFILLED" if fulfilled else "NOT_FULFILLED"

        store_attempt(
            intent_id=state.intent["intent"]["id"],
            strategies=state.strategies,
            outcomes=state.outcomes,
            failure_analysis=state.report.get("expectationFulfilmentResult", [{}])[0].get("failureAnalysis", {}),
            pm_data=state.pm_data,
            cell_capabilities=CELL_CAPABILITIES,
            fulfilment_status=state.fulfilmentStatus
        )

        if state.fulfilmentStatus == "NOT_FULFILLED" and state.attempt_count < 3:
            state.strategies = {}
            state.outcomes = []
            state.patches = {}
            state.applied = False
            state.pm_data = {}
            # state.pm_summary = {}
            state.history_summary = {}
            return "data_agent"
        elif state.fulfilmentStatus == "NOT_FULFILLED":
            state.report["expectationFulfilmentResult"][0]["failureAnalysis"] = {
                "notFulfilledState": "FULFILMENTFAILED",
                "reason": "Maximum attempts reached"
            }
    return END

def data_agent_node(state: IntentState) -> IntentState:
    print(f"Inside function: {inspect.currentframe().f_code.co_name}")

    cell_id = ""
    for each in state.intent["intent"]["intentExpectation"]["expectationObject"]["ObjectTarget"]:
        if each in CELL_CAPABILITIES:
            cell_id = each
            break

    # for strategy in state.strategies:
    #     for config in strategy.get("configurations", []):
    #         cell_id = config["affectedCells"][0]
            # if each in CELL_CAPABILITIES[config["affectedCells"][0]]:

    pm_data = state.pm_data
    if not pm_data.get("subnetwork"):
        logger.error("No subnetwork specified in pm_data")
        state.pm_data = {}
        return state

    # Simulate fetching real PM metrics (replace with actual API/database call)
    real_metrics = json.loads(get_measurement_data(cell_id,cell_id))
    print("##real_metrics: ", real_metrics)
    # real_metrics["cell_id"] = cell_id
    # real_metrics["subnetwork"] = pm_data["subnetwork"]
 
    state.pm_data = real_metrics
    logger.info(f"Fetched real PM metrics for {pm_data['subnetwork']}: {real_metrics}")
    return state


def orchestrator_agent_node(state: IntentState) -> IntentState:
    """Apply strategies to the network."""
    state.applied = True  # Simulate application
    for strategy in state.strategies:
        for config in strategy.get("configurations", []):
            for each in config["parameters"]:
                if each in CELL_CAPABILITIES[config["affectedCells"][0]]["configurableParameters"]["parameters"]:
                    new_tx_power = config["parameters"][each]
                    print(f"Applying {each} with value {new_tx_power} to {config['affectedCells'][0]}")
                else:
                    print(f"Invalid parameter {each} for cell {config['affectedCells'][0]}")
           

    object_target = intent["intent"]["intentExpectation"]["expectationObject"]["ObjectTarget"]
    print("##object_target \n",object_target)
    
    for each in object_target:
        netconf_client = NETCONFCLIENT(object_target, object_target)
        netconf_client.configure_tx_power(new_tx_power)

    if state.attempt_count > 2:
        state.outcomes = [
        # {"targetName": "AveDLRANUEThpt", "targetAchievedValue": 350, "targetValue": 300},
        {"targetName": "RANEnergyEfficiency", "targetAchievedValue": 10, "targetValue": 7}
    ]  # 
    else:
         state.outcomes = [
        # {"targetName": "AveDLRANUEThpt", "targetAchievedValue": 350, "targetValue": 300},
        {"targetName": "RANEnergyEfficiency", "targetAchievedValue": 10, "targetValue": 1}
    ]  # 
    state.attempt_count +=1
    # state.outcomes = [
    #     # {"targetName": "AveDLRANUEThpt", "targetAchievedValue": 350, "targetValue": 300},
    #     {"targetName": "RANEnergyEfficiency", "targetAchievedValue": 900, "targetValue": 1000}
    # ]  # Simulate outcomes
    logger.info("Strategies applied, outcomes generated")
    return state

# Graph definition
def build_graph():
    graph = StateGraph(IntentState)

    graph.add_node("strategy_agent", strategy_agent)
    graph.add_node("data_agent", data_agent_node)
    # graph.add_node("pm_data_analyzer_agent", pm_data_analyzer_agent)
    graph.add_node("history_agent", history_agent)
    graph.add_node("orchestrator_agent", orchestrator_agent_node)

    # Edges
    graph.set_entry_point("strategy_agent")
    graph.add_edge("data_agent", "history_agent")
    graph.add_edge("history_agent", "strategy_agent")
    graph.add_edge("orchestrator_agent", "strategy_agent")

    # Conditional edges
    graph.add_conditional_edges("strategy_agent", route_main_agent, {
        "data_agent": "data_agent",
        "history_agent": "history_agent",
        "orchestrator_agent": "orchestrator_agent",
        END: END,
        #  "main_agent": "main_agent"
    })

    return graph.compile()

# Example usage


if __name__ == "__main__":
    graph = build_graph()

    intent = {
        "intent": {
            "id": "INTENT_001",
            "intentExpectation": {
                "expectationObject": {
                    "objectInstance": "SubNetwork_1",
                    "ObjectTarget": ["2"]
                    },
                "expectationTargets": [
                    # {targetName": "RANEnergyConsumption", "targetCondition": "IS_LESS_THAN", "targetValue": 1000, "targetUnit": "W"},
                    # {"targetName": "SINR", "targetCondition": "IS_EQUAL_OR_GREATER_THAN", "targetValue": 13,  "targetUnit": "dB"},
                    {
                        "targetName": "RANEnergyEfficiency",
                        "targetCondition": "IS_GREATER_THAN",
                        "targetValueRange": "10",
                        "targetUnit": "percentage"
                    }
                ]
            },
            "startTime": "2025-10-28T22:00:00Z",
            "endTime": "2025-10-28T22:10:00Z"
        }
    }
    state = IntentState(intent=intent)
    result = graph.invoke(state)


    print("STRATEGY_ATTEMPTS:", json.dumps(STRATEGY_ATTEMPTS, indent=2))



