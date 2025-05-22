
CELL_CAPABILITIES = {
    "Cell_1": {
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
    "Cell_3": {
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
    "Cell_4": {
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

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = "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"
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
# --- History Analysis Service Logic (adapted from your history_agent) ---
def history_analysis_service_logic(intent_id: str, intent_targets: List[Dict], all_attempts: Dict) -> Dict:
    logger.info(f"[Service Logic] HistoryAnalysisService processing for intent {intent_id}")
    # Simplified for brevity: your original history_agent logic for LLM call
    # This function would be the core of the external HistoryAnalysisService
    
    # Construct prompt similar to your original history_agent
    intent_type = [t["targetName"] for t in intent_targets] # Assuming intent_targets is passed
    user_prompt = f"""
Filter past strategy attempts for relevance to the current intent.
Intent Type: {json.dumps(intent_type)}
Past Attempts: {json.dumps([a for a in all_attempts.values() if a["metadata"]["intent_type"] == intent_type][-10:])}
Instructions: Select attempts matching Intent Type and similar PM conditions. Rank by fulfilment_status and recency.
Output JSON with: result: {{relevant_attempts: [...]}}, explanation: "...", error: null
Example:
```json
{{
  "result": {{ "relevant_attempts": [{{ "attempt_id": "123", ... }}] }},
  "explanation": "Selected recent successful attempts for {intent_type}", "error": null
}}
"""
    messages=[
    {"role": "system", "content": "You are a 5G network history analysis agent..."},
    {"role": "user", "content": user_prompt}
    ]
    response = call_llm(messages) # Assuming call_llm is globally available or passed

    if response.get("result") is not None:
        return {"status": "SUCCESS", "data": response}
    else:
        return {"status": "ERROR", "error_message": "Failed to get history analysis from LLM", "data": response}
    
    # --- Energy Saving Strategy Service Logic (adapted from parts of your strategy_agent) ---
def energy_saving_strategy_service_logic(
    intent_id: str,
    intent_targets: List[Dict],
    pm_data: Dict,
    history_summary_result: Dict, # This comes from HistoryAnalysisService
    cell_capabilities: Dict,
    strategy_database: Dict,
    config_templates: Dict
    ) -> Dict:
    logger.info(f"[Service Logic] EnergySavingStrategyService processing for intent {intent_id}")
    # This function would be the core of the external EnergySavingStrategyService
    # Focuses ONLY on energy saving. The prompt needs to be tailored.

    # Simplified prompt for energy saving:
    user_prompt = f"""
Generate energy-saving strategies for intent {intent_id}.
Intent Targets: {json.dumps(intent_targets)}
PM Data: {json.dumps(pm_data)}
Relevant Past Attempts (History): {json.dumps(history_summary_result.get("relevant_attempts", []))}
CELL_CAPABILITIES: {json.dumps(cell_capabilities)}
STRATEGY_DATABASE (filter for energy related): {json.dumps({k:v for k,v in strategy_database.items() if "Energy" in k or "Power" in k})}
CONFIG_TEMPLATES (filter for energy related): {json.dumps({k:v for k,v in config_templates.items() if "Energy" in k or "Power" in k})}
Instructions:
Propose strategies focusing on energy efficiency (e.g., PowerOptimization).
Use STRATEGY_DATABASE and CONFIG_TEMPLATES.
Ensure proposed configurations are within CELL_CAPABILITIES.
Output JSON with: result: {{appliedStrategies: [...]}}, explanation: "...", error: null
Example:
{{
  "result": {{
    "appliedStrategies": [
      {{
        "strategyId": "ES1", "strategyType": "PowerOptimization", 
        "configurations": [{{ "configId": "C1", "configType": "EnergySavingConfig", "parameters": {{...}}, "affectedCells": ["Cell_X"]}}]
      }}
    ]
  }},
  "explanation": "Proposed power reduction for Cell_X due to low traffic.", "error": null
}}
"""
    messages=[
    {"role": "system", "content": "You are an AI assistant for 5G energy saving strategies..."},
    {"role": "user", "content": user_prompt}
    ]
    response = call_llm(messages)

    if response.get("result", {}).get("appliedStrategies"):
        return {"status": "SUCCESS", "data": {"strategies": response["result"]["appliedStrategies"], "explanation": response.get("explanation")}}
    else:
        return {"status": "ERROR", "error_message": "Failed to generate energy saving strategies", "data": response}
    

    # --- Conflict Resolution Service Logic (New) ---
def conflict_resolution_service_logic(
    intent_id: str,
    proposed_strategies_by_agent: Dict[str, List[Dict]], # e.g. {"EnergySavingService": [...], "ThroughputService": [...]}
    intent_targets: List[Dict] # To understand priorities
    ) -> Dict:
    logger.info(f"[Service Logic] ConflictResolutionService processing for intent {intent_id}")
    # This function would be the core of the external ConflictResolutionService
    # If only one set of strategies, it might just pass them through or validate.
    # If multiple, it uses LLM or rules to resolve.

    if not proposed_strategies_by_agent:
        return {"status": "ERROR", "error_message": "No strategies provided for conflict resolution."}

    # For now, if only one agent proposed strategies, assume no conflict
    # In a real scenario, even single-agent strategies might need validation against policies
    if len(proposed_strategies_by_agent) == 1:
        agent_name, strategies = list(proposed_strategies_by_agent.items())[0]
        logger.info(f"Only one set of strategies from {agent_name}, passing through.")
        return {"status": "SUCCESS", "data": {"resolved_strategies": strategies, "explanation": f"Strategies from {agent_name} adopted directly."}}

    # Placeholder for LLM-based conflict resolution if multiple strategy types
    # You would build a prompt asking the LLM to reconcile the different proposals.
    # For now, let's naively merge or pick one (this needs to be much smarter)
    all_strategies = []
    explanation_parts = []
    for agent_name, strats in proposed_strategies_by_agent.items():
        all_strategies.extend(strats)
        explanation_parts.append(f"Considering strategies from {agent_name}")

    # Extremely naive: just take all. A real CR agent would be complex.
    # A better naive approach: if "EnergySaving" and "Throughput" conflict on same param for same cell,
    # it would need a policy or LLM judgement.
    resolved_strategies = all_strategies 
    explanation = ". ".join(explanation_parts) + ". Naively combined all strategies. Advanced conflict resolution needed."

    logger.warning("Using naive conflict resolution. Implement proper logic.")
    return {"status": "SUCCESS", "data": {"resolved_strategies": resolved_strategies, "explanation": explanation}}


    # --- Orchestration Service Logic (adapted from your orchestrator_agent_node) ---
def orchestration_service_logic(intent_id: str, resolved_strategies: List[Dict], intent_object_target: List[str], current_attempt_count: int) -> Dict:
    logger.info(f"[Service Logic] OrchestrationService processing for intent {intent_id}")
    # This function would be the core of the external OrchestrationService
    applied_patches = {}
    netconf_application_log = []

    for strategy in resolved_strategies:
        for config in strategy.get("configurations", []):
            for cell_id in config.get("affectedCells", []): # Ensure affectedCells is present
                if cell_id not in CELL_CAPABILITIES:
                    logger.warning(f"Cell {cell_id} not found in CELL_CAPABILITIES. Skipping config.")
                    continue
                
                applied_patches.setdefault(cell_id, {})
                applied_patches[cell_id][config["configType"]] = config["parameters"]

                for param_name, param_value in config.get("parameters", {}).items():
                    # Check if parameter is configurable for the cell
                    if param_name in CELL_CAPABILITIES[cell_id]["configurableParameters"]["parameters"]:
                        log_msg = f"Applying {param_name} with value {param_value} to {cell_id}"
                        logger.info(log_msg)
                        netconf_application_log.append(log_msg)
                        
                        # SIMULATE NETCONF CALL
                        # In a real system, this is where you'd use your NETCONFCLIENT
                        # For now, we assume intent_object_target refers to something configurable
                        # and param_name is what we configure (e.g. TxPower)
                        # This part needs careful mapping from strategy parameters to actual NETCONF operations
                        if "TxPower" in param_name: # Example
                            for target_device_ip in intent_object_target: # Assuming these are IPs or resolvable names
                                try:
                                    # netconf_client = NETCONFCLIENT(target_device_ip, target_device_ip) # Your actual client
                                    # netconf_client.configure_tx_power(param_value) # Example method
                                    logger.info(f"SIMULATED: NETCONF set {param_name}={param_value} on {target_device_ip} for {cell_id}")
                                except Exception as e:
                                    logger.error(f"SIMULATED: NETCONF failed for {target_device_ip}: {e}")
                                    return {"status": "ERROR", "error_message": f"NETCONF application failed for {target_device_ip}: {e}"}
                    else:
                        log_msg = f"Invalid parameter {param_name} for cell {cell_id} based on CELL_CAPABILITIES. Skipping."
                        logger.warning(log_msg)
                        netconf_application_log.append(log_msg)

    # Simulate outcomes based on attempt count (as in your original code)
    # This outcome generation should ideally be based on actual network feedback after applying changes.
    # For a closed-loop system, there would be a separate monitoring phase that generates these.
    simulated_outcomes = []
    # Example: Assuming intent targets 'RANEnergyEfficiency'
    # This is a placeholder. Real outcomes come from network monitoring post-change.
    # The OrchestrationService might just report "applied successfully"
    # and a later "Monitoring/Evaluation" service would determine actual outcomes.

    # For now, let's keep the outcome simulation here as per your original logic
    # to simplify the flow, but acknowledge it's not ideal for true closed-loop.
    if current_attempt_count > 1: # Adjusted from >2 to >1 for quicker "success" in demo
        simulated_outcomes = [
            {"targetName": "RANEnergyEfficiency", "targetAchievedValue": 10, "targetUnit": "percentage"}
        ]
    else:
        simulated_outcomes = [
            {"targetName": "RANEnergyEfficiency", "targetAchievedValue": 5, "targetUnit": "percentage"}
        ]

    logger.info(f"Orchestration complete for intent {intent_id}. Patches: {applied_patches}, Simulated Outcomes: {simulated_outcomes}")
    return {
        "status": "SUCCESS", 
        "data": {
            "applied_patches": applied_patches, 
            "simulated_outcomes": simulated_outcomes, # Ideally, outcomes are from a separate monitoring step
            "application_log": netconf_application_log
        }
    }