from typing import Dict, List
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# agent_registry.py
AGENT_REGISTRY = {}

def register_agent(agent_name: str, capabilities: List[str], endpoint_url: str, api_contract: Dict = None):
    """
    Registers a specialized agent with the core system.
    api_contract could define expected input/output schemas.
    """
    AGENT_REGISTRY[agent_name] = {
        "capabilities": capabilities,
        "endpoint_url": endpoint_url, # URL for the external agent service
        "api_contract": api_contract or {}
    }
    logger.info(f"Agent '{agent_name}' registered with capabilities: {capabilities} at {endpoint_url}")

def get_agents_for_capability(capability: str) -> List[Dict]:
    """Finds agents that can fulfill a given capability."""
    return [
        {"name": name, **details}
        for name, details in AGENT_REGISTRY.items()
        if capability in details["capabilities"]
    ]

# Example registrations (done at startup of the core system)
def initialize_agent_registry():
    # For now, endpoint_url is conceptual. In a real system, these are actual URLs.
    # We'll simulate calls by calling local functions.
    register_agent(
        agent_name="HistoryAnalysisService",
        capabilities=["history_analysis"],
        endpoint_url="local/history_analysis_service" 
    )
    register_agent(
        agent_name="EnergySavingStrategyService",
        capabilities=["strategy_generation:energy_saving"],
        endpoint_url="local/energy_saving_strategy_service"
    )
    # Example for another strategy agent
    # register_agent(
    #     agent_name="ThroughputStrategyService",
    #     capabilities=["strategy_generation:throughput"],
    #     endpoint_url="local/throughput_strategy_service"
    # )
    register_agent(
        agent_name="ConflictResolutionService",
        capabilities=["conflict_resolution"],
        endpoint_url="local/conflict_resolution_service"
    )
    register_agent(
        agent_name="OrchestrationService", # Your netconf_cli interaction
        capabilities=["network_configuration"],
        endpoint_url="local/orchestration_service"
    )

# Call this when your main application starts
# initialize_agent_registry()