from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import json
from datetime import datetime
from uuid import uuid4
from enum import Enum
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from src.netconf_cli import NETCONFCLIENT

import re
import logging

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

class AgentCallRequest(BaseModel):
    agent_name: str
    capability_needed: str # e.g., "energy_saving_strategy", "history_analysis"
    payload: Dict[str, Any] # Data to send to the agent

class AgentCallResponse(BaseModel):
    agent_name: str
    status: str # "SUCCESS", "ERROR"
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class IntentState(BaseModel):
    intent: Dict
    pm_data: Dict = {}
    
    # Store raw outputs from "external" agents
    history_analysis_result: Optional[Dict] = None # Output from HistoryAnalysisService
    proposed_strategies_by_agent: Dict[str, List[Dict]] = Field(default_factory=dict) # e.g., {"EnergySavingService": [...]}
    
    # Final, resolved strategies after conflict resolution
    resolved_strategies: List[Dict] = Field(default_factory=list)
    
    outcomes: List[Dict] = Field(default_factory=list)
    report: Dict = Field(default_factory=dict)
    fulfilmentStatus: str = "NONE"
    attempt_count: int = 0
    max_attempts: int = 3 # Define max attempts here
    
    # For managing calls to external agents in the workflow
    pending_agent_calls: List[AgentCallRequest] = Field(default_factory=list)
    agent_call_responses: List[AgentCallResponse] = Field(default_factory=list)

    # Patches and application status
    patches: Dict = {}
    applied: bool = False

    # For routing decisions
    current_stage: str = "INITIALIZATION" # e.g., INITIALIZATION, DATA_COLLECTION, STRATEGY_GENERATION, ...
    error_flag: bool = False
    error_details: Optional[str] = None

    # User approval status
    user_approved_strategies: Optional[bool] = None