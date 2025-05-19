# Intent-Based System (Project Handover)

## System Architecture
![image](/architecture/v2.png)
## 1. Introduction & System Goal
This document provides a high-level overview of the Intent-Based System (IBS) project. The primary goal of this system is to allow users to specify what they want to achieve (their "intent") in a high-level, declarative manner, and have the system autonomously figure out how to achieve it.
The system is designed to:
Receive user intents.
Understand the intent and map it to available system capabilities.
Orchestrate one or more "Specialized Agents" to propose and then execute strategies to fulfill the intent.
Monitor and evaluate the outcome of the executed strategies.
Provide a report on the intent's fulfillment status.
This approach abstracts away the complex, low-level configurations and procedures typically required, making the system more agile, adaptable, and user-friendly.
### Core Concepts
- Intent: A high-level declaration of a desired outcome or state. It describes the "what" not the "how."
Example of intent:
```
{
    "goal": "increase ran power efficiency"
    "expectation": "ran efficiency can be > 10% but throughput remain stable"
    "constrain": "geolocation around Taipei"
}
```
- Structure: Defined by Pydantic models (e.g., IntentDetail, IntentExpectation, ExpectationTarget), specifying things like the desired object, target conditions, values, and units.
Example: "Reduce energy consumption in building X by 20% within the next month."
Core Intent Handler (Main Agent): The central brain of the system.
- Responsibility: Receives intents, manages their lifecycle, orchestrates specialized agents, and maintains the overall state of intent processing.
Implementation: A FastAPI application (core_intent_handler_fastapi_app.py) that uses LangGraph for workflow management.
- Specialized Agents (or Strategy Agents/Orchestration Agents): External, independent services or modules that possess specific domain knowledge and capabilities.

```
1. System Architecture & High-Level Flow
+-------------+       (1) Submit Intent        +-----------------------+
|    User     | -----------------------------> | Core Intent Handler   |
+-------------+       (JSON via API)           | (FastAPI + LangGraph) |
                                               +-----------------------+
                                                        | ^
                                                        | | (2) Register Agent
                                                        | | (Capabilities, Endpoint)
                                                        v |
                                               +-----------------------+
                                               | Agent Registry        |
                                               +-----------------------+
                                                        |
     (3) Decompose Intent, Select Capability (LLM/Logic) |
                                                        |
     (4) Request Strategy Generation / Orchestration    v
   +-------------------------------------------------------------------+
   |                                                                   |
   v                                                                   ^
+-----------------------+       (API Call)       +-----------------------+
| Specialized Agent 1   | <---------------------> | Specialized Agent 2   |
| (e.g., Energy Strategy)|                        | (e.g., Network Orchestrator)|
+-----------------------+                        +-----------------------+
   ^ (5) Propose/Execute                                  |
   |    Strategy                                          |
   +-------------------------------------------------------+
                         (6) Update State, Report Back to Core Handler
                                                        |
                                                        v
                                               +-----------------------+
                                               | Core Intent Handler   |
                                               | (Updates IntentState,  |
                                               |  Evaluates)           |
                                               +-----------------------+
                                                        |
                                                        v (7) Final Response
+-------------+                                  +-----------------------+
|    User     | <------------------------------ | Intent Result         |
+-------------+       (JSON via API)           +-----------------------+
```



### Areas for Further Exploration & Development
- Robustness of Agent Communication: Implement resilient communication (retries, circuit breakers) between the Core Handler and Specialized Agents.
Advanced Strategy Selection/Ranking: If multiple strategies are proposed, how does the system choose the best one? (Could involve cost, impact, user preference).
- Complex Intent Decomposition: Handling intents that require multiple capabilities or a sequence of sub-intents.
- Enhanced Evaluation: More sophisticated methods for evaluating intent fulfillment, potentially involving feedback loops and learning.
Monitoring & Observability: Detailed logging, tracing, and metrics for system performance and intent processing.
- Human-in-the-Loop: More sophisticated integration of user approval or intervention points (e.g., for USER_APPROVAL node).
- Scalability: Ensuring the Core Handler and Specialized Agents can handle a large number of concurrent intents.
- Security: Authentication and authorization for API endpoints.
- Dynamic Capability Discovery: More advanced mechanisms beyond simple registration.
