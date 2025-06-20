import json
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import XSD # Import XSD namespace for datatypes

# Define namespaces
ex = Namespace("http://example.org/ran-ontology#")
rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
owl = Namespace("http://www.w3.org/2002/07/owl#")
xsd = XSD

# Create a Graph
g = Graph()

# Bind namespaces
g.bind("ex", ex)
g.bind("rdf", rdf)
g.bind("rdfs", rdfs)
g.bind("owl", owl)
g.bind("xsd", xsd)

# --- Add the RDF data (using the Turtle string) ---
# Including the Intent and the link to PowerConsumption and Throughput
turtle_data = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <http://example.org/ran-ontology#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Ontology Definition
ex:ranOntology rdf:type owl:Ontology ;
    rdfs:comment "Ontology describing 5G RAN parameters, performance metrics, network intents, and their relationships." .

# Class Definitions
ex:RanParameter rdf:type owl:Class ;
    rdfs:label "RAN Parameter" ;
    rdfs:comment "A configuration parameter in a 5G Radio Access Network." .

ex:PerformanceMetric rdf:type owl:Class ;
    rdfs:label "Performance Metric" ;
    rdfs:comment "A measurable performance indicator or resource consumption in a 5G RAN." .

ex:NetworkIntent rdf:type owl:Class ;
    rdfs:label "Network Intent" ;
    rdfs:comment "A high-level goal or objective for network performance or state." .

# Property Definitions
ex:hasEffectOn rdf:type owl:ObjectProperty ;
    rdfs:label "affects" ; # Using a more LLM-friendly label
    rdfs:comment "Indicates that a RAN parameter influences a performance metric." ;
    rdfs:domain ex:RanParameter ;
    rdfs:range ex:PerformanceMetric .

ex:isAffectedBy rdf:type owl:ObjectProperty ;
    rdfs:label "is affected by" ;
    rdfs:comment "Indicates that a performance metric is influenced by a RAN parameter." ;
    owl:inverseOf ex:hasEffectOn ;
    rdfs:domain ex:PerformanceMetric ;
    rdfs:range ex:RanParameter .

ex:isBasedOnMetric rdf:type owl:ObjectProperty ;
    rdfs:label "is based on metric" ; # Label for LLM
    rdfs:comment "Indicates that a Network Intent or KPI is measured or assessed based on a specific Performance Metric." ;
    rdfs:domain ex:NetworkIntent ;
    rdfs:range ex:PerformanceMetric .

ex:hasUnit rdf:type owl:DatatypeProperty ;
    rdfs:label "has unit" ;
    rdfs:comment "The unit of measurement for a parameter or metric." ;
    rdfs:domain [ rdf:type owl:Class ; owl:unionOf ( ex:RanParameter ex:PerformanceMetric ) ] ;
    rdfs:range xsd:string .

# Instance Definitions and Relationships

# Parameters
ex:txPower rdf:type ex:RanParameter ;
    rdfs:label "Transmit Power" ;
    rdfs:comment "The power level at which a cell transmits radio signals. Higher power generally increases coverage but also power consumption." ;
    ex:hasUnit "dBm"^^xsd:string .

ex:bandwidth rdf:type ex:RanParameter ;
    rdfs:label "Bandwidth" ;
    rdfs:comment "The width of the frequency band allocated for communication. Wider bands allow for higher data rates (throughput)." ;
    ex:hasUnit "MHz"^^xsd:string .

ex:carrierFrequency rdf:type ex:RanParameter ;
    rdfs:label "Carrier Frequency" ;
    rdfs:comment "The central frequency used for transmission. Affects propagation characteristics and potential coverage area." ;
    ex:hasUnit "GHz"^^xsd:string .

# Performance Metrics
ex:powerConsumption rdf:type ex:PerformanceMetric ;
    rdfs:label "Power Consumption" ;
    rdfs:comment "The electrical power used by the RAN node (e.g., base station). A key aspect for energy efficiency." ;
    ex:hasUnit "Watt"^^xsd:string .

ex:coverageArea rdf:type ex:PerformanceMetric ;
    rdfs:label "Coverage Area" ;
    rdfs:comment "The geographical region where mobile devices can successfully connect and communicate with the RAN cell." ;
    ex:hasUnit "km²"^^xsd:string .

ex:throughput rdf:type ex:PerformanceMetric ;
    rdfs:label "Throughput" ;
    rdfs:comment "The average or peak data rate achieved by users or the cell. A primary indicator of network capacity and user experience." ;
    ex:hasUnit "Mbps"^^xsd:string .

# Intent Instance linking to metrics
ex:RANEnergyEfficiency_Intent rdf:type ex:NetworkIntent ;
    rdfs:label "RAN Energy Efficiency" ;
    rdfs:comment "Goal to reduce energy consumption in the Radio Access Network while maintaining performance." ;
    ex:isBasedOnMetric ex:powerConsumption ;
    ex:isBasedOnMetric ex:throughput .

# Parameter-Metric Relationships
ex:txPower ex:hasEffectOn ex:powerConsumption .
ex:txPower ex:hasEffectOn ex:coverageArea .
ex:txPower ex:hasEffectOn ex:throughput .

ex:bandwidth ex:hasEffectOn ex:throughput .

ex:carrierFrequency ex:hasEffectOn ex:coverageArea .
"""

# Parse the Turtle data into the graph
g.parse(data=turtle_data, format='turtle')

print(f"Graph contains {len(g)} triples.")
print("-" * 20)

# --- Define namespace bindings for SPARQL queries ---
query_namespaces = {
    "ex": ex,
    "rdf": rdf,
    "rdfs": rdfs,
    "owl": owl,
    "xsd": xsd
}

# --- Input JSON (matching your example structure) ---
input_json = """
{
   "expectationTargets": [
       {
           "targetName": "RAN Energy Efficiency",
           "targetCondition": "IS_GREATER_THAN",
           "targetValueRange": "10",
           "targetUnit": "percentage"
       },
       {
           "targetName": "Throughput",
           "targetCondition": "IS_GREATER_THAN",
           "targetValueRange": "100",
           "targetUnit": "Mbps"
       }
   ]
}
"""

# --- Intermediate Processing to Build Structured JSON ---

# 1. JSON Parser
parsed_data = json.loads(input_json)
targets_from_json = parsed_data.get("expectationTargets", [])

# Structure for the final output JSON
output_json_structure = {
  "request_type": "explain_parameter_adjustment_strategy",
  "context": {
    "user_goals": [],
    "goal_dependencies": [], # Intent -> Metrics
    "parameter_effects": [] # Parameter -> Metrics
  },
  "output_format_preference": "natural_language_explanation"
}

# Set of all unique relevant metric URIs identified through targets/intents
relevant_metric_uris = set()

# Dictionary to map Intent URIs to lists of their dependent metric URIs
intent_to_metrics_map = {}

# Process JSON Targets
print("--- Processing JSON Targets ---")
for target in targets_from_json:
    target_name = target.get("targetName")
    condition = target.get("targetCondition")
    value_str = target.get("targetValueRange")
    unit = target.get("targetUnit")

    if not target_name:
        print(f"Warning: Skipping target with no name: {target}")
        continue

    # Map target name to a URI and determine its type (Intent or Metric)
    # Using SPARQL query on rdfs:label
    sparql_map_query = """
    SELECT ?uri ?type
    WHERE {
      ?uri rdfs:label ?label .
      FILTER(LCASE(STR(?label)) = LCASE(?targetNameString))
      ?uri rdf:type ?type .
      FILTER(?type IN (ex:NetworkIntent, ex:PerformanceMetric)) # Only consider Intents or Metrics
    }
    LIMIT 1
    """
    initial_map_bindings = {"targetNameString": Literal(target_name)}

    mapped_uri = None
    mapped_type = None
    map_results = g.query(sparql_map_query, initNs=query_namespaces, initBindings=initial_map_bindings)
    for row in map_results:
        mapped_uri = row.uri
        mapped_type = row.type
        break # Found a match, take the first one

    if mapped_uri:
        # Add to user_goals in the output structure
        try:
            # Attempt to convert value to number if possible, keep as string otherwise
            value_parsed = float(value_str) if value_str else None
            if value_parsed is not None and value_parsed.is_integer():
                 value_parsed = int(value_parsed)
        except ValueError:
            value_parsed = value_str # Keep as string if conversion fails

        output_json_structure["context"]["user_goals"].append({
            "name": target_name,
            "condition": condition,
            "value": value_parsed,
            "unit": unit
        })

        # Identify relevant metrics based on the mapped target
        if mapped_type == ex.NetworkIntent:
            # If it's an Intent, find the metrics it's based on
            intent_label = g.value(mapped_uri, rdfs.label) or target_name
            dependent_metrics = [] # Store labels for JSON output
            dependent_metric_uris_for_intent = [] # Store URIs to add to relevant_metric_uris

            sparql_intent_metrics = """
            SELECT ?metric ?metricLabel
            WHERE {
              BIND(?inputIntent AS ?intent) .
              ?intent ex:isBasedOnMetric ?metric .
              ?metric rdfs:label ?metricLabel .
              # Optional: Ensure ?metric is a PerformanceMetric
              ?metric rdf:type ex:PerformanceMetric .
            }
            """
            initial_intent_bindings = {"inputIntent": mapped_uri}
            intent_metrics_results = g.query(
                sparql_intent_metrics,
                initNs=query_namespaces,
                initBindings=initial_intent_bindings
            )

            for row in intent_metrics_results:
                dependent_metrics.append(str(row.metricLabel))
                dependent_metric_uris_for_intent.append(row.metric)

            # Add to goal_dependencies in the output structure
            if dependent_metrics:
                output_json_structure["context"]["goal_dependencies"].append({
                    "goal_name": intent_label,
                    "depends_on_metrics": dependent_metrics
                })
            relevant_metric_uris.update(dependent_metric_uris_for_intent) # Add URIs to the set


        elif mapped_type == ex.PerformanceMetric:
            # If it's directly a Metric, add its URI to the relevant set
             relevant_metric_uris.add(mapped_uri)
             print(f"Mapped target '{target_name}' directly to metric: {mapped_uri}")

        else:
             # Should not happen with the FILTER in sparql_map_query, but good practice
             print(f"Warning: Mapped URI '{mapped_uri}' has unhandled type: {mapped_type}")


    else:
        print(f"Warning: Could not map target name '{target_name}' to a known Intent or Metric in the graph.")

print("\n--- Identified Relevant Metric URIs ---")
print([str(uri) for uri in relevant_metric_uris])
print("-" * 20)

# 4. Parameter Effects Identification for Relevant Metrics
if relevant_metric_uris:
    # Build VALUES clause for the SPARQL query
    values_clause = "VALUES ?metric { " + " ".join([f"<{uri}>" for uri in relevant_metric_uris]) + " }"

    # Query for parameters that affect these relevant metrics
    sparql_param_effects = f"""
    SELECT DISTINCT ?param ?paramLabel ?paramUnit ?metric ?metricLabel ?metricUnit ?predicate ?predicateLabel
    WHERE {{
      {values_clause}
      ?param ex:hasEffectOn ?metric . # Find parameters affecting these metrics

      # Get details for the parameter
      ?param rdf:type ex:RanParameter . # Ensure it's a parameter
      ?param rdfs:label ?paramLabel .
      OPTIONAL {{ ?param ex:hasUnit ?paramUnit . }} # Unit might be missing

      # Get details for the relationship predicate
      # We are only querying for ex:hasEffectOn directly here
      BIND(ex:hasEffectOn AS ?predicate) .
      ?predicate rdfs:label ?predicateLabel . # Get label "affects"

      # Get details for the metric (already in VALUES clause)
      ?metric rdfs:label ?metricLabel .
      OPTIONAL {{ ?metric ex:hasUnit ?metricUnit . }} # Unit might be missing
    }}
    """

    param_effect_results = g.query(sparql_param_effects, initNs=query_namespaces)

    # Process and add parameter effects to the output structure
    print("\n--- Identifying Parameter Effects ---")
    for row in param_effect_results:
        # Determine correlation. With just ex:hasEffectOn, we can't know.
        # For this example, based on common knowledge of TxPower/Bandwidth effects, we can infer.
        # A real system would need explicit properties (hasPositiveEffectOn, etc.) or rules.
        correlation = "unknown" # Default if we can't determine

        # Simple inference based on common RAN behavior for demonstration
        param_uri_str = str(row.param)
        metric_uri_str = str(row.metric)

        if param_uri_str == str(ex.txPower) and metric_uri_str in [str(ex.powerConsumption), str(ex.coverageArea), str(ex.throughput)]:
             correlation = "positive" # Higher TxPower -> higher power, coverage, throughput (up to a point)
        elif param_uri_str == str(ex.bandwidth) and metric_uri_str == str(ex.throughput):
             correlation = "positive" # Higher Bandwidth -> higher throughput
        # Add more inference rules for other parameters/metrics if needed

        output_json_structure["context"]["parameter_effects"].append({
            "parameter_name": str(row.paramLabel) if row.paramLabel else str(row.param), # Use label, fallback to URI
            "parameter_unit": str(row.paramUnit) if row.paramUnit else None,
            "metric": str(row.metricLabel) if row.metricLabel else str(row.metric), # Use label, fallback to URI
            "metric_unit": str(row.metricUnit) if row.metricUnit else None,
            "correlation": correlation # Add inferred or determined correlation
        })

    print(f"Found {len(output_json_structure['context']['parameter_effects'])} parameter effects.")
else:
    print("No relevant metrics identified, skipping parameter effects query.")


# --- Final Output ---

# Serialize the Python dictionary to a JSON string
final_json_output = json.dumps(output_json_structure, indent=2)

print("\n--- Generated Structured JSON Output ---")
print(final_json_output)
print("--------------------------------------")