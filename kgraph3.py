from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.plugins.sparql import prepareQuery # prepareQuery is useful but not strictly needed for this fix

# Define namespaces
ex = Namespace("http://example.org/ran-ontology#")
rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
owl = Namespace("http://www.w3.org/2002/07/owl#")

# Create a Graph
g = Graph()

# Bind namespaces (optional for SPARQL queries, mainly for serialization/URIRef creation)
g.bind("ex", ex)
g.bind("rdf", rdf)
g.bind("rdfs", rdfs)
g.bind("owl", owl)

# --- Add the RDF data (using the Turtle string directly) ---
turtle_data = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ex: <http://example.org/ran-ontology#> .

# Ontology Definition (etc. - the rest of your ontology definition and instance data goes here)
ex:ranOntology rdf:type owl:Ontology ;
    rdfs:comment "Ontology describing 5G RAN parameters and their effects on performance metrics." .

# Class Definitions
ex:RanParameter rdf:type owl:Class ;
    rdfs:label "RAN Parameter" ;
    rdfs:comment "A configuration parameter in a 5G Radio Access Network." .

ex:PerformanceMetric rdf:type owl:Class ;
    rdfs:label "Performance Metric" ;
    rdfs:comment "A measurable performance indicator or resource consumption in a 5G RAN." .

# Property Definition
ex:hasEffectOn rdf:type owl:ObjectProperty ;
    rdfs:label "has effect on" ;
    rdfs:comment "Indicates that a RAN parameter has an effect on a performance metric." ;
    rdfs:domain ex:RanParameter ;
    rdfs:range ex:PerformanceMetric .

ex:isAffectedBy rdf:type owl:ObjectProperty ;
    rdfs:label "is affected by" ;
    rdfs:comment "Indicates that a performance metric is affected by a RAN parameter." ;
    owl:inverseOf ex:hasEffectOn ;
    rdfs:domain ex:PerformanceMetric ;
    rdfs:range ex:RanParameter .

# Instance Definitions and Relationships

# Parameters
ex:txPower rdf:type ex:RanParameter ;
    rdfs:label "TxPower" ;
    rdfs:comment "The power level at which a cell transmits signals." .

ex:bandwidth rdf:type ex:RanParameter ;
    rdfs:label "Bandwidth" ;
    rdfs:comment "The width of the frequency band used for communication." .

ex:carrierFrequency rdf:type ex:RanParameter ;
    rdfs:label "Carrier Frequency" ;
    rdfs:comment "The central frequency of the radio channel." .

# Performance Metrics
ex:powerConsumption rdf:type ex:PerformanceMetric ;
    rdfs:label "Power Consumption" ;
    rdfs:comment "The amount of electrical power consumed by the RAN node." .

ex:coverageArea rdf:type ex:PerformanceMetric ;
    rdfs:label "Coverage Area" ;
    rdfs:comment "The geographical area where a cell provides sufficient signal strength." .

ex:throughput rdf:type ex:PerformanceMetric ;
    rdfs:label "Throughput" ;
    rdfs:comment "The data transfer rate achievable by users." .

# Specific Relationships (Effects)
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
    "owl": owl # Include owl just in case, though not strictly needed for these queries
}

def get_affected_metrics(graph, parameter_local_name)->dict[str]:
    """
    Queries the graph to find performance metrics affected by a given RAN parameter.

    Args:
        graph (rdflib.Graph): The RDF graph containing the data.
        parameter_local_name (str): The local name of the RAN parameter (e.g., "txPower").

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              'metric_uri' and 'metric_label' of an affected metric.
              Returns an empty list if the parameter is not found or has no effects.
    """
    # Construct the full URI for the parameter instance
    # Assumes the local name corresponds directly to the instance URI in the ex namespace
    param_uri = ex[parameter_local_name]

    # Check if the parameter URI actually exists in the graph
    # This prevents querying for non-existent resources
    if (param_uri, None, None) not in graph:
        print(f"Warning: Parameter '{parameter_local_name}' ({param_uri}) not found in graph.")
        return []


    # Define the SPARQL query string using a variable for the parameter subject
    # We use '?param' as the variable name in the query
    sparql_query = """
    SELECT ?effectedMetric ?metricLabel
    WHERE {
      ?param ex:hasEffectOn ?effectedMetric . # Use the variable ?param here
      ?effectedMetric rdfs:label ?metricLabel .
    }
    """

    # Define the initial bindings for the SPARQL query
    # This maps the SPARQL variable '?param' to the specific parameter URI we constructed
    initial_bindings = {
        "param": param_uri
    }

    results_list = []

    # Execute the query, passing both namespace bindings and initial bindings
    results = graph.query(
        sparql_query,
        initNs=query_namespaces,     # For resolving prefixes like ex: and rdfs: in the query string
        initBindings=initial_bindings # For setting the value of the ?param variable
    )

    # Process and store the results
    for row in results:
        results_list.append({
            "metric_uri": str(row.effectedMetric), # Convert URIRef to string
            "metric_label": str(row.metricLabel)   # Convert Literal to string
        })

    return results_list

def get_parameter_effects_for_llm(graph, parameter_local_name)->str:
    """
    Queries the graph to find performance metrics affected by a given RAN parameter,
    and formats the relationship details using labels for LLM consumption.

    Args:
        graph (rdflib.Graph): The RDF graph containing the data.
        parameter_local_name (str): The local name of the RAN parameter (e.g., "txPower").

    Returns:
        list: A list of dictionaries, where each dictionary contains
              'param_label', 'predicate_label', and 'metric_label' for
              each relationship found. Returns an empty list if the parameter
              is not found or has no matching relationships.
    """
    param_uri = ex[parameter_local_name]

    if (param_uri, None, None) not in graph:
        print(f"Warning: Parameter '{parameter_local_name}' ({param_uri}) not found in graph.")
        return ""

    sparql_query = """
    SELECT ?paramLabel ?predicateLabel ?metricLabel
    WHERE {
      BIND(?inputParam AS ?param) .

      ?param ex:hasEffectOn ?effectedMetric .

      ?param rdfs:label ?paramLabel .
      ex:hasEffectOn rdfs:label ?predicateLabel .
      ?effectedMetric rdfs:label ?metricLabel .
    }
    """

    initial_bindings = {
        "inputParam": param_uri
    }

    results_list = []

    results = graph.query(
        sparql_query,
        initNs=query_namespaces,
        initBindings=initial_bindings
    )

    for row in results:
        results_list.append({
            "param_label": str(row.paramLabel),
            "predicate_label": str(row.predicateLabel),
            "metric_label": str(row.metricLabel)
        })

    if results_list:
        print(f"Facts about '{parameter_name}' effects (formatted for LLM):")
        param_label = ""
        predicate_label = ""
        metric_labels = []
        for fact in results_list:
            param_label = fact['param_label']
            predicate_label = fact['predicate_label']
            metric_labels.append(fact['metric_label'])
            # Use the keys that are actually returned by the function
        aaa = f"- Parameter \"{param_label}\" {predicate_label} Metric \"{"\", \"".join(metric_labels)}\"."
        return aaa
        # print(f"- Parameter \"{fact['param_label']}\" {fact['predicate_label']} Metric \"{fact['metric_labels']}\".")
    else:
        return f"No direct '{ex.hasEffectOn}' effects found for '{parameter_name}' or parameter not in graph."
        # print(f"No direct '{ex.hasEffectOn}' effects found for '{parameter_name}' or parameter not in graph.")


    # return results_list

# # --- Example Usage ---

# print("Querying effects for 'txPower':")
# effects_txpower = get_affected_metrics(g, "txPower")
# if effects_txpower:
#     print(f"Parameter 'txPower' ({ex.txPower}) affects the following metrics:")
#     for effect in effects_txpower:
#         print(f"  - '{effect['metric_label']}' ({effect['metric_uri']})")
# else:
#     print("No effects found for 'txPower' or parameter not in graph.")

# print("-" * 20)

# print("Querying effects for 'bandwidth':")
# effects_bandwidth = get_affected_metrics(g, "bandwidth")
# if effects_bandwidth:
#     print(f"Parameter 'bandwidth' ({ex.bandwidth}) affects the following metrics:")
#     for effect in effects_bandwidth:
#         print(f"  - '{effect['metric_label']}' ({effect['metric_uri']})")
# else:
#      print("No effects found for 'bandwidth' or parameter not in graph.")

# print("-" * 20)

# print("Querying effects for 'carrierFrequency':")
# effects_carrier_frequency = get_affected_metrics(g, "carrierFrequency")
# if effects_carrier_frequency:
#     print(f"Parameter 'carrierFrequency' ({ex.carrierFrequency}) affects the following metrics:")
#     for effect in effects_carrier_frequency:
#         print(f"  - '{effect['metric_label']}' ({effect['metric_uri']})")
# else:
#     print("No effects found for 'carrierFrequency' or parameter not in graph.")

# print("-" * 20)

# print("Querying effects for 'nonExistentParameter':")
# effects_nonexistent = get_affected_metrics(g, "nonExistentParameter")
# if effects_nonexistent:
#     # This block shouldn't be reached for 'nonExistentParameter' due to the check
#     print(f"Parameter 'nonExistentParameter' affects the following metrics:")
#     for effect in effects_nonexistent:
#         print(f"  - '{effect['metric_label']}' ({effect['metric_uri']})")
# else:
#     print("No effects found for 'nonExistentParameter' or parameter not in graph.")

# # --- Example of using SPARQL to summarize relationships ---

# # Query 1: Find all Parameters and the Metrics they affect
# print("Summarizing: Which Parameters affect which Metrics?")
# sparql_query_param_effects = """
# SELECT ?paramLabel ?metricLabel
# WHERE {
#   ?param rdf:type ex:RanParameter .
#   ?param rdfs:label ?paramLabel .
#   ?param ex:hasEffectOn ?metric .
#   ?metric rdfs:label ?metricLabel .
# }
# ORDER BY ?paramLabel ?metricLabel
# """ # No need for prepareQuery here, can pass directly

# # Process the query results to group by parameter
# param_effects_summary = {}
# for row in g.query(sparql_query_param_effects, initNs=query_namespaces): # Pass the namespaces here!
#     param_label = str(row.paramLabel)
#     metric_label = str(row.metricLabel)
#     if param_label not in param_effects_summary:
#         param_effects_summary[param_label] = []
#     param_effects_summary[param_label].append(metric_label)

# # Print the summary
# for param, metrics in param_effects_summary.items():
#     print(f"Parameter '{param}' affects the following metrics:")
#     for metric in metrics:
#         print(f"  - '{metric}'")
# print("-" * 20)


# # Query 2: Find all Metrics and the Parameters that affect them
# print("Summarizing: Which Metrics are affected by which Parameters?")
# sparql_query_metric_affected = """
# SELECT ?metricLabel ?paramLabel
# WHERE {
#   ?metric rdf:type ex:PerformanceMetric .
#   ?metric rdfs:label ?metricLabel .
#   ?metric ex:isAffectedBy ?param . # Using the inverse property
#   ?param rdfs:label ?paramLabel .
# }
# ORDER BY ?metricLabel ?paramLabel
# """ # No need for prepareQuery here either

# # Process the query results to group by metric
# metric_affected_summary = {}
# for row in g.query(sparql_query_metric_affected, initNs=query_namespaces): # Pass the namespaces here!
#     metric_label = str(row.metricLabel)
#     param_label = str(row.paramLabel)
#     if metric_label not in metric_affected_summary:
#         metric_affected_summary[metric_label] = []
#     metric_affected_summary[metric_label].append(param_label)

# # Print the summary
# for metric, parameters in metric_affected_summary.items():
#     print(f"Metric '{metric}' is affected by the following parameters:")
#     for param in parameters:
#         print(f"  - '{param}'")
# print("-" * 20)


# # Example of a simple triple query (This doesn't use SPARQL string, so it works directly with URIRefs)
# print("\nExample: Finding all effects of Transmit Power using triple patterns")
# for s, p, o in g.triples((ex.txPower, ex.hasEffectOn, None)):
#      # Retrieve the labels for better readability
#      param_label = g.value(subject=s, predicate=rdfs.label)
#      metric_label = g.value(subject=o, predicate=rdfs.label)
#      # Retrieve the label for the predicate too
#      predicate_label = g.value(subject=p, predicate=rdfs.label)
#      print(f"'{param_label}' ({s}) '{predicate_label}' ('{metric_label}' ({o}))")



# # --- Simple SPARQL query to find all effects of Transmit Power ---
# print("\nExample: Finding all effects of Transmit Power using SPARQL")

# # Define the SPARQL query string
# # We want to SELECT the effected metric (?effectedMetric) and its label (?metricLabel)
# # WHERE the triple pattern is ex:txPower hasEffectOn some ?effectedMetric
# # AND the ?effectedMetric has rdfs:label ?metricLabel
# sparql_query_txpower_effects = """
# SELECT ?effectedMetric ?metricLabel
# WHERE {
#   ex:be ex:hasEffectOn ?effectedMetric .
#   ?effectedMetric rdfs:label ?metricLabel .
# }
# """

# # Execute the query, passing the namespace bindings
# results = g.query(sparql_query_txpower_effects, initNs=query_namespaces)

# # Process and print the results
# # Each result row will contain the URI of the metric and its label
# for row in results:
#      # row.effectedMetric will be the URIRef (e.g., ex:powerConsumption)
#      # row.metricLabel will be the Literal (e.g., "Power Consumption")
#      print(f"'{g.value(ex.be, rdfs.label)}' ({ex.be}) has effect on '{row.metricLabel}' ({row.effectedMetric})")

# print("-" * 20)

parameter_name = "txPower"
print(f"Querying effects for '{parameter_name}':")
effects_data = get_parameter_effects_for_llm(g, parameter_name)

# # 2. Format the results into natural language sentences for an LLM prompt
# if effects_data:
#     print(f"Facts about '{parameter_name}' effects (formatted for LLM):")
#     for fact in effects_data:
#         # Use the keys that are actually returned by the function
#         print(f"- Parameter \"{fact['param_label']}\" {fact['predicate_label']} Metric \"{fact['metric_label']}\".")
# else:
#     print(f"No direct '{ex.hasEffectOn}' effects found for '{parameter_name}' or parameter not in graph.")

# print("-" * 20)


print(effects_data)