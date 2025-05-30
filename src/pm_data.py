import xml.etree.ElementTree as ET
import json
import os
import re
import datetime


def get_latest_xml_file_by_time(directory_path, filename_pattern=r'B?(\d{8})\.(\d{4})\+\d{4}-\d{4}\+\d{4}_Cell_\d+\.xml'):
    """
    Scans a directory for XML files matching a specific timestamp pattern in their filename,
    and returns the path to the file with the latest timestamp.

    Args:
        directory_path (str): The path to the directory to scan.
        filename_pattern (str): A regex pattern to match the filenames and capture
                                the date (group 1) and start hour/minute (group 2).
                                Default pattern matches BYYYYMMDD.HHMM+ZZZZ-HHMM+ZZZZ_Cell_ID.xml
                                and handles the optional 'B' prefix.

    Returns:
        str or None: The full path to the latest XML file, or None if no matching files are found.
    """
    latest_file = None
    latest_timestamp = None


    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return None

    # Compile the regex pattern for efficiency
    compiled_pattern = re.compile(filename_pattern)

    for filename in os.listdir(directory_path):
        match = compiled_pattern.match(filename)
        if match:
            date_str = match.group(1) # YYYYMMDD
            time_hhmm_str = match.group(2) # HHMM (start time of the interval)

            # Construct a datetime string (assuming +0000 offset for filename implies UTC for this purpose)
            # We'll use the start time of the interval for comparison.
            # Example: 20250522.0700 -> 2025-05-22 07:00:00
            datetime_str = f"{date_str}{time_hhmm_str}"

            try:
                # Parse the datetime string. We don't need timezone info here if just comparing.
                current_timestamp = datetime.datetime.strptime(datetime_str, '%Y%m%d%H%M')

                if latest_timestamp is None or current_timestamp > latest_timestamp:
                    latest_timestamp = current_timestamp
                    latest_file = os.path.join(directory_path, filename)
            except ValueError:
                # Handle cases where the date/time part of the filename is malformed
                print(f"Warning: Could not parse timestamp from filename: {filename}")
                continue
    
    return latest_file

def parse_measurement_data(xml_file_path, filter_gnb_du=None, filter_nrcelldu=None)->str:
    """
    Opens and reads an XML measurement data file, filters data based on
    GnbDuFunction and NrCellDu, and returns the parsed data as JSON.

    Args:
        xml_file_path (str): The path to the XML measurement data file.
        filter_gnb_du (int, optional): The GnbDuFunction ID to filter by.
                                       If None, no filter is applied.
        filter_nrcelldu (int, optional): The NrCellDu ID to filter by.
                                         If None, no filter is applied.

    Returns:
        str: A JSON string containing the filtered measurement data.
             Returns an empty JSON object "{}" if the file cannot be parsed
             or no matching data is found.
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        ns = {'m': 'http://www.3gpp.org/ftp/specs/archive/28_series/28.532#measData'}

        # This will hold the aggregated parameters for the target cell
        target_cell_parameters = {}

        # The measTypes are defined per measInfo block, so we need to collect them
        # before processing measValues.
        # Assuming there's generally one main measInfo block containing all relevant types.
        # If there are multiple measInfo blocks with different measTypes,
        # the logic might need to be refined to handle scope.
        meas_types = {}
        for meas_info in root.findall('.//m:measInfo', ns):
            for mt in meas_info.findall('m:measType', ns):
                p_attr = mt.get('p')
                if p_attr:
                    meas_types[p_attr] = mt.text
        
        if not meas_types:
            print("Warning: No measType definitions found in the XML.")
            return json.dumps({})

        # Iterate through all measValue elements to find those matching the filter
        # NOTE: The provided XML structure has each parameter in its own measValue block.
        # The logic below correctly aggregates these if the GnbDuFunction and NrCellDu match.
        for mv in root.findall('.//m:measValue', ns):
            meas_obj_ldn = mv.get('measObjLdn')

            current_gnb_du = None
            current_nrcelldu = None

            # Extract GnbDuFunction and NrCellDu from measObjLdn
            if meas_obj_ldn:
                parts = meas_obj_ldn.split(',')
                for part in parts:
                    if part.startswith('GnbDuFunction='):
                        try:
                            current_gnb_du = int(part.split('=')[1])
                        except ValueError:
                            pass
                    elif part.startswith('NrCellDu='):
                        try:
                            current_nrcelldu = int(part.split('=')[1])
                        except ValueError:
                            pass

            # Apply filters.
            # If filter_gnb_du is None, it means no filter is applied for GnbDuFunction,
            # so current_gnb_du will always match. Same for NrCellDu.
            gnb_du_filter_applies = (filter_gnb_du is None) or (current_gnb_du == filter_gnb_du)
            nrcelldu_filter_applies = (filter_nrcelldu is None) or (current_nrcelldu == filter_nrcelldu)

            if gnb_du_filter_applies and nrcelldu_filter_applies:
                # This measValue block is either unfiltered or matches our criteria
                r_tag = mv.find('m:r', ns)
                if r_tag is not None:
                    p_value = r_tag.get('p')
                    if p_value and p_value in meas_types:
                        param_name = meas_types[p_value]
                        try:
                            float_val = float(r_tag.text)
                            target_cell_parameters[param_name] = float_val
                        except (ValueError, TypeError):
                            target_cell_parameters[param_name] = r_tag.text
        
        return json.dumps(target_cell_parameters, indent=2)

    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return json.dumps({})
    except FileNotFoundError:
        print(f"Error: File not found at {xml_file_path}")
        return json.dumps({})
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return json.dumps({})
    
def get_measurement_data(filter_gnb_du=None, filter_nrcelldu=None)-> str:
    """
    Wrapper function to get measurement data from an XML file.
    This function is a placeholder for any additional processing or
    integration with other components in the system.

    Args:
        xml_file_path (str): The path to the XML measurement data file.
        filter_gnb_du (int, optional): The GnbDuFunction ID to filter by.
                                       If None, no filter is applied.
        filter_nrcelldu (int, optional): The NrCellDu ID to filter by.
                                         If None, no filter is applied.

    Returns:
        str: A JSON string containing the filtered measurement data.
    """
    # target_directory = "examples/measdata" # Change this to your actual directory
    target_directory = "/tmp/viavi/o1/pm_reports" # Change this to your actual directory
    filter_gnb_du = int(filter_gnb_du) if filter_gnb_du else None
    filter_nrcelldu = int(filter_nrcelldu) if filter_nrcelldu else None

    print("Debug value of filter_gnb_du:", filter_gnb_du)
    print("Debug value of  filter_nrcelldu:", filter_nrcelldu)
    latest_file_path = get_latest_xml_file_by_time(target_directory)
    print("file opened: ",latest_file_path)
    parsed_data = parse_measurement_data(latest_file_path, filter_gnb_du, filter_nrcelldu)
    pm_metrics = json.loads(parsed_data)
    
    keys_for_value_conversion = {
        # "DRB.UEThpDl",
        # "DRB.UEThpUl",
        # "QosFlow.TotPdcpPduVolumeUl",
        # "QosFlow.TotPdcpPduVolumeDl",
        # "PEE.AvgPower" 
        "Viavi.PEE.EnergyEfficiency"
    }

    processed_data = {} # Initialize a new dictionary for the results

    # Iterate through each key-value pair in the original PM metrics
    for original_key, original_value in pm_metrics.items():

        processed_value = original_value # Start with the original value

        # First, attempt to convert string values to actual numbers (int/float)
        if isinstance(original_value, str):
            try:
                processed_value = int(original_value)
            except ValueError:
                try:
                    processed_value = float(original_value)
                except ValueError:
                    # If it's a string but not a number, keep it as is
                    print(f"Metric '{original_key}' has non-numeric string value '{original_value}'. Skipping numeric conversion.")
                    processed_value = original_value # Keep the original string if not convertible
        
        # Now, check if this key is one we need to apply unit conversions to
        if original_key in keys_for_value_conversion:
            # Apply specific unit conversions based on the ORIGINAL key's assumed input unit
            # if original_key in ["QosFlow.TotPdcpPduVolumeUl", "QosFlow.TotPdcpPduVolumeDl"]:
            #     # Assuming input is in Mbits, convert to kbits
            #     if isinstance(processed_value, (int, float)):
            #         processed_value *= 1_000 # 1 Mbit = 1,000,000 bits
            #         print(f"Converted '{original_key}' from Mbits to kbits. New value: {processed_value}")
                
            # if original_key in ["DRB.UEThpUl", "DRB.UEThpDl"]:
            #     # Assuming input is in Gbps, convert to bps
            #     if isinstance(processed_value, (int, float)):
            #         processed_value *= 1_000_000 # 1 Gbps = 1,000,000 bps
            #         print(f"Converted '{original_key}' from Gbps to bps. New value: {processed_value}")
            
            # Add other conversion logic for PEE.AvgPower or other metrics here if needed
            # For example, if PEE.AvgPower was in kW and you want Watts:
            # elif original_key == "PEE.AvgPower":
            #     if isinstance(processed_value, (int, float)):
            #         processed_value *= 1_000 # kW to Watts

            #   Add other conversion logic for PEE.AvgPower or other metrics here if needed
            # For example, if PEE.AvgPower was in kW and you want Watts:
            if original_key == "Viavi.PEE.EnergyEfficiency":
                if isinstance(processed_value, (int, float)):
                    processed_value *= 1_000 # kbit/joule to bit/joule

        # Add the processed value with its original key to the result dictionary
        processed_data[original_key] = processed_value
            
    return json.dumps(processed_data, indent=2)
    # return parse_measurement_data(latest_file_path, filter_gnb_du, filter_nrcelldu)

# --- Example Usage ---

# # file_path = "measdata.xml"
# target_directory = "examples/measdata" # Change this to your actual directory

# latest_file_path = get_latest_xml_file_by_time(target_directory)

# # Example 1: Read all data
# print("--- All Measurement Data ---")
# all_data_json = parse_measurement_data(latest_file_path)
# print(all_data_json)
# print("-" * 30 + "\n")

# # Example 2: Filter by GnbDuFunction=1
# print("--- Filtered for GnbDuFunction=1 ---")
# filtered_data_gnb1_json = parse_measurement_data(latest_file_path, filter_gnb_du=1)
# print(filtered_data_gnb1_json)
# print("-" * 30 + "\n")

# # Example 3: Filter by GnbDuFunction=2 and NrCellDu=2
# print("--- Filtered for GnbDuFunction=2, NrCellDu=2 ---")
# filtered_data_gnb2_cell2_json = parse_measurement_data(latest_file_path, filter_gnb_du=2, filter_nrcelldu=2)
# print(filtered_data_gnb2_cell2_json)
# print("-" * 30 + "\n")

# # Example 4: Filter for a non-existent combination (should return empty measurements)
# print("--- Filtered for GnbDuFunction=99, NrCellDu=99 (non-existent) ---")
# non_existent_data_json = parse_measurement_data(latest_file_path, filter_gnb_du=99, filter_nrcelldu=99)
# print(non_existent_data_json)
# print("-" * 30 + "\n")

# print("--- Filtered for GnbDuFunction=99, NrCellDu=99 (non-existent) ---")
# print(get_measurement_data(2,"2"))
# print("-" * 30 + "\n")
