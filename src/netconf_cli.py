import requests
import logging
from ncclient import manager
import xml.etree.ElementTree as ET
logger = logging.getLogger(__name__)

conf = '''
    <config xmlns:nc="urn:ietf:params:xml:ns:netconf:base:1.0">
    <ManagedElement xmlns="urn:3gpp:sa5:_3gpp-common-managed-element">
    <id>1193046</id>
    <GNBDUFunction xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-gnbdufunction">
        <id>2</id>
        <attributes>
        <peeParametersList>
            <idx>0</idx>
            <siteIdentification>S2</siteIdentification>
            <siteLatitude>0.0</siteLatitude>
            <siteLongitude>0.0054</siteLongitude>
            <siteAltitude>20.0</siteAltitude>
        </peeParametersList>
        <gNBId>1193046</gNBId>
        <gNBIdLength>24</gNBIdLength>
        <gNBDUId>2</gNBDUId>
        <gNBDUName>S2</gNBDUName>
        </attributes>
        <NRCellDU xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-nrcelldu">
        <id>2</id>
        <attributes>
            <userLabel>iMCGY7Gq3T0SphHWxoMAOEitNSldod</userLabel>
            <peeParametersList>
            <idx>0</idx>
            <siteIdentification>S2</siteIdentification>
            <siteLatitude>0.0</siteLatitude>
            <siteLongitude>0.0054</siteLongitude>
            <siteAltitude>20.0</siteAltitude>
            </peeParametersList>
            <priorityLabel>498288101</priorityLabel>
            <cellLocalId>2</cellLocalId>
            <operationalState>ENABLED</operationalState>
            <administrativeState>UNLOCKED</administrativeState>
            <cellState>ACTIVE</cellState>
            <pLMNInfoList>
            <mcc>001</mcc>
            <mnc>01</mnc>
            <sd>FFFFFF</sd>
            <sst>128</sst>
            </pLMNInfoList>
            <nRPCI>2</nRPCI>
            <nRTAC>4275132</nRTAC>
            <arfcnDL>713333</arfcnDL>
            <arfcnUL>713333</arfcnUL>
            <arfcnSUL>589612803</arfcnSUL>
            <bSChannelBwDL>60</bSChannelBwDL>
            <rimRSMonitoringWindowDuration>4059</rimRSMonitoringWindowDuration>
            <rimRSMonitoringWindowStartingOffset>19</rimRSMonitoringWindowStartingOffset>
            <rimRSMonitoringWindowPeriodicity>4</rimRSMonitoringWindowPeriodicity>
            <rimRSMonitoringOccasionInterval>296610638</rimRSMonitoringOccasionInterval>
            <rimRSMonitoringOccasionStartingOffset>46367730</rimRSMonitoringOccasionStartingOffset>
            <ssbFrequency>4700</ssbFrequency>
            <ssbPeriodicity>80</ssbPeriodicity>
            <ssbSubCarrierSpacing>120</ssbSubCarrierSpacing>
            <ssbOffset>48</ssbOffset>
            <ssbDuration>1</ssbDuration>
            <bSChannelBwUL>10</bSChannelBwUL>
            <bSChannelBwSUL>30</bSChannelBwSUL>
        </attributes>
        <CPCIConfigurationFunction xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-cpciconfigurationfunction">
            <id>2</id>
            <attributes>
            <cSonPciList>2</cSonPciList>
            </attributes>
        </CPCIConfigurationFunction>
        <NESPolicy xmlns="urn:viavi:viavi-o1">
            <id>1</id>
            <attributes>
            <policyType>ASM</policyType>
            <sleepMode>SLEEP_MODE0</sleepMode>
            </attributes>
        </NESPolicy>
        <viavi-attributes xmlns="urn:viavi:viavi-o1">
            <cellSize>medium</cellSize>
            <cellName>S2/SMALL CELL/C1</cellName>
            <siteName>S2</siteName>
            <latitude>0.0</latitude>
            <longitude>0.0054</longitude>
        </viavi-attributes>
        </NRCellDU>
        <NRSectorCarrier xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-nrnetwork-nrsectorcarrier">
        <id>2</id>
        <attributes>
            <userLabel>6T-Vw0Oo6BgI2n5</userLabel>
            <priorityLabel>349489325</priorityLabel>
            <txDirection>UL</txDirection>
            <configuredMaxTxPower>30</configuredMaxTxPower>
            <arfcnDL>240773</arfcnDL>
            <arfcnUL>421551</arfcnUL>
            <bSChannelBwDL>10</bSChannelBwDL>
            <bSChannelBwUL>10</bSChannelBwUL>
        </attributes>
        </NRSectorCarrier>
    </GNBDUFunction>
    </ManagedElement>
    </config>
    '''

class NETCONFCLIENT():
    

    def __init__(self, target_carrier_id,target_gnbdu_id):
        self.bandwidth = "100"
        self.target_carrier_id = target_carrier_id
        self.target_gnbdu_id = target_gnbdu_id
        self.new_tx_power = "0"
    
    def convert_to_xml(self, index):
        # Create the root element
        root = ET.Element("config", xmlns="urn:ietf:params:xml:ns:netconf:base:1.0")

        # Create ManagedElement
        managed_element = ET.SubElement(root, "ManagedElement", xmlns="urn:3gpp:sa5:_3gpp-common-managed-element")
        id_element = ET.SubElement(managed_element, "id")
        id_element.text = "1193046"
        
        # Create GNBDUFunction
        gnb_du_func = ET.SubElement(managed_element, "GNBDUFunction", xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-gnbdufunction")
        id_element = ET.SubElement(gnb_du_func, "id")
        id_element.text = "1"
        
        # # Create NRCellDU
        # nr_cell_du = ET.SubElement(gnb_du_func, "NRCellDU", xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-nrcelldu")
        # id_element = ET.SubElement(nr_cell_du, "id")
        # id_element.text = "1"

        # Create NRSectorCarrier
        nr_sector_carr = ET.SubElement(gnb_du_func, "NRSectorCarrier", xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-nrnetwork-nrsectorcarrier")
        id_element = ET.SubElement(nr_sector_carr, "id")
        id_element.text = "1"

        attributes = ET.SubElement(nr_sector_carr, "attributes")
        configured_tx_power = ET.SubElement(attributes, "configuredMaxTxPower")
        configured_tx_power.text = "30"
      
        userLabel = ET.SubElement(attributes, "userLabel")
        userLabel.text = "6T-Vw0Oo6BgI2n5"

        priorityLabel = ET.SubElement(attributes, "priorityLabel")
        priorityLabel.text = "349489325"

        txDirection = ET.SubElement(attributes, "txDirection")
        txDirection.text = "349489325"
        arfcn_dl = ET.SubElement(attributes, "arfcnDL")
        arfcn_dl.text = "380000"
        arfcn_ul = ET.SubElement(attributes, "arfcnUL")
        arfcn_ul.text = "380000"
        bSChannelBwDL = ET.SubElement(attributes, "bSChannelBwDL")
        bSChannelBwDL.text = "10"
        bSChannelBwUL = ET.SubElement(attributes, "bSChannelBwUL")
        bSChannelBwUL.text = "10"
        # bandwidth_element = ET.SubElement(attributes, "bandwidth")
        # bandwidth_element.text = self.bandwidth

        # cell_local_id = ET.SubElement(attributes, "cellLocalId")
        # cell_local_id.text = str(index)
        # administrative_state = ET.SubElement(attributes, "administrativeState")
        # administrative_state.text = "UNLOCKED"
        # pLMN_info_list = ET.SubElement(attributes, "pLMNInfoList")
        # mcc = ET.SubElement(pLMN_info_list, "mcc")
        # mcc.text = "001"
        # mnc = ET.SubElement(pLMN_info_list, "mnc")
        # mnc.text = "01"
        # sd = ET.SubElement(pLMN_info_list, "sd")
        # sd.text = "FFFFFF"
        # sst = ET.SubElement(pLMN_info_list, "sst")
        # sst.text = "128"
        # nRPCI = ET.SubElement(attributes, "nRPCI")
        # nRPCI.text = str(index)
        # arfcn_dl = ET.SubElement(attributes, "arfcnDL")
        # arfcn_dl.text = "380000"
        # arfcn_ul = ET.SubElement(attributes, "arfcnUL")
        # arfcn_ul.text = "380000"
        # ssb_frequency = ET.SubElement(attributes, "ssbFrequency")
        # ssb_frequency.text = "1900"
        
        # cpciconfiguration_function = ET.SubElement(nr_cell_du, "CPCIConfigurationFunction", xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-cpciconfigurationfunction")
        # id_element = ET.SubElement(cpciconfiguration_function, "id")
        # id_element.text = "1"
        # attributes = ET.SubElement(cpciconfiguration_function, "attributes")
        # cson_pci_list = ET.SubElement(attributes, "cSonPciList")        
        # cson_pci_list.text = str(index)
        
        # viavi_attributes = ET.SubElement(nr_cell_du, "viavi-attributes", xmlns="urn:viavi:viavi-o1")
        # cell_size = ET.SubElement(viavi_attributes, "cellSize")
        # cell_size.text = "medium"
        # cell_name = ET.SubElement(viavi_attributes, "cellName")
        # cell_name.text = "S1/B2/C1" 
        # site_name = ET.SubElement(viavi_attributes, "siteName")
        # site_name.text = "S1"
        # latitude = ET.SubElement(viavi_attributes, "latitude")
        # latitude.text = "0.0037"
        # longitude = ET.SubElement(viavi_attributes, "longitude")
        # longitude.text = "-0.016"
        
        # return ET.tostring(root, encoding="unicode")
        return 

    def perform_action(self, index):
        xml_data = self.convert_to_xml(index)
        print(xml_data)
        with manager.connect(host="192.168.8.28", port=30932, username="root", password="viavi", hostkey_verify=False) as m:
            try:
                print("*** Server Capabilities ***")
                for capability in m.server_capabilities:
                    print(capability)

                has_subtree_filter = ':subtree' in m.server_capabilities
                has_xpath_filter = ':xpath' in m.server_capabilities

                print("\n*** Filter Support ***")
                print(f"Supports Subtree Filtering: {has_subtree_filter}")
                print(f"Supports XPath Filtering: {has_xpath_filter}")

                namespaces = {
                    'ns0': 'urn:3gpp:sa5:_3gpp-common-managed-element',
                    'ns1': 'urn:3gpp:sa5:_3gpp-nr-nrm-gnbdufunction'
                }



 
            #     subtree_filter = f"""
            # <filter type="subtree">
            # <ManagedElement xmlns="urn:3gpp:sa5:_3gpp-common-managed-element">
            #     <GNBDUFunction xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-gnbdufunction">
            #     <id>1</id>
            #     </GNBDUFunction>
            # </ManagedElement>
            # </filter>
            # """
            #     carrier_to_update = "2"
            #     new_power_value = "21"
            #     updated_xml = update_tx_power_in_xml_string_with_prefixes_v3(conf, carrier_to_update, new_power_value)
            #     # edit_response = m.edit_config(target="running", config=updated_xml)
            #     get_response = m.get_config(source='running', filter=subtree_filter)
            #     print(get_response)
            #     logger.info(f"Successfully turn off the cell")
            #     # logger.info(f"Response: {edit_response}")

            except Exception as e:
                logger.error(f"Failed to turn off the cell: {str(e)}")

    # def convert_to_xml_1(self, index):
    #     root = ET.Element("config", xmlns="urn:ietf:params:xml:ns:netconf:base:1.0")
    #     managed_element = ET.SubElement(root, "ManagedElement", xmlns="urn:3gpp:sa5:_3gpp-common-managed-element")
    #     id_element = ET.SubElement(managed_element, "id")
    #     id_element.text = "1193046"
    #     gnb_ucp_function = ET.SubElement(managed_element, "GNBCUCPFunction", xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-gnbcucpfunction")
    #     id_element = ET.SubElement(gnb_ucp_function, "id")
    #     id_element.text = "1"
    #     nr_cell_cu = ET.SubElement(gnb_ucp_function, "NRCellCU", xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-nrcellcu")
    #     id_element = ET.SubElement(nr_cell_cu, "id")
    #     id_element.text = str(index)
    #     ces_management_function = ET.SubElement(nr_cell_cu, "CESManagementFunction", xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-cesmanagementfunction")
    #     id_element = ET.SubElement(ces_management_function, "id")
    #     id_element.text = str(index)
    #     attributes = ET.SubElement(ces_management_function, "attributes")
    #     energy_saving_control = ET.SubElement(attributes, "energySavingControl")
    #     energy_saving_control.text = "toBeNotEnergySaving"
    #     energy_saving_state = ET.SubElement(attributes, "energySavingState")
    #     energy_saving_state.text = "isNotEnergySaving"
    #     return ET.tostring(root, encoding="unicode")

    # def perform_action_1(self, index):
    #     xml_data = self.convert_to_xml_1(index)
    #     with manager.connect(host="192.168.8.28", port=31383, username="root", password="viavi", hostkey_verify=False) as m:
    #         try:
    #             edit_response = m.edit_config(target="running", config=xml_data)
    #             logger.info(f"Successfully turn on the cell")
    #         except Exception as e:
    #             logger.error(f"Failed to turn on the cell: {str(e)}")

    def configure_tx_power(self, new_tx_power: str):
        with manager.connect(host="192.168.8.28", port=30932, username="root", password="viavi", hostkey_verify=False) as m:
            try:
                # Perform the get-config operation with the XPath filter
                subtree_filter = f"""
                <filter type="subtree">
                <ManagedElement xmlns="urn:3gpp:sa5:_3gpp-common-managed-element">
                    <GNBDUFunction xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-gnbdufunction">
                    <id>1</id>
                    </GNBDUFunction>
                </ManagedElement>
                </filter>
                """

                xpath_filter = f"""
                <filter type="xpath" select="ns0:ManagedElement/ns1:GNBDUFunction[id='12']"
                </filter> 
                """

                result = m.get_config(source='running')
                # filtered_result = m.get_config(source='running', filter=xpath_filter)
                print("\n*** Filtered Result ***")
                
                xml_content = result.xml

                    # Define namespaces
                namespaces = {
                    'ns0': 'urn:3gpp:sa5:_3gpp-common-managed-element',
                    'ns1': 'urn:3gpp:sa5:_3gpp-nr-nrm-gnbdufunction',
                    'ns2': 'urn:3gpp:sa5:_3gpp-nr-nrm-nrcelldu',
                    'ns3': 'urn:3gpp:sa5:_3gpp-nr-nrm-nrnetwork-nrsectorcarrier',
                    'nc': 'urn:ietf:params:xml:ns:netconf:base:1.0'
                    # Add other namespaces if needed based on the full XML response
                }
                # Parse the XML response
                root = ET.fromstring(xml_content)

                found_gnbdu = None

                # Iterate through all GNBDUFunction elements
                for gnbdu_element in root.findall('.//ns1:GNBDUFunction', namespaces=namespaces):
                    id_element = gnbdu_element.find('ns1:id', namespaces=namespaces)
                    if id_element is not None and id_element.text == self.target_gnbdu_id:
                        found_gnbdu = gnbdu_element
                        break  # Stop iterating once found

                if found_gnbdu is not None:
                    # Find the specific NRSectorCarrier within this GNBDUFunction
                    for carrier_element in found_gnbdu.findall('ns3:NRSectorCarrier', namespaces=namespaces):
                        carrier_id_element = carrier_element.find('ns3:id', namespaces=namespaces)
                        if carrier_id_element is not None and carrier_id_element.text == self.target_carrier_id:
                            # Found the target NRSectorCarrier, now update the tx power
                            attributes_element = carrier_element.find('ns3:attributes', namespaces=namespaces)
                            if attributes_element is not None:
                                tx_power_element = attributes_element.find('ns3:configuredMaxTxPower', namespaces=namespaces)
                                print(tx_power_element.text)
                                if tx_power_element is not None:
                                    tx_power_element.text = new_tx_power
                                    print(f"Updated configuredMaxTxPower to '{new_tx_power}' in the local XML for GNBDU ID '{target_gnbdu_id}' and Carrier ID '{target_carrier_id}'.")

                                    # Serialize the modified gnbdu_element back to XML
                                    gnbdu_config_xml = ET.tostring(found_gnbdu, encoding='utf-8').decode()

                                    # Construct the <edit-config> payload
                                    edit_config_payload = f"""
                                    <config xmlns:nc="urn:ietf:params:xml:ns:netconf:base:1.0">
                                    <ManagedElement xmlns="urn:3gpp:sa5:_3gpp-common-managed-element">
                                        <id>1193046</id>
                                        {gnbdu_config_xml}
                                    </ManagedElement>
                                    </config>
                                    """

                                    edit_result = m.edit_config(target='running', config=edit_config_payload, default_operation='merge')
                                    if edit_result.ok:
                                        print("Configuration update successful.")
                                    else:
                                        print(f"Error during configuration update: {edit_result}")

                                else:
                                    print("configuredMaxTxPower element not found for the specified NRSectorCarrier.")
                            else:
                                print("attributes element not found for the specified NRSectorCarrier.")
                            break  # Stop searching for other carriers once the target is found
                    else:
                        print(f"NRSectorCarrier with id '{self.target_carrier_id}' not found under GNBDU ID '{self.target_gnbdu_id}'.")

                else:
                    print(f"GNBDUFunction with id '{self.target_gnbdu_id}' not found.")
            except Exception as e:
                logger.error(f"Failed to turn off the cell: {str(e)}")

    def get_current_tx_power(self):
        try:
            with manager.connect(host="192.168.8.28", port=30932, username="root", password="viavi", hostkey_verify=False) as m:
                # Use subtree filter to get NRSectorCarrier configuration
                filter_xml = f'''
                <ManagedElement xmlns="urn:3gpp:sa5:_3gpp-common-managed-element">
                    <id>1193046</id>
                    <GNBDUFunction xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-gnbdufunction">
                        <id>{self.target_gnbdu_id}</id>
                        <NRSectorCarrier xmlns="urn:3gpp:sa5:_3gpp-nr-nrm-nrnetwork-nrsectorcarrier">
                            <id>{self.target_carrier_id}</id>
                        </NRSectorCarrier>
                    </GNBDUFunction>
                </ManagedElement>
                '''
                
                result = m.get_config(source="running", filter=("subtree", filter_xml))
                
                # Define namespaces for parsing 
                namespaces = {
                    'ns0': 'urn:3gpp:sa5:_3gpp-common-managed-element',
                    'ns1': 'urn:3gpp:sa5:_3gpp-nr-nrm-gnbdufunction',
                    'ns3': 'urn:3gpp:sa5:_3gpp-nr-nrm-nrnetwork-nrsectorcarrier',
                    'nc': 'urn:ietf:params:xml:ns:netconf:base:1.0'
                }
                
                root = ET.fromstring(result.xml)
                
                # Find TX power value
                tx_power_element = root.find('.//ns3:configuredMaxTxPower', namespaces=namespaces)
                if tx_power_element is not None:
                    return tx_power_element.text
                else:
                    logger.warning(f"configuredMaxTxPower not found for GNBDU ID '{self.target_gnbdu_id}' and Carrier ID '{self.target_carrier_id}'")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to get current TX power: {str(e)}")
            return None

      
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     # netconf_client.perform_action_1(1)
#     target_carrier_id = "2"
#     target_gnbdu_id = "2"
#     new_tx_power = "28"
#     netconf_client = NETCONFCLIENT(target_carrier_id, target_gnbdu_id)


#     print("=== Before Configuaration ===")
#     current_tx_power = netconf_client.get_current_tx_power()
#     if current_tx_power:
#         print(f"Current TX Power: {current_tx_power} dBm")
#     else:
#         print("Unable to get TX Power value.")

#     netconf_client.configure_tx_power(new_tx_power)

#     print("\n=== After Configurattion ===")
#     updated_tx_power = netconf_client.get_current_tx_power()
#     if updated_tx_power:
#         print(f"New TX Power: {updated_tx_power} dBm")
        
#         # Comparision
#         if current_tx_power and updated_tx_power:
#             if current_tx_power != updated_tx_power:
#                 print(f"TX Power successfully updated from {current_tx_power} dBm to {updated_tx_power} dBm")
#             else:
#                 print(f"TX Power value unchanged, still {updated_tx_power} dBm")
#     else:
#         print("Unable to get new TX Power value.")