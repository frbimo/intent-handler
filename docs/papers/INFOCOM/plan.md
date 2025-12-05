# Plan Revision for INFOCOM '25 Submission


| Aspects                                              | Complexity(1/easy - 9/difficult) | Importance(1/high-9/low) | Implement?         |
| ---------------------------------------------------- | -------------------------------- | ------------------------ | ------------------ |
| **I. Novelty& Contribution**                         |                                  |                          |                    |
| Lack of Novelty (Redesign for unique technical core) | 9                                | 1                        |  |
| Missing Technical Contribution Section               | 3                                | 7                        | :white_check_mark: |
| Agent Lacks RAN-Specific Logic                       | 4                                | 6                        | :white_check_mark: |
| **II. Validation**                                   |                                  |                          |                    |
| Missing Comparation Baselines                        | 6                                | 4                        |                    |
| Experimental validation                              | 8                                | 2                        |    :white_check_mark:                 |
| Lack of Statistical analysis                         | 5                                | 5                        | |
| **III. Modeling**                                        |                                  |                          |                    |
| System Model                                         | 7                                | 2                        |      :white_check_mark:               |
| Over-explanation of Concepts                         | 1                                | 8                        |     :white_check_mark:                |
| Poor Figure Quality                                  | 2                                | 9                        |          :white_check_mark:            |


1. Agent Lacks RAN-Specific Logic
   Action:
   - Revised Architecture

    <img width="827" height="392" alt="image" src="https://github.com/user-attachments/assets/31cb28c8-b838-48cd-aa57-e8a6e8786df2" />

   - Retain the idea of IBN with LLM for RAN Management. Use slicing as use case. The current architecture replaced with Kenny's intent translation(simplified intent input), partial function of Tobby's Graph and rapp that connects to Yueh Huan xapp to control AI RSG.
2. System Model
   Action:
   - Add system model and its explanatory

   <img width="685" height="397" alt="image" src="https://github.com/user-attachments/assets/f8183a54-7ae1-4db3-b94d-b9c5b7c077de" />

3. Over-explanation of Concepts
   Action:
   - Reduce intent as background knowledge in introduction. Add more reference for existing intent solution
  
4. Missing Technical Contribution Section
   Action:
   - Add paragraph on novel technical innovation we bring to RAN management. This can be done by solving "Agent Lacks RAN-Specific Logic" issue

5. Poor Figure Quality
   Action:
   - Increase font size and resolution for all figures
