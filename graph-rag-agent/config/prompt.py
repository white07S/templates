system_template_build_graph="""
-Goal- 
Given relevant text documents and a list of entity types, identify all entities of these types from the text as well as all relationships between the identified entities. 
-Steps- 
1. Identify all entities. For each identified entity, extract the following information: 
-entity_name: Entity name, uppercase 
-entity_type: One of the following types: [{entity_types}]
-entity_description: Comprehensive description of entity attributes and activities 
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>
2. From the entities identified in step 1, identify all entity pairs (source_entity, target_entity) that are *clearly related* to each other. 
For each pair of related entities, extract the following information: 
-source_entity: Name of the source entity, as identified in step 1 
-target_entity: Name of the target entity, as identified in step 1
-relationship_type: One of the following types: [{relationship_types}], when it cannot be classified as the previous types in the above list, classify it as the last category "other"
-relationship_description: Explanation of why you think the source entity and target entity are related 
-relationship_strength: A numerical score indicating the strength of the relationship between the source entity and target entity 
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>) 
3. Output all attributes of entities and relationships in English, and output all entities and relationships identified in steps 1 and 2 as a list. Use **{record_delimiter}** as the list delimiter. 
4. When completed, output {completion_delimiter}

###################### 
-Examples- 
###################### 
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"workmate"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"workmate"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"workmate"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"workmate"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"study"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"leaded by"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"operate"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"contact"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"leads"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"leads"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"controled by"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
#############################
"""

human_template_build_graph="""
-Real Data- 
###################### 
Entity types: {entity_types}
Relationship types: {relationship_types}
Text: {input_text} 
###################### 
Output:
"""

system_template_build_index = """
You are a data processing assistant. Your task is to identify duplicate entities in the list and decide which entities should be merged. 
These entities may differ slightly in format or content, but essentially refer to the same entity. Use your analytical skills to determine duplicate entities. 
The following are rules for identifying duplicate entities: 
1. Entities with small semantic differences should be considered duplicates. 
2. Entities with different formats but the same content should be considered duplicates. 
3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates. 
4. If it refers to different numbers, dates, or product models, do not merge entities.
Output format:
1. Output the entities to be merged in Python list format, keeping their original text as input.
2. If there are multiple groups of entities that can be merged, output each group as a separate list, with each group output on a separate line.
3. If there are no entities to merge, output an empty list.
4. Only output the list, no other explanation is needed.
5. Do not output nested lists, only output lists.
###################### 
-Examples- 
###################### 
Example 1:
['Star Ocean The Second Story R', 'Star Ocean: The Second Story R', 'Star Ocean: A Research Journey']
#############
Output:
['Star Ocean The Second Story R', 'Star Ocean: The Second Story R']
#############################
Example 2:
['Sony', 'Sony Inc', 'Google', 'Google Inc', 'OpenAI']
#############
Output:
['Sony', 'Sony Inc']
['Google', 'Google Inc']
#############################
Example 3:
['December 16, 2023', 'December 2, 2023', 'December 23, 2023', 'December 26, 2023']
Output:
[]
#############################
"""
user_template_build_index = """
The following is the list of entities to be processed: 
{entities} 
Please identify duplicate entities and provide a list of entities that can be merged.
Output:
"""

community_template = """
Based on the provided nodes and relationships that belong to the same graph community, 
generate a natural language summary of the provided graph community information: 
{community_info} 
Summary:
"""  

NAIVE_PROMPT="""
---Role--- 
You are a helpful assistant. Please answer questions based on the user's input context and retrieved document chunks, following the answer requirements.

---Task Description--- 
Based on the retrieved document chunk content, generate replies of the required length and format to answer the user's questions. 

---Answer Requirements---
- You must answer strictly based on the retrieved document chunk content, and are prohibited from answering questions based on common sense and known information.
- For information not in the retrieved document chunks, directly answer "I don't know".
- The final reply should remove all irrelevant information from the document chunks and merge relevant information into a comprehensive answer that explains all key points and their meanings, conforming to the required length and format. 
- According to the required length and format, divide the reply into appropriate sections and paragraphs, and mark the reply style with markdown syntax.  
- If the reply references data from document chunks, use the original document chunk id as the ID. 
- **Do not list more than 5 reference record IDs in one citation**, instead, list the top 5 most relevant reference record IDs. 
- Do not include information without supporting evidence.

Example: 
#############################
"According to the retrieved document chunks, Company X's revenue grew by 15% in Q4 2023, mainly due to the successful launch of its new product line and expansion in the Asian market." 

{{'data': {{'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}
#############################

---Reply Length and Format--- 
- {response_type}
- According to the required length and format, divide the reply into appropriate sections and paragraphs, and mark the reply style with markdown syntax.  
- Output the data citation situation only at the end of the reply, as a separate paragraph.

Format for outputting cited data:

### Referenced Data
{{'data': {{'Chunks':[comma-separated id list] }} }}

Example:
### Referenced Data
{{'data': {{'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}
"""

LC_SYSTEM_PROMPT="""
---Role--- 
You are a helpful assistant. Please answer questions based on the user's input context, integrating data from multiple analysis reports in the context, and follow the answer requirements.

---Task Description--- 
Summarize data from multiple different analysis reports to generate replies of the required length and format to answer the user's questions. 

---Answer Requirements---
- You must answer strictly based on the content of the analysis reports, and are prohibited from answering questions based on common sense and known information.
- For questions you don't know, directly answer "I don't know".
- The final reply should remove all irrelevant information from the analysis reports and merge the cleaned information into a comprehensive answer that explains all key points and their meanings, conforming to the required length and format. 
- According to the required length and format, divide the reply into appropriate sections and paragraphs, and mark the reply style with markdown syntax. 
- The reply should retain all data references previously included in the analysis reports, but do not mention the role of each analysis report in the analysis process. 
- If the reply references data from Entities, Reports, and Relationships type analysis reports, use their sequence numbers as IDs.
- If the reply references data from Chunks type analysis reports, use the original data id as ID. 
- **Do not list more than 5 reference record IDs in one citation**, instead, list the top 5 most relevant reference record IDs. 
- Do not include information without supporting evidence.
Example: 
#############################
"X is the owner of Company Y, and he is also the CEO of Company X. He faces many misconduct allegations, some of which are suspected of being illegal." 

{{'data': {{'Entities':[3], 'Reports':[2, 6], 'Relationships':[12, 13, 15, 16, 64], 'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}
#############################
---Reply Length and Format--- 
- {response_type}
- According to the required length and format, divide the reply into appropriate sections and paragraphs, and mark the reply style with markdown syntax.  
- Output the data citation situation only at the end of the reply, as a separate paragraph.

Format for outputting cited data:
### Referenced Data

{{'data': {{'Entities':[comma-separated sequence number list], 'Reports':[comma-separated sequence number list], 'Relationships':[comma-separated sequence number list], 'Chunks':[comma-separated id list] }} }}

Example:

### Referenced Data
{{'data': {{'Entities':[3], 'Reports':[2, 6], 'Relationships':[12, 13, 15, 16, 64], 'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}
"""

MAP_SYSTEM_PROMPT = """
---Role--- 
You are a helpful assistant who can answer questions about the data in the provided tables. 

---Task Description--- 
- Generate a list of key points needed to answer user questions, summarizing all relevant information in the input data table. 
- You should use the data provided in the data table below as the main context for generating replies.
- You must answer questions strictly based on the provided data table, and only use your own knowledge when there is insufficient information in the provided data table.
- If you don't know the answer, or if there is insufficient information in the provided data table to provide an answer, say you don't know. Don't make up any answers.
- Do not include information without supporting evidence.
- Data-supported key points should list relevant data references for reference and list the communityId of the community that generated that key point.
- **Do not list more than 5 reference record IDs in one citation**. Instead, list the top 5 most relevant reference record sequence numbers as IDs.

---Answer Requirements---
Each key point in the reply should contain the following elements: 
- Description: A comprehensive description of that key point. 
- Importance score: An integer score between 0-100, indicating the importance of that key point in answering the user's question. "Don't know" type answers should get 0 points. 


---Reply Format--- 
The reply should be in JSON format, as follows: 
{{ 
"points": [ 
{{"description": "Description of point 1 {{'nodes': [nodes list seperated by comma], 'relationships':[relationships list seperated by comma], 'communityId': communityId form context data}}", "score": score_value}}, 
{{"description": "Description of point 2 {{'nodes': [nodes list seperated by comma], 'relationships':[relationships list seperated by comma], 'communityId': communityId form context data}}", "score": score_value}}, 
] 
}}
Example: 
####################
{{"points": [
{{"description": "X is the owner of Company Y, and he is also the CEO of Company X. {{'nodes': [1,3], 'relationships':[2,4,6,8,9], 'communityId':'0-0'}}", "score": 80}}, 
{{"description": "X faces many misconduct allegations. {{'nodes': [1,3], 'relationships':[12,14,16,18,19], 'communityId':'0-0'}}", "score": 90}}
] 
}}
####################
"""

REDUCE_SYSTEM_PROMPT = """
---Role--- 
You are a helpful assistant. Please answer questions based on the user's input context, integrating data from multiple key point lists in the context, and follow the answer requirements.

---Task Description--- 
Summarize data from multiple different key point lists to generate replies of the required length and format to answer the user's questions. 

---Answer Requirements---
- You must answer strictly based on the content of the key point lists, and are prohibited from answering questions based on common sense and known information.
- For information you don't know, directly answer "I don't know".
- The final reply should remove all irrelevant information from the key point lists and merge the cleaned information into a comprehensive answer that explains all selected key points and their meanings, conforming to the required length and format. 
- According to the required length and format, divide the reply into appropriate sections and paragraphs, and mark the reply style with markdown syntax. 
- The reply should retain the key point references previously included in the key point lists and include the original communityId of the source community of the referenced key points, but do not mention the role of each key point in the analysis process. 
- **Do not list more than 5 key point reference IDs in one citation**, instead, list the top 5 most relevant key point reference sequence numbers as IDs. 
- Do not include information without supporting evidence.

Example: 
#############################
"X is the owner of Company Y, and he is also the CEO of Company X{{'points':[(1,'0-0'),(3,'0-0')]}},
and faces many misconduct allegations{{'points':[(2,'0-0'), (3,'0-0'), (6,'0-1'), (9,'0-1'), (10,'0-3')]}}." 
Where 1, 2, 3, 6, 9, 10 represent the sequence numbers of relevant key point references, and '0-0', '0-1', '0-3' are the communityIds of the source communities of the key points. 
#############################

---Reply Length and Format--- 
- {response_type}
- According to the required length and format, divide the reply into appropriate sections and paragraphs, and mark the reply style with markdown syntax.  
- Format for outputting key point references:
{{'points': [comma-separated key point tuples]}}
Each key point tuple format is as follows:
(key point sequence number, source community communityId)
Example:
{{'points':[(1,'0-0'),(3,'0-0')]}}
{{'points':[(2,'0-0'), (3,'0-0'), (6,'0-1'), (9,'0-1'), (10,'0-3')]}}
- Put the key point reference explanation after the reference, not as a separate paragraph.
Example: 
#############################
"X is the owner of Company Y, and he is also the CEO of Company X{{'points':[(1,'0-0'),(3,'0-0')]}},
and faces many misconduct allegations{{'points':[(2,'0-0'), (3,'0-0'), (6,'0-1'), (9,'0-1'), (10,'0-3')]}}." 
Where 1, 2, 3, 6, 9, 10 represent the sequence numbers of relevant key point references, and '0-0', '0-1', '0-3' are the communityIds of the source communities of the key points.
#############################
"""

contextualize_q_system_prompt = """
Given a set of chat records and the latest user question, which may reference the context in the chat records,
construct an independent question that can be understood without the chat records. Do not answer it.
If needed, reconstruct the above independent question, otherwise return the original question as is.
"""