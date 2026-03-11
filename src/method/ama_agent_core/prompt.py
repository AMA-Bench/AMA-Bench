CODE_GENERATION_PROMPT_TEMPLATE = """You are helping to extract relevant information from a trajectory to answer a question by writing Python code.

**Question:** {query}

**Task:** {task}

**Trajectory Format Reference (first 2 turns):**
{trajectory_sample}

**Trajectory Data (JSON format):**
Available as variable `trajectory_json` with structure:
{{
  "trajectory": [
    {{
      "turn_idx": 0,      // Turn number (int)
      "action": "...",    // Action taken at this turn (string)
      "observation": "..." // Environment observation after action (string)
    }},
    ...
  ],
  "task": "...",        // Task description (string)
  "episode_id": "..."   // Episode identifier (string)
}}

**Your Task:**
Write Python code that processes the trajectory JSON and extracts the relevant information to answer the question.

**Available Examples:**

Example 1: Finding specific actions
```python
import json

# Parse trajectory
trajectory_data = json.loads(trajectory_json)
trajectory = trajectory_data['trajectory']

# Find turns where a specific action was taken
relevant_turns = []
for turn in trajectory:
    if 'pick up' in turn.get('action', '').lower():
        relevant_turns.append({{
            'turn': turn['turn_idx'],
            'action': turn['action'],
            'observation': turn.get('observation', '')[:200]
        }})

result = {{
    'relevant_turns': relevant_turns,
    'count': len(relevant_turns)
}}
```



Example 2: Finding when something happened (until turn X)
```python
import json

trajectory_data = json.loads(trajectory_json)
trajectory = trajectory_data['trajectory']

# Find first turn when door was opened
first_open = None
for turn in trajectory:
    if 'open door' in turn.get('action', '').lower():
        first_open = turn['turn_idx']
        break

# Get all turns until that point
turns_until_open = [t for t in trajectory if t['turn_idx'] <= first_open] if first_open else []

result = {{
    'event': 'door opened',
    'first_occurrence': first_open,
    'turns_until_event': len(turns_until_open)
}}
```

Example 3: Finding last occurrence of something
```python
import json

trajectory_data = json.loads(trajectory_json)
trajectory = trajectory_data['trajectory']

# Find last turn where agent picked up something
last_pickup = None
for turn in reversed(trajectory):
    if 'pick up' in turn.get('action', '').lower():
        last_pickup = {{
            'turn': turn['turn_idx'],
            'action': turn['action'],
            'observation': turn.get('observation', '')[:200]
        }}
        break

result = {{'last_pickup': last_pickup}}
```

Example 4: Causal relationship - what happened after X
```python
import json

trajectory_data = json.loads(trajectory_json)
trajectory = trajectory_data['trajectory']

# Find when key was picked up, then what happened next
key_pickup_turn = None
for turn in trajectory:
    if 'key' in turn.get('action', '').lower() and 'pick' in turn.get('action', '').lower():
        key_pickup_turn = turn['turn_idx']
        break

# Get next 3 turns after picking up key
subsequent_actions = []
if key_pickup_turn is not None:
    for turn in trajectory:
        if turn['turn_idx'] > key_pickup_turn and len(subsequent_actions) < 3:
            subsequent_actions.append({{
                'turn': turn['turn_idx'],
                'action': turn['action']
            }})

result = {{
    'trigger_event': 'picked up key',
    'trigger_turn': key_pickup_turn,
    'subsequent_actions': subsequent_actions
}}
```

**Instructions:**
1. Write Python code that processes the trajectory JSON (available as variable `trajectory_json`)
2. Extract information relevant to answering the question
3. The code should be self-contained and executable
4. Store the final result in a variable named `result`

**Output Format:**
You MUST format your response as follows:

**CODE**:
```python
# Your Python code here
```

Important: The code must be wrapped with **CODE**: marker followed by ```python code block.
<think><\think>
"""

COMPRESS_PROMPT_TEMPLATE = """You are analyzing a trajectory chunk to extract structured state information.

Task: {task}

Trajectory Chunk:
{trajectory_text}

{previous_state_text}

Your task is to extract and organize key state information from the trajectory.

You can use any format that works best (JSON, structured text, bullet points, etc.). For example:

Example JSON format:
{{
  "objects": ["obj_1", "obj_2", "obj_3"],
  "obj_state": {{
    "obj_1": [
      {{"t": 1, "action":"", "state": "exact description"}},
      {{"t": 5, "action":"", "state": "exact description"}}
    ]
  }}
}}


Key requirements:
1. Identify ALL relevant objects/entities/locations mentioned
2. Track their state changes at specific turns with EXACT details
3. Include precise values, locations, and concrete actions
4. Record exact error messages or feedback when relevant
5. List events chronologically for each object

After your analysis, output your state memory after the marker:

**STATE_MEMORY**
[Your state memory content here]
"""

CHECK_STATE_MEM_PROMPT_TEMPLATE = """You are analyzing whether the compressed state memory contains enough information to answer a question.

State Memory (compressed representation of the trajectory):
{state_mem_str}

Question: {query}

Analyze if the state memory contains sufficient information to answer this question accurately.

Consider:
1. Does the state memory mention the relevant objects/entities in the question?
2. Does it contain the specific information needed (states, relationships, actions)?
3. Is the information detailed enough or just vague references?

Respond with ONLY "SUFFICIENT" or "NEED_RETRIEVAL" followed by a brief reason.

Format:
SUFFICIENT: [reason why state memory is enough]
or
NEED_RETRIEVAL: [what specific information is missing and needs to be retrieved]

Response:"""

TOOL_USE_PROMPT_TEMPLATE = """You are helping retrieve relevant information from a trajectory to answer a question.

**Question:** {query}

**Available Tools:**

You have access to TWO powerful tools to search and retrieve information from the trajectory:

1. **traj_find** - Locates relevant turns
   - Purpose: Search for specific keywords/entities/actions in the trajectory
   - Parameters:
     * query (required): The search term (e.g., "open door", "key", "red box")
     * mode (optional): Search strategy
       - "keyword": Search anywhere in text (default)
       - "action": Search only in action field
       - "entity": Search for specific entity mentions
   - Returns: List of turn indices where the query was found
   - Example: traj_find(query="pick up", mode="action")

2. **traj_get** - Retrieves detailed information
   - Purpose: Get full details from specific turns
   - Parameters:
     * span (required): Which turns to get
       - {{"indices": [1, 2, 3]}} for specific turns
       - {{"start": 1, "end": 5}} for a range
     * fields (optional): What info to include ["action", "observation", "action_space"]
   - Returns: Formatted text with detailed turn information
   - Example: traj_get(span={{"indices": [5, 7, 9]}})

**Recommended Strategy:**
1. Use traj_find to locate turns related to the question
2. Use traj_get to retrieve detailed information from those turns
3. You can call tools multiple times to gather complete information

**Your Task:**
Use these tools strategically to find and retrieve ALL relevant information needed to answer the question thoroughly."""

ANSWER_WITH_RETRIEVAL_PROMPT_TEMPLATE = """Based on the compressed state memory and retrieved detailed information, provide a natural language answer to the query.

Query: {query}

State Memory (compressed):
{state_mem_str}

Retrieved Detailed Information:
{relevant_mem}

CRITICAL: You MUST format your response as follows:
ANSWER: [Your concise, accurate answer here]

Only include the answer after "ANSWER:", nothing else."""

ANSWER_WITHOUT_RETRIEVAL_PROMPT_TEMPLATE = """Based on the compressed state memory, provide a natural language answer to the query.

Query: {query}

State Memory:
{state_mem_str}

CRITICAL: You MUST format your response as follows:
ANSWER: [Your concise, accurate answer here]

Only include the answer after "ANSWER:", nothing else."""

CAUSAL_PROMPT_TEMPLATE = """You are analyzing a trajectory to extract causal relationships between events and state changes.

Task: {task}

Trajectory:
{trajectory_text}

{previous_state_text}

Your task is to identify and extract causal relationships from the trajectory.

For each causal relationship, identify:
1. The CAUSE: an action or event that triggers a change
2. The EFFECT: the resulting state change or consequence
3. The TURN(S): when this causal relationship occurs

Output your response after the markers below.

**CAUSAL_GRAPH**
[
  {{
    "cause": "description of triggering action/event",
    "effect": "description of resulting state change",
    "cause_turn": <turn number>,
    "effect_turn": <turn number>,
    "entities": ["entity1", "entity2"]
  }},
  ...
]

**STATE_MEMORY**
[Your state memory content here]
"""
