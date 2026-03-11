"""
Query tools for state memory retrieval
"""
import re
import json
from typing import Dict, List, Any, Optional, Union


def traj_get(
    trajectory_text: str,
    span: Optional[Dict[str, Any]] = None,
    fields: Optional[List[str]] = None,
    auto_compress: bool = False
) -> str:
    """
    Get evidence segments from trajectory.
    
    Args:
        trajectory_text: Full trajectory text in JSON format
        span: Optional dict with either {'indices': [1, 2, 3]} or {'start': 1, 'end': 3}
        fields: Optional list of fields to include, default ['action', 'observation', 'action_space']
        auto_compress: Whether to auto-compress the output
    
    Returns:
        Extracted trajectory segment as formatted string
    """
    if fields is None:
        fields = ['action', 'observation', 'action_space']
    
    trajectory_data = json.loads(trajectory_text)
    trajectory = trajectory_data.get('trajectory', [])
    
    if span is None:
        selected_turns = trajectory
    elif 'indices' in span:
        indices = span['indices']
        if not isinstance(indices, list):
            indices = [indices]
        selected_turns = [turn for turn in trajectory if turn.get('turn_idx') in indices]
    elif 'start' in span and 'end' in span:
        start_idx = span['start']
        end_idx = span['end']
        selected_turns = [turn for turn in trajectory 
                         if start_idx <= turn.get('turn_idx', 0) <= end_idx]
    else:
        selected_turns = trajectory
    
    result_lines = []
    for turn in selected_turns:
        turn_idx = turn.get('turn_idx', 0)
        result_lines.append(f"Turn {turn_idx}:")
        
        for field in fields:
            if field in turn:
                value = turn[field]
                if auto_compress and isinstance(value, str) and len(value) > 300:
                    value = value[:300] + "..."
                result_lines.append(f"  {field}: {value}")
    
    return "\n".join(result_lines)


def traj_find(
    trajectory_text: str,
    query: str,
    mode: str = "keyword"
) -> List[int]:
    """
    Find turn indices that match the query.
    
    Args:
        trajectory_text: Full trajectory text in JSON format
        query: Query string (keyword, regex, action, entity, room, etc.)
        mode: Search mode - 'keyword', 'regex', 'action', or 'entity'
    
    Returns:
        List of turn indices that match the query
    """
    trajectory_data = json.loads(trajectory_text)
    trajectory = trajectory_data.get('trajectory', [])
    
    matched_indices = []
    
    for turn in trajectory:
        turn_idx = turn.get('turn_idx', 0)
        matched = False
        
        if mode == "keyword":
            query_lower = query.lower()
            action = turn.get('action', '').lower()
            observation = turn.get('observation', '').lower()
            if query_lower in action or query_lower in observation:
                matched = True
        
        elif mode == "regex":
            pattern = re.compile(query, re.IGNORECASE)
            action = turn.get('action', '')
            observation = turn.get('observation', '')
            if pattern.search(action) or pattern.search(observation):
                matched = True
        
        elif mode == "action":
            action = turn.get('action', '').lower()
            if query.lower() in action:
                matched = True
        
        elif mode == "entity":
            action = turn.get('action', '')
            observation = turn.get('observation', '')
            combined_text = action + " " + observation
            if query in combined_text:
                matched = True
        
        if matched:
            matched_indices.append(turn_idx)
    
    return matched_indices

def get_openai_tools() -> List[Dict[str, Any]]:
    """
    Get tool definitions in OpenAI function calling format.
    
    Returns:
        List of tool definitions compatible with OpenAI's tools parameter
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "traj_find",
                "description": """Search and find turn indices in the trajectory that match a specific query. 
                
This tool helps you locate relevant turns before retrieving detailed information.
                
                **Parameters:**
                - query: The search term (e.g., 'open door', 'pick up key', 'red box', 'living room')
                - mode: Search strategy
                  * 'keyword': Search for the query string anywhere in action or observation (default)
                  * 'action': Search only in the action field
                  * 'regex': Use regular expression pattern matching
                  * 'entity': Search for specific entity mentions
                
                **Returns:** A JSON object with 'indices' (list of matching turn numbers) and 'count' (number of matches)
                
                **Examples:**
                - Find turns with 'open': traj_find(query='open', mode='keyword')
                - Find turns with action 'go to': traj_find(query='go to', mode='action')
                - Find turns mentioning 'key': traj_find(query='key', mode='entity')
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query string to search for. Examples: 'open door', 'key', 'cabinet', 'red', 'living room'"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["keyword", "regex", "action", "entity"],
                            "description": "Search mode: 'keyword' (default, search anywhere), 'action' (only in action field), 'regex' (pattern matching), 'entity' (specific entity)",
                            "default": "keyword"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "traj_get",
                "description": """Retrieve detailed information from specific turns in the trajectory.
                
                Use this tool AFTER traj_find to get the full details of relevant turns.
                
                **Parameters:**
                - span: Specifies which turns to retrieve (REQUIRED)
                  * Use {'indices': [1, 2, 3]} to get specific turn numbers
                  * Use {'start': 1, 'end': 5} to get a range of turns (inclusive)
                  
                - fields: Which information to include (optional, defaults to all)
                  * 'action': The action taken at this turn
                  * 'observation': The observation/feedback received
                  * 'action_space': Available actions at this turn
                
                **Returns:** Formatted text with detailed information from the requested turns
                
                **Examples:**
                - Get turn 5: traj_get(span={'indices': [5]})
                - Get turns 1, 3, 7: traj_get(span={'indices': [1, 3, 7]})
                - Get turns 10-15: traj_get(span={'start': 10, 'end': 15})
                - Get only actions from turn 5: traj_get(span={'indices': [5]}, fields=['action'])
                
                **Usage Pattern:**
                1. First use traj_find to locate relevant turns
                2. Then use traj_get with the found indices to get details
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "span": {
                            "type": "object",
                            "description": "Specifies which turns to retrieve. Use 'indices' for specific turns [1,2,3] or 'start'/'end' for a range.",
                            "properties": {
                                "indices": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "description": "List of specific turn indices to retrieve. Example: [1, 3, 5, 7]"
                                },
                                "start": {
                                    "type": "integer",
                                    "description": "Start turn index (inclusive) for range retrieval. Example: 1"
                                },
                                "end": {
                                    "type": "integer",
                                    "description": "End turn index (inclusive) for range retrieval. Example: 10"
                                }
                            }
                        },
                        "fields": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["action", "observation", "action_space"]
                            },
                            "description": "Which fields to include. Options: 'action' (what was done), 'observation' (result/feedback), 'action_space' (available actions). Default: all fields",
                            "default": ["action", "observation", "action_space"]
                        }
                    },
                    "required": ["span"]
                }
            }
        }
    ]


def execute_tool_call(
    tool_name: str,
    arguments: Dict[str, Any],
    trajectory_text: str
) -> str:
    """
    Execute a tool call with given arguments.
    
    Args:
        tool_name: Name of the tool to execute ('traj_find' or 'traj_get')
        arguments: Arguments for the tool
        trajectory_text: Full trajectory text in JSON format
        
    Returns:
        Result of the tool execution as string
    """
    if tool_name == "traj_find":
        query = arguments.get("query", "")
        mode = arguments.get("mode", "keyword")
        indices = traj_find(trajectory_text, query, mode)
        return json.dumps({"indices": indices, "count": len(indices)})
    
    elif tool_name == "traj_get":
        span = arguments.get("span")
        fields = arguments.get("fields")
        result = traj_get(trajectory_text, span, fields)
        return result
    
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
