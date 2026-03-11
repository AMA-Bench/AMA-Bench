import re
from typing import Dict, List, Tuple


def parse_task(task: str) -> Tuple[str, List[str], str]:
    """Parse task to identify type, target objects and goal location.
    
    Args:
        task: Task description string
        
    Returns:
        Tuple of (task_type, target_objects, target_location)
    """
    task_lower = task.lower()
    task_type = ""
    target_objects = []
    target_location = ""
    
    if "cook" in task_lower and "eat" in task_lower:
        task_type = "cook_and_eat"
        match = re.search(r"cook (?:and eat )?(?:the )?(\w+)", task_lower)
        if match:
            target_objects.append(match.group(1))
    
    elif "prepare" in task_lower and "meal" in task_lower:
        task_type = "meal_preparation"
        words = task_lower.split()
        for i, word in enumerate(words):
            if word in ["with", "using"] and i + 1 < len(words):
                target_objects.append(words[i + 1])
    
    elif ("take" in task_lower or "get" in task_lower) and ("put" in task_lower or "place" in task_lower):
        task_type = "pick_and_place"
        take_match = re.search(r"(?:take|get) (?:the )?(\w+)", task_lower)
        if take_match:
            target_objects.append(take_match.group(1))
        place_match = re.search(r"(?:put|place|insert) .+? (?:in|on|into) (?:the )?(\w+)", task_lower)
        if place_match:
            target_location = place_match.group(1)
    
    elif "unlock" in task_lower or "open" in task_lower and "door" in task_lower:
        task_type = "treasure_hunt"
        key_match = re.search(r"(?:key|unlock) (?:the )?(\w+)", task_lower)
        if key_match:
            target_objects.append(key_match.group(1))
    
    elif "go to" in task_lower or "navigate" in task_lower:
        task_type = "navigation"
        loc_match = re.search(r"go to (?:the )?(\w+)", task_lower)
        if loc_match:
            target_location = loc_match.group(1)
    
    elif "examine" in task_lower or "look at" in task_lower:
        task_type = "examination"
        obj_match = re.search(r"(?:examine|look at) (?:the )?(\w+)", task_lower)
        if obj_match:
            target_objects.append(obj_match.group(1))
    
    elif "eat" in task_lower:
        task_type = "eat_food"
        food_match = re.search(r"eat (?:the )?(\w+)", task_lower)
        if food_match:
            target_objects.append(food_match.group(1))
    
    else:
        task_type = "general"
        words = re.findall(r'\b[a-z]+\b', task_lower)
        potential_objects = [w for w in words if w not in 
                           ['the', 'a', 'an', 'you', 'your', 'to', 'from', 'in', 'on', 'at', 'and', 'or']]
        if potential_objects:
            target_objects = potential_objects[:2]
    
    return task_type, target_objects, target_location


def is_key_action(action: str, observation: str, task_type: str,
                  target_objects: List[str], target_location: str) -> bool:
    """Determine if an action is key based on task type and target objects.

    Logic priority:
    1. Check if action failed (return False if failed)
    2. Check if action contains target objects (high priority)
    3. Check if action contains target location (high priority)
    4. Check if action contains task-relevant keywords (task-specific)

    Args:
        action: Action string
        observation: Observation text after action
        task_type: Type of task (cook_and_eat, pick_and_place, etc.)
        target_objects: List of target object names
        target_location: Target location name

    Returns:
        True if action is a key action, False otherwise
    """
    action_lower = action.lower()
    obs_lower = observation.lower()

    # Check if action failed
    action_failed = any(keyword in obs_lower for keyword in
                       ["can't", "cannot", "nothing happens", "fail", "don't see"])

    if action_failed:
        return False

    # Priority 1: Check if any target object is in the action
    if target_objects and any(obj in action_lower for obj in target_objects):
        return True

    # Priority 2: Check if target location is in the action
    if target_location and target_location in action_lower:
        return True
    
    return False