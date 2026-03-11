"""
BabyAI QA Generator

Generates QA pairs from BabyAI trajectories following similar patterns to TextWorld.
"""

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple


class BabyAIQAGenerator:
    """Generate QA pairs based on BabyAI trajectories."""

    def __init__(self, trajectory: List[Dict[str, Any]], task: str = "", task_type: str = "", seed: Optional[int] = None):
        self.trajectory = trajectory
        self.task = task
        self.task_type = task_type
        self.qa_pairs: List[Dict[str, str]] = []
        self.candidate_qa_list: List[Dict[str, Any]] = []
        self.used_keys: set = set()
        self.target_count: int = 12
        self.seed = seed
        # Set seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)

    def _extract_mission(self, observation: str) -> str:
        """Extract mission from observation."""
        match = re.search(r"Mission:\s*(.+?)(?:\n|$)", observation)
        return match.group(1).strip() if match else ""

    def _extract_visible_objects(self, observation: str) -> List[str]:
        """Extract visible objects from observation."""
        match = re.search(r"In your view:\s*(.+?)(?:\.|$)", observation)
        if not match:
            return []
        
        view_text = match.group(1)
        # Parse objects like "2x a green ball, a green box, a purple box"
        objects = []
        # Split by comma and process each item
        parts = [p.strip() for p in view_text.split(",")]
        for part in parts:
            # Handle "2x a green ball" -> "a green ball" (count 2)
            part = re.sub(r"^\d+x\s+", "", part)
            if part and part != "Walls border the area":
                objects.append(part.strip())
        return objects

    def _extract_direction(self, observation: str) -> str:
        """Extract facing direction from observation."""
        match = re.search(r"facing\s+(\w+)", observation)
        return match.group(1).strip() if match else ""

    def _extract_target_object(self, task: str) -> Optional[str]:
        """Extract target object from task description."""
        # Patterns: "go to a green ball", "pick up the grey box", "put the blue box next to..."
        patterns = [
            r"go to (?:a|the) (.+?)(?:$|,|\.)",
            r"pick up (?:a|the) (.+?)(?:$|,|\.)",
            r"put (?:a|the) (.+?) next to",
            r"put (?:a|the) (.+?) in",
        ]
        for pattern in patterns:
            match = re.search(pattern, task.lower())
            if match:
                return match.group(1).strip()
        return None

    def _add_candidate(self, key: str, question: str, answer: str, qa_type: str, step: int, subtype: str = None) -> bool:
        """Add a candidate QA pair. qa_type should be A, B, C, or D. subtype should be like A1, A2, B1, etc."""
        if key in self.used_keys:
            return False
        self.used_keys.add(key)
        
        normalized_type = qa_type.upper() if qa_type else "A"
        if normalized_type not in ["A", "B", "C", "D"]:
            normalized_type = "A"  # Default to A if invalid
        
        # Extract subtype from key if not provided
        if subtype is None:
            # Extract subtype from key (e.g., "a1_action_sequence" -> "A1")
            subtype_match = re.match(r"^([a-d])(\d+)_", key.lower())
            if subtype_match:
                subtype = f"{subtype_match.group(1).upper()}{subtype_match.group(2)}"
            else:
                # Default subtype based on type
                subtype = f"{normalized_type}1"
        
        self.candidate_qa_list.append({
            "question": question,
            "answer": answer,
            "type": normalized_type,
            "subtype": subtype.upper(),
            "step": step,
            "key": key
        })
        return True

    def _get_inventory_state(self, observation: str) -> Dict[str, Any]:
        """Extract inventory information from observation if available."""
        # BabyAI observations might contain inventory info
        # This is a placeholder - adjust based on actual observation format
        inventory = []
        if "carrying" in observation.lower() or "inventory" in observation.lower():
            # Try to extract what agent is carrying
            match = re.search(r"(?:carrying|inventory):\s*(.+?)(?:\n|$)", observation, re.IGNORECASE)
            if match:
                items = [item.strip() for item in match.group(1).split(",")]
                inventory = items
        return {"items": inventory, "empty": len(inventory) == 0}

    def maybe_add_per_step(self, t: int):
        """Collect candidate QA pairs for current step.
        
        Question types:
        - A: Temporal Information (A1-A4: various temporal questions)
        - B: State Dependency (B1-B3: state-action dependency questions)
        - C: State Update (C1-C2: inventory and object state changes)
        - D: State Summary (D1-D2: trajectory summary questions)
        """
        if t >= len(self.trajectory):
            return

        step = self.trajectory[t]
        action = step.get("action", "")
        observation = step.get("observation", "")

        def add(kind: str, key: str, q: str, a: str, subtype: str = None) -> None:
            self._add_candidate(key, q, a, qa_type=kind, step=t, subtype=subtype)

        mission = self._extract_mission(observation)
        visible_objects = self._extract_visible_objects(observation)
        direction = self._extract_direction(observation)

        # =========================================
        # A: Temporal Information
        # =========================================

        # A1: Ask what actions are like between several steps
        if t >= 2:
            # Get actions between steps
            start_step = max(0, t - 3)
            actions_sequence = []
            for step_idx in range(start_step, min(t + 1, len(self.trajectory))):
                act = self.trajectory[step_idx].get("action", "")
                if act:
                    actions_sequence.append(f"step {step_idx}: {act}")
            
            if len(actions_sequence) >= 3:
                q = f"What actions does the agent execute between step {start_step} and step {t}? Describe the sequence of actions."
                a = f"Between step {start_step} and step {t}, the agent executes the following actions: {'; '.join(actions_sequence)}."
                add("A", f"a1_action_sequence_{start_step}_{t}", q, a, subtype="A1")

        # A2: Ask before which step, what actions are there at step X for obj
        if t >= 1:
            prev_obs = self.trajectory[t-1].get("observation", "") if t > 0 else ""
            prev_visible = self._extract_visible_objects(prev_obs) if prev_obs else []
            prev_action = self.trajectory[t-1].get("action", "") if t > 0 else ""
            
            if prev_visible and visible_objects:
                # Find objects that were acted upon
                for obj in visible_objects:
                    if obj in prev_visible and action in ["pickup", "drop", "toggle"]:
                        # Check if this object was acted upon
                        action_desc = {"pickup": "picks up", "drop": "drops", "toggle": "toggles"}.get(action, action)
                        # Natural question that requires understanding context from previous step
                        q = f"When the agent encounters '{obj}' at step {t}, what does it do?"
                        a = f"At step {t}, the agent {action_desc} '{obj}' by executing the '{action}' action. This happens after the agent observed '{obj}' at step {t-1}."
                        add("A", f"a2_obj_action_before_{t}_{obj[:20]}", q, a, subtype="A2")
                        break

        # A3: Ask before which step, which objs were xxx'd at step X
        if t >= 1:
            prev_obs = self.trajectory[t-1].get("observation", "") if t > 0 else ""
            prev_visible = self._extract_visible_objects(prev_obs) if prev_obs else []
            
            if prev_visible and visible_objects:
                # Find objects that disappeared (picked up or moved away from)
                disappeared = [obj for obj in prev_visible if obj not in visible_objects]
                if disappeared and action in ["pickup", "forward", "left", "right"]:
                    obj = disappeared[0]
                    # Natural question that requires comparing states across steps
                    if action == "pickup":
                        q = f"What object disappears from the agent's view at step {t}?"
                        a = f"The object '{obj}' disappears from view at step {t} because the agent picks it up using the '{action}' action. It was visible at step {t-1} but is now in the agent's inventory."
                    else:
                        q = f"Which object is no longer visible after the agent moves at step {t}?"
                        a = f"After the agent executes the '{action}' action at step {t}, '{obj}' is no longer visible. It was visible at step {t-1}, but the agent's movement changed its field of view, causing '{obj}' to move out of sight."
                    add("A", f"a3_obj_acted_before_{t}", q, a, subtype="A3")

        # A4: Background: stepX1 state1, action1; stepX2 state2 action2
        # Ask: agent in state1 does action1, what are the subsequent state and action?
        if t >= 1 and t < len(self.trajectory) - 1:
            prev_step = self.trajectory[t-1]
            prev_obs = prev_step.get("observation", "")
            prev_action = prev_step.get("action", "")
            next_step = self.trajectory[t+1]
            next_obs = next_step.get("observation", "")
            next_action = next_step.get("action", "")
            
            if prev_obs and next_obs:
                prev_state_summary = f"state at step {t-1}: {prev_obs[:60]}..."
                next_state_summary = f"state at step {t+1}: {next_obs[:60]}..."
                q = f"Background: At step {t-1}, the environment state is '{prev_state_summary}', and the agent executes action '{prev_action}'. At step {t}, the agent executes action '{action}'. Question: When the agent is in the environment state at step {t-1} and executes action '{prev_action}', what are the subsequent state and action?"
                a = f"After the agent executes action '{prev_action}' in the state at step {t-1}, the subsequent state is '{next_state_summary}' (observed at step {t+1}), and the subsequent action is '{next_action}' executed at step {t+1}."
                add("A", f"a4_subsequent_state_action_{t}", q, a, subtype="A4")

        # =========================================
        # B: State Dependency
        # =========================================

        # B1: Background: stepX1 state1, action1; stepX2 state2 action2
        # Ask: what action can agent do to make state1 become state2
        if t >= 1:
            prev_step = self.trajectory[t-1]
            prev_obs = prev_step.get("observation", "")
            prev_visible = self._extract_visible_objects(prev_obs) if prev_obs else []
            curr_obs = observation
            curr_visible = visible_objects
            
            if prev_visible and curr_visible:
                # Check for state changes
                disappeared = [obj for obj in prev_visible if obj not in curr_visible]
                appeared = [obj for obj in curr_visible if obj not in prev_visible]
                
                if (disappeared or appeared) and action:
                    state1_summary = f"state at step {t-1}: visible objects include {', '.join(prev_visible[:3])}"
                    state2_summary = f"state at step {t}: visible objects include {', '.join(curr_visible[:3])}"
                    q = f"Background: At step {t-1}, the environment state is '{state1_summary}'. At step {t}, the environment state is '{state2_summary}'. Question: What action can the agent execute to make the state at step {t-1} become the state at step {t}?"
                    a = f"To make the state at step {t-1} become the state at step {t}, the agent should execute action '{action}'. This action causes: "
                    if disappeared:
                        a += f"objects {', '.join(disappeared[:2])} to disappear; "
                    if appeared:
                        a += f"objects {', '.join(appeared[:2])} to appear."
                    add("B", f"b1_action_to_change_state_{t}", q, a, subtype="B1")

        # B2: Background: stepX1 state1, action1; stepX2 state2 action2
        # Ask: agent in state1 does action1, what state2 will it become?
        if t >= 1:
            prev_step = self.trajectory[t-1]
            prev_obs = prev_step.get("observation", "")
            prev_visible = self._extract_visible_objects(prev_obs) if prev_obs else []
            curr_obs = observation
            curr_visible = visible_objects
            
            if prev_visible and curr_visible and action:
                state1_summary = f"state at step {t-1}: visible objects include {', '.join(prev_visible[:3])}"
                state2_summary = f"state at step {t}: visible objects include {', '.join(curr_visible[:3])}"
                q = f"Background: At step {t-1}, the environment state is '{state1_summary}', and the agent executes action '{action}'. At step {t}, the environment state is '{state2_summary}'. Question: When the agent is in the environment state at step {t-1} and executes action '{action}', what state will it become?"
                a = f"When the agent is in the environment state at step {t-1} and executes action '{action}', it will become the state at step {t}: '{state2_summary}'."
                add("B", f"b2_state_after_action_{t}", q, a, subtype="B2")

        # B3: Test dependency: background, this step inventory is empty or agent is not at that position, etc.
        # Ask: at step X can execute what action
        if t >= 1:
            prev_step = self.trajectory[t-1]
            prev_obs = prev_step.get("observation", "")
            prev_visible = self._extract_visible_objects(prev_obs) if prev_obs else []
            inventory_state = self._get_inventory_state(prev_obs)
            
            # Check if there are constraints that would prevent certain actions
            if inventory_state["empty"] and action == "drop":
                # Can't drop if inventory is empty
                q = f"Background: At step {t-1}, the agent's inventory is empty. Question: At step {t}, can the agent execute the 'drop' action? Why or why not?"
                a = f"At step {t}, the agent cannot execute the 'drop' action because the inventory is empty at step {t-1}. The agent needs to have an item in inventory to drop it."
                add("B", f"b3_dependency_empty_inventory_{t}", q, a, subtype="B3")
            elif not inventory_state["empty"] and action == "pickup":
                # Check if pickup is possible (object must be visible)
                if prev_visible:
                    q = f"Background: At step {t-1}, the agent's inventory is not empty (contains items), and visible objects include {', '.join(prev_visible[:2])}. Question: At step {t}, can the agent execute the 'pickup' action? Why or why not?"
                    a = f"At step {t}, the agent can execute the 'pickup' action because there are visible objects ({', '.join(prev_visible[:2])}) at step {t-1}, and the agent has space in inventory or can pick up objects."
                    add("B", f"b3_dependency_pickup_{t}", q, a, subtype="B3")

        # =========================================
        # C: State Update
        # =========================================

        # C1: What are the inventory changes, at which steps did they change respectively?
        if t >= 2:
            inventory_changes = []
            for step_idx in range(max(0, t-3), min(t+1, len(self.trajectory))):
                obs = self.trajectory[step_idx].get("observation", "")
                act = self.trajectory[step_idx].get("action", "")
                inv_state = self._get_inventory_state(obs)
                
                if act == "pickup":
                    inventory_changes.append(f"step {step_idx}: agent picks up item (inventory changes)")
                elif act == "drop":
                    inventory_changes.append(f"step {step_idx}: agent drops item (inventory changes)")
            
            if inventory_changes:
                q = f"What are the inventory changes from step {max(0, t-3)} to step {t}? At which steps did the inventory change respectively?"
                a = f"The inventory changes from step {max(0, t-3)} to step {t} are: {'; '.join(inventory_changes)}."
                add("C", f"c1_inventory_changes_{t}", q, a, subtype="C1")

        # C2: Background: obj changed at X1, X2
        # Ask: obj at step X2+Y state is what? At which step did it change, were there other changes before?
        if t >= 2:
            # Track object changes across steps
            target_obj = self._extract_target_object(self.task)
            if not target_obj and visible_objects:
                target_obj = visible_objects[0] if visible_objects else None
            
            if target_obj:
                change_steps = []
                for step_idx in range(max(0, t-4), min(t+1, len(self.trajectory))):
                    obs = self.trajectory[step_idx].get("observation", "")
                    act = self.trajectory[step_idx].get("action", "")
                    visible = self._extract_visible_objects(obs)
                    
                    prev_obs = self.trajectory[step_idx-1].get("observation", "") if step_idx > 0 else ""
                    prev_visible = self._extract_visible_objects(prev_obs) if prev_obs else []
                    
                    # Check if object appeared or disappeared
                    was_visible = any(target_obj.lower() in obj.lower() for obj in prev_visible) if prev_visible else False
                    is_visible = any(target_obj.lower() in obj.lower() for obj in visible) if visible else False
                    
                    if was_visible != is_visible:
                        change_type = "appeared" if is_visible and not was_visible else "disappeared"
                        change_steps.append((step_idx, change_type, act))
                
                if len(change_steps) >= 2:
                    x1, change1, act1 = change_steps[0]
                    x2, change2, act2 = change_steps[1] if len(change_steps) > 1 else change_steps[0]
                    y = 1  # X2 + Y
                    future_step = min(x2 + y, len(self.trajectory) - 1)
                    future_obs = self.trajectory[future_step].get("observation", "")
                    future_visible = self._extract_visible_objects(future_obs)
                    future_state = "visible" if any(target_obj.lower() in obj.lower() for obj in future_visible) else "not visible"
                    
                    q = f"Background: The object '{target_obj}' changed at step {x1} ({change1}) and step {x2} ({change2}). Question: What is the state of '{target_obj}' at step {x2 + y}? At which step did it change? Were there other changes before step {x2}?"
                    a = f"The object '{target_obj}' at step {x2 + y} is {future_state}. It changed at step {x2} ({change2}, action: '{act2}'). "
                    if len(change_steps) > 1:
                        a += f"Before step {x2}, there was another change at step {x1} ({change1}, action: '{act1}')."
                    else:
                        a += f"Before step {x2}, there were no other changes."
                    add("C", f"c2_obj_state_tracking_{t}", q, a, subtype="C2")

        # =========================================
        # D: State Summary
        # =========================================

        # D1: Summary of trajectory progress
        if t >= 3 and self.task:
            target_obj = self._extract_target_object(self.task)
            milestones = []
            key_actions = []
            
            for step_idx in range(min(t + 1, len(self.trajectory))):
                obs = self.trajectory[step_idx].get("observation", "")
                act = self.trajectory[step_idx].get("action", "")
                visible = self._extract_visible_objects(obs)
                
                if act and act not in ["forward", "left", "right"]:  # Non-movement actions
                    key_actions.append(f"step {step_idx}: {act}")
                
                if target_obj:
                    is_target_visible = any(target_obj.lower() in obj.lower() for obj in visible)
                    if is_target_visible and step_idx == 0:
                        milestones.append(f"step {step_idx}: target '{target_obj}' initially visible")
                    elif is_target_visible and step_idx > 0:
                        prev_obs = self.trajectory[step_idx-1].get("observation", "") if step_idx > 0 else ""
                        prev_visible = self._extract_visible_objects(prev_obs) if prev_obs else []
                        if not any(target_obj.lower() in obj.lower() for obj in prev_visible):
                            milestones.append(f"step {step_idx}: target '{target_obj}' becomes visible")
            
            if len(key_actions) >= 2 or len(milestones) >= 1:
                q = f"Summarize the agent's trajectory from step 0 to step {t} for the task '{self.task}'. What are the key actions and milestones?"
                a = f"From step 0 to step {t}, the agent's trajectory includes: "
                if milestones:
                    a += f"Milestones: {'; '.join(milestones)}. "
                if key_actions:
                    a += f"Key actions: {'; '.join(key_actions[:5])}."
                add("D", f"d1_trajectory_summary_{t}", q, a, subtype="D1")

        # D2: Goal achievement path
        if t >= 2 and self.task:
            target_obj = self._extract_target_object(self.task)
            if target_obj:
                path_steps = []
                for step_idx in range(min(t + 1, len(self.trajectory))):
                    obs = self.trajectory[step_idx].get("observation", "")
                    act = self.trajectory[step_idx].get("action", "")
                    visible = self._extract_visible_objects(obs)
                    
                    is_target_visible = any(target_obj.lower() in obj.lower() for obj in visible)
                    
                    if is_target_visible:
                        path_steps.append(f"step {step_idx}: sees target '{target_obj}'")
                    if act == "pickup" and is_target_visible:
                        path_steps.append(f"step {step_idx}: picks up target '{target_obj}'")
                    if "go to" in self.task.lower() and act in ["forward", "left", "right"] and is_target_visible:
                        path_steps.append(f"step {step_idx}: moves towards target '{target_obj}'")
                
                if len(path_steps) >= 2:
                    q = f"Trace the agent's path to achieving the goal '{self.task}'. What are the key steps and how does the agent progress from step 0 to step {t}?"
                    a = f"To achieve the goal '{self.task}', the agent follows this path: {'; '.join(path_steps)}. This requires examining the trajectory from step 0 to step {t}, tracking when the target '{target_obj}' appears and the sequence of actions taken."
                    add("D", f"d2_goal_path_{t}", q, a, subtype="D2")


    def add_final_QA(self):
        """Select target_count QA pairs from candidates with balanced category distribution.
        
        Quotas: A+B+C = 10, D = 2 (total = 12)
        If target_count is different, adjust proportionally but keep D=2 when possible.
        """
        from collections import defaultdict
        
        candidates_by_type = defaultdict(list)
        
        for candidate in self.candidate_qa_list:
            candidates_by_type[candidate["type"]].append(candidate)

        # Quotas: A+B+C should add up to 10, D should be 2
        # If target_count is not 12, adjust proportionally but prioritize D=2
        if self.target_count >= 12:
            abc_total = 10
            d_total = 2
        else:
            # If target_count < 12, adjust proportionally
            # Keep D at least 1 if possible, rest goes to ABC
            d_total = min(2, max(1, self.target_count // 6))
            abc_total = self.target_count - d_total
        
        total_needed = abc_total + d_total
        
        # Adjust target_count if needed (should be 12, but use what's available)
        if len(self.candidate_qa_list) <= total_needed:
            for candidate in self.candidate_qa_list:
                self.qa_pairs.append({
                    "question": candidate["question"],
                    "answer": candidate["answer"],
                    "type": candidate["type"],
                    "sub_type": candidate.get("subtype", f"{candidate['type']}1")
                })
            return
        
        # Check available categories
        abc_categories = [cat for cat in ["A", "B", "C"] if len(candidates_by_type[cat]) > 0]
        d_available = len(candidates_by_type["D"]) > 0
        
        if not abc_categories and not d_available:
            return
        
        # Set quotas for A, B, C (should add up to 10)
        if len(abc_categories) == 0:
            category_quotas = {"A": 0, "B": 0, "C": 0, "D": min(d_total, len(candidates_by_type["D"]))}
        elif len(abc_categories) == 1:
            category_quotas = {abc_categories[0]: min(abc_total, len(candidates_by_type[abc_categories[0]])), 
                              "A": 0, "B": 0, "C": 0, 
                              "D": min(d_total, len(candidates_by_type["D"])) if d_available else 0}
        elif len(abc_categories) == 2:
            # Distribute 10 between 2 categories: 5 each
            cat1, cat2 = abc_categories[0], abc_categories[1]
            quota1 = min(5, len(candidates_by_type[cat1]))
            quota2 = min(abc_total - quota1, len(candidates_by_type[cat2]))
            category_quotas = {cat1: quota1, cat2: quota2, 
                              "A": 0, "B": 0, "C": 0,
                              "D": min(d_total, len(candidates_by_type["D"])) if d_available else 0}
            # Set unused category to 0
            for cat in ["A", "B", "C"]:
                if cat not in abc_categories:
                    category_quotas[cat] = 0
        else:
            # All three categories available: A=4, B=3, C=3
            category_quotas = {
                "A": min(4, len(candidates_by_type["A"])),
                "B": min(3, len(candidates_by_type["B"])),
                "C": min(3, len(candidates_by_type["C"])),
                "D": min(d_total, len(candidates_by_type["D"])) if d_available else 0
            }
            
            # Adjust if some categories don't have enough candidates
            abc_actual = category_quotas["A"] + category_quotas["B"] + category_quotas["C"]
            if abc_actual < abc_total:
                # Redistribute remaining quota
                remaining = abc_total - abc_actual
                for cat in ["A", "B", "C"]:
                    if remaining <= 0:
                        break
                    available = len(candidates_by_type[cat]) - category_quotas[cat]
                    if available > 0:
                        add_quota = min(remaining, available)
                        category_quotas[cat] += add_quota
                        remaining -= add_quota

        selected_candidates = []
        
        # Select candidates for each category, prioritizing subtype diversity
        for category, quota in category_quotas.items():
            if quota == 0:
                continue
            
            category_candidates = candidates_by_type[category]
            
            # Group by subtype for diversity
            subtype_groups = defaultdict(list)
            for cand in category_candidates:
                subtype = cand.get("subtype", f"{category}1")
                subtype_groups[subtype].append(cand)
            
            selected_for_category = []
            
            # Special handling for D category: ensure D1 and D2 are selected
            if category == "D" and quota >= 2:
                # Try to get both D1 and D2
                if "D1" in subtype_groups and len(selected_for_category) < quota:
                    selected_for_category.append(subtype_groups["D1"][0])
                if "D2" in subtype_groups and len(selected_for_category) < quota:
                    selected_for_category.append(subtype_groups["D2"][0])
            
            # For all categories, prioritize selecting different subtypes
            if len(selected_for_category) < quota:
                # Sort subtypes to prioritize diversity
                available_subtypes = sorted(subtype_groups.keys())
                
                # Round-robin through subtypes to maximize diversity
                subtype_idx = 0
                while len(selected_for_category) < quota and available_subtypes:
                    # Try each subtype in turn
                    for _ in range(len(available_subtypes)):
                        if len(selected_for_category) >= quota:
                            break
                        
                        subtype = available_subtypes[subtype_idx % len(available_subtypes)]
                        subtype_idx += 1
                        
                        # Get candidates of this subtype that haven't been selected
                        available = [c for c in subtype_groups[subtype] if c not in selected_for_category]
                        if available:
                            selected_for_category.append(available[0])
                            break
                    else:
                        # If no more candidates available, break
                        break
            
            # If still need more, fill with any remaining candidates
            if len(selected_for_category) < quota:
                remaining_cands = [c for c in category_candidates if c not in selected_for_category]
                selected_for_category.extend(remaining_cands[:quota - len(selected_for_category)])
            
            selected_candidates.extend(selected_for_category)
        
        # Sort by step for final output
        selected_candidates.sort(key=lambda x: x["step"])

        for candidate in selected_candidates:
            self.qa_pairs.append({
                "question": candidate["question"],
                "answer": candidate["answer"],
                "type": candidate["type"],
                "sub_type": candidate.get("subtype", f"{candidate['type']}1")
            })

    def generate_all(self, target_count: int = 12) -> List[Dict[str, str]]:
        """Generate all QA pairs up to target_count.
        
        Default target_count is 12: A+B+C=10, D=2
        """
        self.target_count = target_count
        
        # Generate QA pairs (including multi-hop questions integrated into A-D categories)
        for t in range(len(self.trajectory)):
            self.maybe_add_per_step(t)
        
        self.add_final_QA()
        
        return self.qa_pairs


def generate_qa_for_trajectory(trajectory_data: Dict[str, Any], target_count: int = 12, seed: Optional[int] = None) -> List[Dict[str, str]]:
    """Generate QA pairs for a single trajectory.
    
    Args:
        trajectory_data: Dictionary containing trajectory data with 'trajectory', 'task', 'task_type'
        target_count: Target number of QA pairs to generate
        seed: Random seed for reproducibility (optional)
        
    Returns:
        List of QA pairs, each with 'question', 'answer', 'type', 'sub_type'
    """
    trajectory = trajectory_data.get("trajectory", [])
    task = trajectory_data.get("task", "")
    task_type = trajectory_data.get("task_type", "")
    
    # Use episode_id as seed if seed not provided and episode_id exists
    if seed is None:
        episode_id = trajectory_data.get("episode_id", "")
        if episode_id:
            # Generate a deterministic seed from episode_id
            seed = hash(episode_id) % (2**31)
    
    generator = BabyAIQAGenerator(trajectory, task, task_type, seed=seed)
    return generator.generate_all(target_count)


if __name__ == "__main__":
    # Test with a sample trajectory
    sample_trajectory = {
        "task": "go to a green ball",
        "task_type": "BabyAI-GoToLocal-v0",
        "trajectory": [
            {
                "turn_idx": 0,
                "action": "forward",
                "observation": "Mission: go to a green ball\nIn your view: 2x a green ball, a green box, a purple box, a purple ball. Walls border the area. You are facing north."
            },
            {
                "turn_idx": 1,
                "action": "forward",
                "observation": "Mission: go to a green ball\nIn your view: 2x a green ball, a green box, a purple box, a purple ball. Walls border the area. You are facing north."
            },
            {
                "turn_idx": 2,
                "action": "right",
                "observation": "Mission: go to a green ball\nIn your view: 2x a green ball, a purple box, a purple ball. Walls border the area. You are facing north."
            }
        ]
    }
    
    qa_pairs = generate_qa_for_trajectory(sample_trajectory, target_count=12)
    print(f"Generated {len(qa_pairs)} QA pairs:")
    for i, qa in enumerate(qa_pairs, 1):
        print(f"\n{i}. [Type {qa['type']}] {qa['question']}")
        print(f"   Answer: {qa['answer']}")
