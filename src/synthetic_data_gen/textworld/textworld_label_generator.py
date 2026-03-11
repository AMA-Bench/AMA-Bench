import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from textworld_utils import parse_task, is_key_action
from textworld_facts_analyzer import FactsTracker, FactsParser


class TextWorldStateTracker:
    """Track state information from TextWorld using facts from environment."""

    def __init__(self, initial_facts: Optional[List[Any]] = None, task: str = ""):
        self.step_states: Dict[int, Dict[str, Any]] = {}
        self.step_actions: Dict[int, str] = {}
        self.step_snapshots: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.task = task
        self.task_type: str = ""
        self.inferred_task_type: str = ""
        self.target_objects: List[str] = []
        self.target_location: str = ""
        self.key_state_action_list: List[Dict[str, Any]] = []
        
        self.object_action_history: Dict[str, List[Tuple[int, str, bool]]] = {}
        self.object_attributes: Dict[str, Dict[str, bool]] = {}
        self.object_locations: Dict[str, str] = {}
        self.container_states: Dict[str, str] = {}
        self.inventory: List[str] = []
        
        if task:
            self.task_type, self.target_objects, self.target_location = parse_task(task)
            self.inferred_task_type = self.task_type
        
        self.facts_tracker: Optional[FactsTracker] = None
        self.use_facts: bool = False
        
        if initial_facts is not None:
            self.facts_tracker = FactsTracker(initial_facts, batch_idx=0)
            self.use_facts = True
    
    def _record_object_action(self, obj: str, step: int, action: str, success: bool):
        if obj not in self.object_action_history:
            self.object_action_history[obj] = []
        self.object_action_history[obj].append((step, action, success))
    
    def update(self, t: int, action: str, observation: str = "", facts: List = None, inventory_text: str = "", location: str = "", admissible_commands: List = None, state: Any = None):
        """Store state and action for each step using facts from environment."""
        if state is not None:
            observation = state.feedback if hasattr(state, 'feedback') else ""
            location = state.location if hasattr(state, 'location') else ""
            inventory = state.inventory if hasattr(state, 'inventory') else ""
            objective = state.objective if hasattr(state, 'objective') else ""
            facts = list(state.facts) if hasattr(state, 'facts') else []
            admissible_commands = state.admissible_commands if hasattr(state, 'admissible_commands') else []
        else:
            inventory = inventory_text
            objective = self.task
            facts = facts if facts is not None else []
            admissible_commands = admissible_commands if admissible_commands is not None else []
        
        self.step_states[t] = {
            "observation": observation,
            "location": location,
            "inventory": inventory,
            "objective": objective,
            "facts": facts,
            "admissible_commands": admissible_commands
        }
        self.step_actions[t] = action
        
        if self.use_facts and self.facts_tracker and facts:
            self.facts_tracker.update(t, facts, action)
            
            fact_state = self.facts_tracker.states.get(t)
            if fact_state:
                prev_inv = list(self.inventory)
                prev_inv_full = list(fact_state.inventory) if t > 0 and (t-1) in self.facts_tracker.states else []
                if (t-1) in self.facts_tracker.states:
                    prev_inv_full = list(self.facts_tracker.states[t-1].inventory)
                
                self.inventory = [FactsParser.extract_object_name(obj) for obj in fact_state.inventory]
                
                for obj_full, loc in fact_state.object_locations.items():
                    obj_name = FactsParser.extract_object_name(obj_full)
                    self.object_locations[obj_name] = FactsParser.extract_object_name(loc) if loc != "inventory" else "inventory"
                
                for container, is_open in fact_state.container_open_states.items():
                    container_name = FactsParser.extract_object_name(container)
                    self.container_states[container_name] = "open" if is_open else "closed"
                
                for obj_full, states in fact_state.object_states.items():
                    obj_name = FactsParser.extract_object_name(obj_full)
                    if obj_name not in self.object_attributes:
                        self.object_attributes[obj_name] = {}
                    
                    if "clean" in states:
                        self.object_attributes[obj_name]["clean"] = True
                    if "cooked" in states:
                        self.object_attributes[obj_name]["cooked"] = True
                    if "sliced" in states:
                        self.object_attributes[obj_name]["sliced"] = True
                    if "raw" in states:
                        self.object_attributes[obj_name]["raw"] = True
                
                success = True
                
                for obj_full in fact_state.inventory:
                    if obj_full not in prev_inv_full:
                        self._record_object_action(obj_full, t, action, success)
                
                for obj_full in prev_inv_full:
                    if obj_full not in fact_state.inventory:
                        self._record_object_action(obj_full, t, action, success)
                
                for obj_full in fact_state.object_states.keys():
                    if t > 0 and (t-1) in self.facts_tracker.states:
                        prev_states = self.facts_tracker.states[t-1].object_states.get(obj_full, set())
                        curr_states = fact_state.object_states[obj_full]
                        if prev_states != curr_states:
                            self._record_object_action(obj_full, t, action, success)
        
        snapshot = {
            "step": t,
            "room": location,
            "inventory": inventory,
            "observation": observation
        }
        self.step_snapshots.append(snapshot)
        
        if action:
            action_verb = action.split()[0].lower() if action.split() else ""
            event = {
                "step": t,
                "action": action,
                "verb": action_verb
            }
            self.events.append(event)
        
        if is_key_action(action, observation, self.task_type, self.target_objects, self.target_location):
            key_entry = {
                "step": t,
                "action": action,
                "state_before": {
                    "location": self.step_states.get(t-1, {}).get("location", "unknown"),
                    "inventory": self.step_states.get(t-1, {}).get("inventory", "unknown")
                },
                "state_after": {
                    "location": location,
                    "inventory": inventory
                },
                "observation": observation[:200] + "..." if len(observation) > 200 else observation,
                "reason": f"Key action for {self.task_type} task"
            }
            self.key_state_action_list.append(key_entry)
    
    def get_object_action_history(self, obj: str) -> List[Tuple[int, str, bool]]:
        return self.object_action_history.get(obj, [])
    
    def get_key_state_actions(self) -> List[Dict[str, Any]]:
        return self.key_state_action_list
    
    def get_state_description(self, step: int, focus_objects: Optional[List[str]] = None) -> str:
        if self.use_facts and self.facts_tracker:
            return self.facts_tracker.format_state_for_answer(step, focus_objects)
        return ""
    
    def get_changes_description(self, from_step: int, to_step: int) -> str:
        if self.use_facts and self.facts_tracker:
            return self.facts_tracker.format_changes_for_answer(from_step, to_step)
        return ""
    
    def get_detailed_changes(self, from_step: int, to_step: int) -> Dict[str, Any]:
        if self.use_facts and self.facts_tracker:
            return self.facts_tracker.get_state_changes(from_step, to_step)
        return {}
    
    def get_current_object_state(self, obj: str) -> Dict[str, Any]:
        state = {
            "object": obj,
            "exists": obj in self.object_locations or obj in self.inventory,
            "in_inventory": obj in self.inventory,
            "location": self.object_locations.get(obj, "unknown"),
            "attributes": {
                "is_clean": self.object_attributes.get(obj, {}).get("clean", False),
                "is_cooked": self.object_attributes.get(obj, {}).get("cooked", False),
                "is_sliced": self.object_attributes.get(obj, {}).get("sliced", False),
                "is_raw": self.object_attributes.get(obj, {}).get("raw", False)
            }
        }
        return state
    
    def get_task_key_state(self) -> Dict[str, Any]:
        """Get key state based on task type from facts."""
        task_state = {
            "task_type": self.inferred_task_type,
            "inventory": self.inventory,
            "object_locations": self.object_locations,
            "object_attributes": self.object_attributes
        }
        
        cooked_objs = [obj for obj, attrs in self.object_attributes.items() if attrs.get("cooked")]
        clean_objs = [obj for obj, attrs in self.object_attributes.items() if attrs.get("clean")]
        sliced_objs = [obj for obj, attrs in self.object_attributes.items() if attrs.get("sliced")]
        
        if cooked_objs:
            task_state["cooked_objects"] = cooked_objs
        if clean_objs:
            task_state["clean_objects"] = clean_objs
        if sliced_objs:
            task_state["sliced_objects"] = sliced_objs
        
        return task_state
    
    def get_task_progress_summary(self) -> str:
        if not self.key_state_action_list:
            return "No key actions identified yet"
        
        summary_parts = [
            f"Task type: {self.task_type}",
            f"Target objects: {', '.join(self.target_objects) if self.target_objects else 'none'}",
            f"Target location: {self.target_location if self.target_location else 'none'}",
            f"Key actions taken: {len(self.key_state_action_list)}"
        ]
        
        return "; ".join(summary_parts)


class TextWorldQAGenerator:
    """Generate QA pairs based on state attributes from TextWorld."""

    def __init__(self, tracker: TextWorldStateTracker, trajectory: List[Dict[str, Any]], task: str = ""):
        self.tracker = tracker
        self.trajectory = trajectory
        self.task = task
        self.qa_pairs: List[Dict[str, str]] = []
        self.candidate_qa_list: List[Dict[str, Any]] = []
        self.used_keys: set = set()
        self.target_count: int = 10
        
        self.MAX_T_ONLY_SUBTYPES = {
            "C5",
            "D1",
            "D2",
            "D3",
            "D4",
        }

    def _add_candidate(self, key: str, question: str, answer: str, qa_type: str, step: int) -> bool:
        """Add a candidate QA pair to the temporary list.
        
        For subtypes in MAX_T_ONLY_SUBTYPES, newer QA (larger time step) replaces older QA.
        """
        
        normalized_type = next(
            (ch for ch in qa_type if ch.isupper()),
            qa_type[:1].upper() if qa_type else ""
        )
        
        new_candidate = {
            "question": question,
            "answer": answer,
            "type": normalized_type,
            "subtype": qa_type,
            "step": step,
            "key": key
        }
        
        if qa_type in self.MAX_T_ONLY_SUBTYPES:
            removed_candidates = [
                c for c in self.candidate_qa_list 
                if c.get("subtype") == qa_type
            ]
            for removed in removed_candidates:
                self.used_keys.discard(removed["key"])
            
            self.candidate_qa_list = [
                c for c in self.candidate_qa_list 
                if c.get("subtype") != qa_type
            ]
        
        self.used_keys.add(key)
        self.candidate_qa_list.append(new_candidate)
        return True

    def maybe_add_per_step(self, t: int):
        """Collect candidate QA pairs for current step using facts-based tracking.

        Question types:
        - A1-A4: Temporal action reasoning
        - B1-B3: State dependency and precondition reasoning
        - C1-C6: Complex state tracking
        - D1, D4: Key action and progress tracking
        """
        if not self.tracker.use_facts or not self.tracker.facts_tracker:
            return
        
        if t not in self.tracker.step_states:
            return

        state = self.tracker.step_states[t]
        action = self.tracker.step_actions.get(t, "")

        def add(kind: str, key: str, q: str, a: str) -> None:
            self._add_candidate(key, q, a, qa_type=kind, step=t)

        observation = state["observation"]
        location = state["location"]
        inventory = state["inventory"]
        objective = state["objective"]
        facts = state["facts"]
        admissible_commands = state["admissible_commands"]

        # =========================================
        # A: Temporal Action Reasoning
        # =========================================

        # A1: Actions Between Steps
        # Template: What actions were performed between step X and step Y?
        if t >= 3:
            # Select two steps with a reasonable gap
            lookback = min(t - 1, random.randint(2, min(7, t - 1)))
            step_start = max(1, t - lookback)
            step_end = t

            actions_between = []
            for step in range(step_start, step_end + 1):
                act = self.tracker.step_actions.get(step, "")
                if act:
                    actions_between.append(f"at step {step}, {act}")

            if len(actions_between) >= 2:
                q = f"What actions were performed between step {step_start} and step {step_end}?"
                a = f"Between step {step_start} and step {step_end}, the agent performed the following actions: {'; '.join(actions_between)}."
                add("A1", f"a1_actions_between_{step_start}_{step_end}_{t}", q, a)


        # A2: Object-Specific Action History from facts
        if t >= 2:
            for obj_name, history in self.tracker.object_action_history.items():
                if len(history) >= 2:
                    actions_before = []
                    for step, act, success in history:
                        if step <= t and success:
                            verb = act.split()[0].lower() if act.split() else ""
                            if verb != "go":
                                actions_before.append(f"step {step}: {act}")

                    if len(actions_before) >= 2:
                        query_step = t + 1
                        q = f"Before step {query_step}, which actions were performed on {obj_name} and at which steps ((excluding 'go to' actions))?"
                        a = f"Before step {query_step}, actions performed on {obj_name}: {'; '.join(actions_before)}."
                        add("A2", f"a2_object_history_{obj_name}_{t}", q, a)
                        break

        # A3: Objects Modified By Action Type from facts
        if t >= 2:
            for step_num in range(1, t + 1):
                if step_num not in self.tracker.facts_tracker.states:
                    continue
                fact_state = self.tracker.facts_tracker.states[step_num]
                if step_num > 0 and (step_num - 1) in self.tracker.facts_tracker.states:
                    prev_state = self.tracker.facts_tracker.states[step_num - 1]
                    
                    changed_objs = []
                    for obj_full, curr_states in fact_state.object_states.items():
                        prev_states = prev_state.object_states.get(obj_full, set())
                        if prev_states != curr_states:
                            state_change = list(curr_states - prev_states)
                            if state_change:
                                changed_objs.append((obj_full, step_num, state_change[0]))
            
            if len(changed_objs) >= 2:
                query_step = t + 1
                q = f"Before step {query_step}, which objects had state changes and at which steps?"
                answer_parts = [f"{obj} at step {step} (became {state})" for obj, step, state in changed_objs[:5]]
                a = f"Before step {query_step}, the following objects had state changes: {'; '.join(answer_parts)}."
                add("A3", f"a3_objects_modified_{t}", q, a)

        # A4: State Transition from facts
        if t >= 3:
            max_lookback = min(7, t - 1)
            lookback = min(t - 1, random.randint(3, max_lookback)) if max_lookback >= 3 else t - 1
            start_step = max(1, t - lookback)
            
            if start_step in self.tracker.step_actions:
                actions_sequence = []
                for step in range(start_step, t + 1):
                    if step in self.tracker.step_actions:
                        actions_sequence.append(f"step {step}: {self.tracker.step_actions[step]}")
                
                if len(actions_sequence) >= 2:
                    start_state_desc = self.tracker.get_state_description(start_step)
                    changes_desc = self.tracker.get_changes_description(start_step, t)
                    
                    q = f"When the agent is at the state: {start_state_desc[:200]}, what were the subsequent actions and state changes until step {t}?"
                    a = f"The subsequent actions were: {'; '.join(actions_sequence)}. State changes: {changes_desc}"
                    add("A4", f"a4_state_transition_{start_step}_{t}", q, a)

        # =========================================
        # B: State Dependency and Precondition Reasoning from facts
        # =========================================

        # B1: Action-to-State-Change Causal Relationship from facts
        if t >= 3:
            max_lookback = min(7, t - 1)
            lookback = min(t - 1, random.randint(3, max_lookback)) if max_lookback >= 3 else t - 1
            start_step = max(1, t - lookback)
            
            if start_step in self.tracker.step_actions:
                changes = self.tracker.get_detailed_changes(start_step, t)
                has_changes = any([changes.get("moved_objects"), changes.get("state_changes"),
                                  changes.get("inventory_changes", {}).get("added"),
                                  changes.get("inventory_changes", {}).get("removed")])
                
                if has_changes:
                    causal_pairs = []
                    
                    for step in range(start_step + 1, t + 1):
                        if step in self.tracker.step_actions:
                            action_taken = self.tracker.step_actions[step]
                            step_changes = self.tracker.get_detailed_changes(step - 1, step)
                            
                            change_desc = []
                            inv_changes = step_changes.get("inventory_changes", {})
                            if inv_changes.get("added"):
                                for obj in inv_changes["added"]:
                                    change_desc.append(f"{obj} added to inventory")
                            if inv_changes.get("removed"):
                                for obj in inv_changes["removed"]:
                                    change_desc.append(f"{obj} removed from inventory")
                            
                            if step_changes.get("moved_objects"):
                                for move_info in step_changes["moved_objects"]:
                                    obj = FactsParser.extract_object_name(move_info["object"])
                                    old_loc = FactsParser.extract_object_name(move_info["from"]) if move_info["from"] != "inventory" else "inventory"
                                    new_loc = FactsParser.extract_object_name(move_info["to"]) if move_info["to"] != "inventory" else "inventory"
                                    change_desc.append(f"{obj} moved from {old_loc} to {new_loc}")
                            
                            if step_changes.get("state_changes"):
                                for state_change in step_changes["state_changes"]:
                                    obj = FactsParser.extract_object_name(state_change["object"])
                                    if state_change.get("added_states"):
                                        for state in state_change["added_states"]:
                                            change_desc.append(f"{obj} became {state}")
                            
                            if change_desc:
                                causal_pairs.append(f"step {step}: '{action_taken}' caused [{'; '.join(change_desc)}]")
                    
                    if len(causal_pairs) >= 2:
                        q = f"From step {start_step} to step {t}, what actions made the environment changes and what were the environment changes?"
                        a = f"Action-environment changes: {' | '.join(causal_pairs[:8])}"
                        add("B1", f"b1_action_for_transition_{start_step}_{t}", q, a)

        # B2: Multi-Step Action State Change Prediction from facts
        if t >= 4 and t in self.tracker.step_actions:
            num_steps = min(3, t - 1)
            start_step = t - num_steps
            
            actions_sequence = []
            for step in range(start_step + 1, t + 1):
                if step in self.tracker.step_actions:
                    actions_sequence.append(self.tracker.step_actions[step])
            
            if len(actions_sequence) >= 2:
                step_changes = self.tracker.get_detailed_changes(start_step, t)
                has_changes = any([step_changes.get("moved_objects"), step_changes.get("state_changes"),
                                  step_changes.get("inventory_changes", {}).get("added"),
                                  step_changes.get("inventory_changes", {}).get("removed")])
                
                if has_changes:
                    change_desc = []
                    inv_changes = step_changes.get("inventory_changes", {})
                    if inv_changes.get("added"):
                        change_desc.append(f"Inventory changes: added {', '.join(inv_changes['added'])}")
                    if inv_changes.get("removed"):
                        change_desc.append(f"Inventory changes: removed {', '.join(inv_changes['removed'])}")
                    
                    if step_changes.get("moved_objects"):
                        moves = []
                        for move_info in step_changes["moved_objects"]:
                            obj = FactsParser.extract_object_name(move_info["object"])
                            old_loc = FactsParser.extract_object_name(move_info["from"]) if move_info["from"] != "inventory" else "inventory"
                            new_loc = FactsParser.extract_object_name(move_info["to"]) if move_info["to"] != "inventory" else "inventory"
                            moves.append(f"{obj} moved from {old_loc} to {new_loc}")
                        if moves:
                            change_desc.append(f"Object movements: {'; '.join(moves)}")
                    
                    if step_changes.get("state_changes"):
                        states = []
                        for state_change in step_changes["state_changes"]:
                            obj = FactsParser.extract_object_name(state_change["object"])
                            if state_change.get("added_states"):
                                states.append(f"{obj} became {', '.join(state_change['added_states'])}")
                        if states:
                            change_desc.append(f"State changes: {'; '.join(states)}")
                    
                    if change_desc:
                        prev_inv = self.tracker.facts_tracker.states[start_step].inventory if start_step in self.tracker.facts_tracker.states else set()
                        prev_inv_desc = f"Inventory: {', '.join([FactsParser.extract_object_name(o) for o in prev_inv])}" if prev_inv else "Inventory: empty"
                        
                        actions_str = "', '".join(actions_sequence)
                        q = f"At step {start_step} with {prev_inv_desc}, if the agent executes actions ['{actions_str}'] sequentially, what state changes will occur? Please list all the objects state changes and inventory changes."
                        a = f"After executing these actions: {' | '.join(change_desc)}"
                        add("B2", f"b2_resulting_state_{start_step}_{t}", q, a)

        # B3: Action Feasibility from facts
        if t >= 2 and admissible_commands and len(admissible_commands) > 1:
            candidate_action = random.choice([a for a in admissible_commands if a != action]) if len(admissible_commands) > 1 else admissible_commands[0]
            can_execute = True
            reason = ""

            tokens = candidate_action.split()
            verb = tokens[0].lower() if tokens else ""

            fact_state = self.tracker.facts_tracker.states.get(t)
            if fact_state:
                if verb in ["put", "drop"]:
                    if not fact_state.inventory:
                        can_execute = False
                        reason = "the inventory is empty, so there is no object to put"
                    elif len(tokens) >= 2:
                        obj_pattern = tokens[1]
                        inv_names = [FactsParser.extract_object_name(o) for o in fact_state.inventory]
                        if obj_pattern not in inv_names:
                            can_execute = False
                            reason = f"{obj_pattern} is not in inventory"

                elif verb == "take":
                    if len(tokens) >= 2:
                        obj_pattern = tokens[1]
                        inv_names = [FactsParser.extract_object_name(o) for o in fact_state.inventory]
                        if obj_pattern in inv_names:
                            can_execute = False
                            reason = f"{obj_pattern} is already in inventory"

                elif verb == "open":
                    if len(tokens) >= 2:
                        container = tokens[1]
                        if container in [FactsParser.extract_object_name(r) for r, is_open in fact_state.container_open_states.items() if is_open]:
                            can_execute = False
                            reason = f"{container} is already open"

                elif verb == "close":
                    if len(tokens) >= 2:
                        container = tokens[1]
                        if container in [FactsParser.extract_object_name(r) for r, is_open in fact_state.container_open_states.items() if not is_open]:
                            can_execute = False
                            reason = f"{container} is already closed"

            state_desc = self.tracker.get_state_description(t)
            
            q = f"At step {t} with state: {state_desc[:150]}, can the agent perform \"{candidate_action}\"?"
            if can_execute:
                a = f"Yes, the agent can perform this action given the current state."
            else:
                a = f"No, the agent cannot perform this action because {reason}."
                add("B3", f"b3_action_feasibility_{t}", q, a)

        # =========================================
        # C: Complex State Tracking from facts
        # =========================================

        # C1: Inventory Change Timeline from facts
        if t >= 5:
            inv_changes = []
            prev_inv = set()
            
            for step in range(1, t + 1):
                if step in self.tracker.facts_tracker.states:
                    curr_inv = self.tracker.facts_tracker.states[step].inventory
                    
                    added = curr_inv - prev_inv
                    for obj in added:
                        inv_changes.append(f"At step {step}, {obj} was added")
                    
                    removed = prev_inv - curr_inv
                    for obj in removed:
                        inv_changes.append(f"At step {step}, {obj} was removed")
                    
                    prev_inv = curr_inv

            if len(inv_changes) >= 2:
                q = f"What changes occurred to the inventory throughout the trajectory and at which steps?"
                a = f"The inventory changed as follows: Initially empty. {'. '.join(inv_changes)}."
                add("C1", f"c1_inventory_timeline_{t}", q, a)

        # C2: Object State Query After Changes from facts
        if t >= 4:
            for obj_name, history in self.tracker.object_action_history.items():
                if len(history) >= 1:
                    change_steps = [step for step, _, success in history if success and step <= t]

                    if len(change_steps) >= 1:
                        query_step = t
                        state_desc = self.tracker.get_state_description(query_step, focus_objects=[obj_name])
                        changes_desc = self.tracker.get_changes_description(0, query_step)
                        
                        change_details = []
                        for step, act, success in history:
                            if success and step <= t:
                                change_details.append(f"step {step} ({act})")
                        
                        all_changes = ', '.join(change_details) if change_details else ', '.join([f"step {s}" for s in change_steps])

                        q = f"What is the state of {obj_name} at step {query_step}? When did this change occur and what were the prior whole changes history?"
                        a = f"At step {query_step}, state: {state_desc}. Changes history: {changes_desc}. Specific changes to {obj_name} occurred at: {all_changes}."
                        
                        add("C2", f"c2_object_state_query_{obj_name}_{t}", q, a)
                        break

        # C3: Container State Evolution from facts
        if t >= 3:
            container_changes_tracked = {}
            
            for step in range(1, t + 1):
                if step in self.tracker.facts_tracker.states:
                    fact_state = self.tracker.facts_tracker.states[step]
                    
                    if step > 0 and (step - 1) in self.tracker.facts_tracker.states:
                        prev_state = self.tracker.facts_tracker.states[step - 1]
                        
                        all_containers = set(fact_state.container_open_states.keys()) | set(prev_state.container_open_states.keys())
                        for container in all_containers:
                            prev_open = prev_state.container_open_states.get(container)
                            curr_open = fact_state.container_open_states.get(container)
                            if prev_open != curr_open and curr_open is not None:
                                if container not in container_changes_tracked:
                                    container_changes_tracked[container] = []
                                action_desc = "opened" if curr_open else "closed"
                                container_changes_tracked[container].append((step, action_desc))
                        
                        for container, contents in fact_state.container_contents.items():
                            prev_contents = prev_state.container_contents.get(container, set())
                            added = contents - prev_contents
                            removed = prev_contents - contents
                            
                            for obj in added:
                                if container not in container_changes_tracked:
                                    container_changes_tracked[container] = []
                                container_changes_tracked[container].append((step, f"{obj} was added"))
                            
                            for obj in removed:
                                if container not in container_changes_tracked:
                                    container_changes_tracked[container] = []
                                container_changes_tracked[container].append((step, f"{obj} was removed"))
            
            if container_changes_tracked:
                best_container = max(container_changes_tracked.items(), key=lambda x: len(x[1]))
                container_full_name, changes = best_container
                
                if len(changes) >= 2:
                    change_desc = '; '.join([f"step {s}: {desc}" for s, desc in changes])
                    
                    current_state = self.tracker.facts_tracker.states.get(t)
                    current_contents = []
                    if current_state and container_full_name in current_state.container_contents:
                        current_contents = list(current_state.container_contents[container_full_name])
                    
                    contents_desc = f" At step {t}, it contains: {', '.join(current_contents)}." if current_contents else f" At step {t}, it is empty or not accessible."
                    
                    q = f"How did the state of {container_full_name} change throughout the trajectory, including what objects were placed in or removed from it?"
                    a = f"{container_full_name.capitalize()} evolution: {change_desc}.{contents_desc}"
                    add("C3", f"c3_container_evolution_{container_full_name}_{t}", q, a)

        # C4: Multi-Object Relationship Tracking from facts
        if t >= 2 and t in self.tracker.facts_tracker.states:
            fact_state = self.tracker.facts_tracker.states[t]
            objects_to_query = list(fact_state.object_locations.items())[:3]
            
            if len(objects_to_query) >= 2:
                relationships = []
                for obj_full, loc in objects_to_query:
                    if loc == "inventory":
                        loc_desc = "in inventory"
                    elif loc.startswith('loc ') or loc.startswith('room '):
                        loc_desc = f"at {loc}"
                    else:
                        loc_desc = f"in/on {loc}"
                    
                    if obj_full in fact_state.container_contents and fact_state.container_contents[obj_full]:
                        contents = list(fact_state.container_contents[obj_full])
                        relationships.append(f"{obj_full} is {loc_desc} and contains {', '.join(contents[:3])}")
                    else:
                        relationships.append(f"{obj_full} is {loc_desc}")
                
                obj_names = [obj for obj, _ in objects_to_query]
                q = f"What is the location of {', '.join(obj_names)} at step {t}?"
                a = f"At step {t}: {'; '.join(relationships)}."
                add("C4", f"c4_relationships_{t}", q, a)

        # C5: Object Attribute Timeline from facts
        if t >= 3:
            objects_with_attrs = {}
            
            for step in range(1, t + 1):
                if step in self.tracker.facts_tracker.states and step > 0:
                    prev_state = self.tracker.facts_tracker.states.get(step - 1)
                    curr_state = self.tracker.facts_tracker.states[step]
                    
                    for obj_full, curr_states in curr_state.object_states.items():
                        if obj_full not in objects_with_attrs:
                            objects_with_attrs[obj_full] = []
                        
                        prev_states = prev_state.object_states.get(obj_full, set()) if prev_state else set()
                        new_states = curr_states - prev_states
                        
                        for new_state in new_states:
                            action_taken = self.tracker.step_actions.get(step, "unknown")
                            objects_with_attrs[obj_full].append(f"at step {step}, became {new_state} (action: {action_taken})")
            
            for obj_full, attr_changes in objects_with_attrs.items():
                if len(attr_changes) >= 1:
                    current_state = self.tracker.facts_tracker.states.get(t)
                    current_attrs = []
                    if current_state and obj_full in current_state.object_states:
                        current_attrs = sorted(list(current_state.object_states[obj_full]))
                    
                    q = f"How did the attributes of {obj_full} change during the task until step {t}?"
                    a = f"{obj_full.capitalize()} started from initial states. {'; '.join(attr_changes).capitalize()}. Current attributes: {', '.join(current_attrs)}."
                    add("C5", f"c5_attribute_timeline_{obj_full}_{t}", q, a)
                    break

        # C6: Location History of Multiple Objects from facts
        if t >= 4:
            objects_with_changes = {}
            
            for step in range(1, t + 1):
                if step in self.tracker.facts_tracker.states:
                    fact_state = self.tracker.facts_tracker.states[step]
                    for obj_full, loc in fact_state.object_locations.items():
                        if obj_full not in objects_with_changes:
                            objects_with_changes[obj_full] = []
                        
                        if step > 0 and (step - 1) in self.tracker.facts_tracker.states:
                            prev_loc = self.tracker.facts_tracker.states[step - 1].object_locations.get(obj_full)
                            if prev_loc != loc:
                                loc_desc = "inventory" if loc == "inventory" else loc
                                objects_with_changes[obj_full].append((step, loc_desc))
            
            objs_with_history = [(obj, history) for obj, history in objects_with_changes.items() if len(history) >= 1]
            if len(objs_with_history) >= 2:
                obj1, obj1_locs = objs_with_history[0]
                obj2, obj2_locs = objs_with_history[1]
                
                obj1_desc = ', '.join([f"step {s}: {loc}" for s, loc in obj1_locs])
                obj2_desc = ', '.join([f"step {s}: {loc}" for s, loc in obj2_locs])
                
                q = f"Until step {t}, what were the location histories of {obj1} and {obj2}?"
                a = f"{obj1.capitalize()}: {obj1_desc if obj1_desc else 'no location changes'}. {obj2.capitalize()}: {obj2_desc if obj2_desc else 'no location changes'}."
                add("C6", f"c6_location_history_{obj1}_{obj2}_{t}", q, a)

        # =========================================
        # D: Key Action and Progress Tracking
        # =========================================

        # D1: Key action identification from tracker
        if t >= 2:
            key_actions = self.tracker.get_key_state_actions()
            if key_actions:
                recent_key_actions = [ka for ka in key_actions if ka["step"] <= t]
                if len(recent_key_actions) >= 1:
                    key_action_summary = [f"Step {ka['step']}: {ka['action']}" for ka in recent_key_actions[-5:]]
                    q = f"Until step {t}, what are the key actions that have been critical for task progress?"
                    a = "; ".join(key_action_summary)
                    add("D1", f"d1_key_actions_{t}", q, a)

        # D2: Complete action frequency
        if t >= 3:
            action_counts = {}
            for step in range(0, t + 1):
                if step in self.tracker.step_actions:
                    action_taken = self.tracker.step_actions[step]
                    if action_taken:
                        action_counts[action_taken] = action_counts.get(action_taken, 0) + 1
            
            if len(action_counts) >= 2:
                sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
                action_summary = [f"'{action_taken}' ({count} times)" for action_taken, count in sorted_actions[:15]]
                q = f"Until step {t}, what actions has the agent performed and how frequently?"
                a = f"Actions performed: {', '.join(action_summary)}"
                add("D2", f"d2_action_frequency_{t}", q, a)

        # D3: Object interaction summary from facts
        if t >= 2 and len(self.tracker.object_action_history) >= 2:
            interacted_objects = []
            for obj_name, history in self.tracker.object_action_history.items():
                actions_on_obj = [act for step, act, success in history if step <= t and success and not act.lower().startswith("go to")]
                if actions_on_obj:
                    interacted_objects.append(f"{obj_name} ({len(actions_on_obj)} interactions)")
            
            if len(interacted_objects) >= 2:
                q = f"At step {t}, what objects has the agent interacted with so far (excluding 'go to' actions) and how many times they were interacted with?"
                a = f"The agent has interacted with: {', '.join(interacted_objects)}"
                add("D3", f"d3_object_interactions_{t}", q, a)

        # D4: Task progress based on task type from facts
        if t >= 3 and self.tracker.inferred_task_type:
            task_state = self.tracker.get_task_key_state()
            phases = []
            
            if task_state.get("inventory"):
                phases.append("object acquisition")
            
            if task_state.get("cooked_objects"):
                phases.append("cooking")
            if task_state.get("clean_objects"):
                phases.append("cleaning")
            if task_state.get("sliced_objects"):
                phases.append("slicing")
            
            if len(phases) >= 1:
                q = f"At step {t}, which phases have been completed for the {self.tracker.inferred_task_type} task?"
                details = [f"Inventory: {task_state.get('inventory', [])}"]
                if task_state.get("cooked_objects"):
                    details.append(f"Cooked objects: {task_state.get('cooked_objects', [])}")
                if task_state.get("clean_objects"):
                    details.append(f"Clean objects: {task_state.get('clean_objects', [])}")
                if task_state.get("sliced_objects"):
                    details.append(f"Sliced objects: {task_state.get('sliced_objects', [])}")
                
                a = f"Completed phases: {', '.join(phases)}. {'. '.join(details)}."
                add("D4", f"d4_task_progress_{t}", q, a)

    def add_final_QA(self):
        """Select target_count QA pairs from candidates with maximum subtype diversity.

        Strategy:
        1. Ensure A, B, C, D categories each have proportional representation
        2. Within each category, maximize subtype diversity - each subtype appears at most once
        3. Only reuse subtypes if necessary to meet quota
        4. Maintain temporal distribution across steps
        """
        if len(self.candidate_qa_list) <= self.target_count:
            for candidate in self.candidate_qa_list:
                self.qa_pairs.append({
                    "question": candidate["question"],
                    "answer": candidate["answer"],
                    "type": candidate["type"],
                    "sub_type": candidate["subtype"]
                })
            return

        from collections import defaultdict
        
        candidates_by_subtype = defaultdict(list)
        candidates_by_category = defaultdict(list)
        
        for candidate in self.candidate_qa_list:
            qa_subtype = candidate.get("subtype") or candidate["type"]
            candidates_by_subtype[qa_subtype].append(candidate)
            category = (candidate["type"][0] if candidate["type"]
                        else (qa_subtype[0] if qa_subtype else "other"))
            candidates_by_category[category].append(candidate)

        category_quotas = {"A": 2, "B": 3, "C": 3, "D": 2}
        
        available_categories = []
        for cat in ["A", "B", "C", "D"]:
            if len(candidates_by_category[cat]) > 0:
                available_categories.append(cat)
            else:
                category_quotas[cat] = 0
        
        if not available_categories:
            return
        
        total_quota = sum(category_quotas.values())
        if total_quota != self.target_count and available_categories:
            base_quota = self.target_count // len(available_categories)
            remainder = self.target_count % len(available_categories)
            for i, cat in enumerate(available_categories):
                category_quotas[cat] = base_quota + (1 if i < remainder else 0)
                category_quotas[cat] = min(category_quotas[cat], len(candidates_by_category[cat]))

        selected_candidates = []
        
        for category, quota in category_quotas.items():
            if quota == 0:
                continue
                
            subtypes_in_category = {}
            for subtype, candidates in candidates_by_subtype.items():
                if subtype.startswith(category):
                    subtypes_in_category[subtype] = candidates
            
            if not subtypes_in_category:
                continue
            
            sorted_subtypes = sorted(subtypes_in_category.keys())
            
            selected_for_category = []
            
            for subtype in sorted_subtypes:
                if len(selected_for_category) >= quota:
                    break
                    
                candidates_of_subtype = subtypes_in_category[subtype]
                
                available = [c for c in candidates_of_subtype if c not in selected_for_category]
                
                if available:
                    available.sort(key=lambda x: x["step"])
                    selected_idx = len(available) // 2
                    selected_for_category.append(available[selected_idx])
            
            subtype_pick_counts = defaultdict(int)
            for c in selected_for_category:
                subtype_pick_counts[c.get("subtype") or c["type"]] += 1
            
            while len(selected_for_category) < quota:
                added_any = False
                
                for subtype in sorted_subtypes:
                    if len(selected_for_category) >= quota:
                        break
                    
                    available = [c for c in subtypes_in_category[subtype] 
                                if c not in selected_for_category]
                    
                    if subtype.upper() == "A1" and subtype_pick_counts["A1"] >= 1:
                        continue
                    
                    if not available:
                        continue
                    
                    available.sort(key=lambda x: x["step"])
                    
                    round_num = subtype_pick_counts[subtype]
                    if round_num == 0:
                        selected_idx = len(available) // 3
                    elif round_num == 1:
                        selected_idx = 2 * len(available) // 3
                    else:
                        selected_idx = len(available) // 2
                    
                    selected_idx = min(selected_idx, len(available) - 1)
                    selected_for_category.append(available[selected_idx])
                    subtype_pick_counts[subtype] += 1
                    added_any = True
                
                if not added_any:
                    break
            
            selected_candidates.extend(selected_for_category[:quota])
        
        if len(selected_candidates) < self.target_count:
            remaining_candidates = [c for c in self.candidate_qa_list if c not in selected_candidates]
            needed = self.target_count - len(selected_candidates)
            
            subtype_counts = defaultdict(int)
            for c in selected_candidates:
                subtype_counts[c["type"]] += 1
            
            remaining_sorted = sorted(remaining_candidates,
                                     key=lambda x: (subtype_counts[x["type"]], x["step"]))
            selected_candidates.extend(remaining_sorted[:needed])
        
        selected_candidates.sort(key=lambda x: x["step"])

        for candidate in selected_candidates[:self.target_count]:
            qa_type = candidate["type"]
            normalized_type = next(
                (ch for ch in qa_type if ch.isupper()),
                qa_type[:1].upper() if qa_type else ""
            )
            self.qa_pairs.append({
                "question": candidate["question"],
                "answer": candidate["answer"],
                "type": qa_type,
                "sub_type": candidate["subtype"]
            })

    def finalize(self) -> Dict[str, Any]:
        """Return final list of QA pairs and summary after selecting from candidates.
        
        Returns:
            Dict with 'qa_pairs'
        """
        if self.candidate_qa_list:
            self.add_final_QA()
        
        result = {
            "qa_pairs": list(self.qa_pairs),
        }
        
        return result
    
    def generate_all(self, target_count: int = 10) -> List[Dict[str, str]]:
        """Generate all QA pairs up to target_count."""
        self.target_count = target_count
        
        for t in range(len(self.tracker.step_states)):
            self.maybe_add_per_step(t)
        
        self.add_final_QA()
        
        return self.qa_pairs
