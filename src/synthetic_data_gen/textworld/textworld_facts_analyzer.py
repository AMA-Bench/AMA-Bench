#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class FactState:
    step: int
    object_locations: Dict[str, str] = field(default_factory=dict)
    object_attributes: Dict[str, Set[str]] = field(default_factory=dict)
    container_contents: Dict[str, Set[str]] = field(default_factory=dict)
    container_open_states: Dict[str, bool] = field(default_factory=dict)
    object_states: Dict[str, Set[str]] = field(default_factory=dict)
    inventory: Set[str] = field(default_factory=set)
    all_entities: Set[str] = field(default_factory=set)
    raw_facts: List[str] = field(default_factory=list)


class FactsParser:
    
    @staticmethod
    def parse_predicate(fact_str: str) -> Optional[Tuple[str, List[str]]]:
        fact_str = str(fact_str).strip()
        
        match = re.match(r'(\w+)\((.*)\)', fact_str)
        if not match:
            return None
        
        predicate = match.group(1).lower()
        args_str = match.group(2)
        
        args = []
        for arg in args_str.split(','):
            arg = arg.strip()
            arg = re.sub(r'\s*:\s*\w+$', '', arg)
            if arg:
                args.append(arg)
        
        return (predicate, args)
    
    @staticmethod
    def extract_object_name(entity: str) -> str:
        match = re.match(r'(.+?)\s+\d+$', entity)
        if match:
            return match.group(1).strip()
        return entity.strip()
    
    @staticmethod
    def parse_facts(facts: List[Any], step: int = 0) -> FactState:
        state = FactState(step=step)
        
        for fact in facts:
            fact_str = str(fact)
            state.raw_facts.append(fact_str)
            
            parsed = FactsParser.parse_predicate(fact_str)
            if not parsed:
                continue
            
            predicate, args = parsed
            
            for arg in args:
                if not arg.startswith('loc ') and not arg.startswith('room '):
                    state.all_entities.add(arg)
            
            if predicate == "at" and len(args) == 2:
                obj, loc = args
                state.object_locations[obj] = loc
            
            elif predicate == "in" and len(args) == 2:
                obj, container = args
                if container not in state.container_contents:
                    state.container_contents[container] = set()
                state.container_contents[container].add(obj)
                state.object_locations[obj] = container
            
            elif predicate == "on" and len(args) == 2:
                obj, surface = args
                if surface not in state.container_contents:
                    state.container_contents[surface] = set()
                state.container_contents[surface].add(obj)
                state.object_locations[obj] = surface
            
            elif predicate == "inventory" and len(args) == 1:
                obj = args[0]
                state.inventory.add(obj)
                state.object_locations[obj] = "inventory"
            
            elif predicate == "open" and len(args) == 1:
                container = args[0]
                state.container_open_states[container] = True
            
            elif predicate == "closed" and len(args) == 1:
                container = args[0]
                state.container_open_states[container] = False
            
            elif predicate in ["cooked", "burned", "raw", "roasted", "fried", 
                              "sliced", "diced", "chopped", "peeled",
                              "clean", "dirty", "wet", "dry",
                              "locked", "unlocked", "edible", "inedible"] and len(args) == 1:
                obj = args[0]
                if obj not in state.object_states:
                    state.object_states[obj] = set()
                state.object_states[obj].add(predicate)
            
            elif predicate in ["cookable", "sliceable", "openable", "lockable", 
                              "takeable", "eatable", "drinkable"] and len(args) == 1:
                obj = args[0]
                if obj not in state.object_attributes:
                    state.object_attributes[obj] = set()
                state.object_attributes[obj].add(predicate)
        
        return state


class FactsTracker:
    
    def __init__(self, initial_facts: List[Any], batch_idx: int = 0):
        self.batch_idx = batch_idx
        self.states: Dict[int, FactState] = {}
        self.actions: Dict[int, str] = {}
        
        self.states[0] = FactsParser.parse_facts(initial_facts, step=0)
    
    def update(self, step: int, facts: List[Any], action: str):
        self.states[step] = FactsParser.parse_facts(facts, step=step)
        self.actions[step] = action
    
    def get_state_changes(self, from_step: int, to_step: int) -> Dict[str, Any]:
        if from_step not in self.states or to_step not in self.states:
            return {}
        
        state_before = self.states[from_step]
        state_after = self.states[to_step]
        
        changes = {
            "moved_objects": [],
            "state_changes": [],
            "container_changes": [],
            "inventory_changes": {
                "added": [],
                "removed": []
            },
            "container_state_changes": []
        }
        
        all_objects = set(state_before.object_locations.keys()) | set(state_after.object_locations.keys())
        for obj in all_objects:
            loc_before = state_before.object_locations.get(obj)
            loc_after = state_after.object_locations.get(obj)
            
            if loc_before != loc_after:
                changes["moved_objects"].append({
                    "object": obj,
                    "from": loc_before,
                    "to": loc_after
                })
        
        all_objects = set(state_before.object_states.keys()) | set(state_after.object_states.keys())
        for obj in all_objects:
            states_before = state_before.object_states.get(obj, set())
            states_after = state_after.object_states.get(obj, set())
            
            added_states = states_after - states_before
            removed_states = states_before - states_after
            
            if added_states or removed_states:
                changes["state_changes"].append({
                    "object": obj,
                    "added_states": list(added_states),
                    "removed_states": list(removed_states)
                })
        
        inv_added = state_after.inventory - state_before.inventory
        inv_removed = state_before.inventory - state_after.inventory
        
        changes["inventory_changes"]["added"] = list(inv_added)
        changes["inventory_changes"]["removed"] = list(inv_removed)
        
        all_containers = set(state_before.container_open_states.keys()) | \
                         set(state_after.container_open_states.keys())
        for container in all_containers:
            state_b = state_before.container_open_states.get(container)
            state_a = state_after.container_open_states.get(container)
            
            if state_b != state_a and state_a is not None:
                changes["container_state_changes"].append({
                    "container": container,
                    "opened" if state_a else "closed": True
                })
        
        return changes
    
    def format_state_for_answer(self, step: int, 
                                focus_objects: Optional[List[str]] = None) -> str:
        if step not in self.states:
            return "State not available."
        
        state = self.states[step]
        lines = []
        
        if state.inventory:
            inv_items = [FactsParser.extract_object_name(obj) for obj in state.inventory]
            lines.append(f"Inventory: {', '.join(inv_items)}")
        else:
            lines.append("Inventory: empty")
        
        objects_to_show = focus_objects if focus_objects else state.all_entities
        
        obj_info = []
        for obj_full in objects_to_show:
            matching_objs = [o for o in state.all_entities 
                           if obj_full in o or FactsParser.extract_object_name(o) == obj_full]
            
            for obj in matching_objs:
                info_parts = [obj]
                
                if obj in state.object_locations:
                    loc = state.object_locations[obj]
                    if not loc.startswith('loc ') and not loc.startswith('room '):
                        info_parts.append(f"in/on {loc}")
                    else:
                        info_parts.append(f"at {loc}")
                
                if obj in state.object_states and state.object_states[obj]:
                    states_str = ", ".join(sorted(state.object_states[obj]))
                    info_parts.append(f"states: {states_str}")
                
                if len(info_parts) > 1:
                    obj_info.append(" - ".join(info_parts))
        
        if obj_info:
            lines.append("\nObject states:")
            lines.extend([f"  - {info}" for info in obj_info[:20]])
        
        return "\n".join(lines)
    
    def format_changes_for_answer(self, from_step: int, to_step: int) -> str:
        changes = self.get_state_changes(from_step, to_step)
        
        if not any(changes.values()):
            return "No significant changes detected."
        
        lines = []
        
        if changes["inventory_changes"]["added"]:
            added = [FactsParser.extract_object_name(obj) 
                    for obj in changes["inventory_changes"]["added"]]
            lines.append(f"Picked up: {', '.join(added)}")
        
        if changes["inventory_changes"]["removed"]:
            removed = [FactsParser.extract_object_name(obj) 
                      for obj in changes["inventory_changes"]["removed"]]
            lines.append(f"Put down: {', '.join(removed)}")
        
        if changes["moved_objects"]:
            for change in changes["moved_objects"][:10]:
                obj = FactsParser.extract_object_name(change["object"])
                from_loc = change["from"] or "unknown"
                to_loc = change["to"] or "unknown"
                lines.append(f"{obj} moved from {from_loc} to {to_loc}")
        
        if changes["state_changes"]:
            for change in changes["state_changes"][:10]:
                obj = FactsParser.extract_object_name(change["object"])
                if change["added_states"]:
                    states = ", ".join(change["added_states"])
                    lines.append(f"{obj} became: {states}")
        
        if changes["container_state_changes"]:
            for change in changes["container_state_changes"]:
                container = FactsParser.extract_object_name(change["container"])
                action = "opened" if "opened" in change else "closed"
                lines.append(f"{container} was {action}")
        
        return "\n".join(lines) if lines else "No significant changes detected."
