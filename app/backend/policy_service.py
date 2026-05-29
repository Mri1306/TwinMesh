# =============================================================================
# TwinMesh App · policy_service.py
# MDP policy queries: tier actions, cost analysis, safety report
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

from src.utils.constants import STATES, STATE_MAP
from src.utils.logger import get_logger

log = get_logger("policy_service")

AUDIT_DIR   = os.path.join(ROOT, "results", "audit")
COMBINED_DIR= os.path.join(ROOT, "results", "combined")

# Hard-coded policy (matches Day 2 mdp_cost_matrix.py)
POLICY_TABLE = {
    ("S0", 0): "Strong",  ("S0", 1): "Escrow", ("S0", 2): "Stale",  ("S0", 3): "Stale",
    ("S1", 0): "Strong",  ("S1", 1): "Escrow", ("S1", 2): "Stale",  ("S1", 3): "Stale",
    ("S2", 0): "Strong",  ("S2", 1): "Escrow", ("S2", 2): "Stale",  ("S2", 3): "Stale",
    ("S3", 0): "Strong",  ("S3", 1): "Escrow", ("S3", 2): "CRDT",   ("S3", 3): "CRDT",
    ("S4", 0): "Strong",  ("S4", 1): "Escrow", ("S4", 2): "Stale",  ("S4", 3): "Stale",
}

COST_TABLE = {
    "S0": {"Strong":1.0,"Escrow":0.6,"CRDT":0.2,"Stale":0.1,"Block":2.0},
    "S1": {"Strong":1.3,"Escrow":0.8,"CRDT":0.3,"Stale":0.1,"Block":2.6},
    "S2": {"Strong":1.7,"Escrow":1.0,"CRDT":0.3,"Stale":0.2,"Block":3.4},
    "S3": {"Strong":3.8,"Escrow":2.3,"CRDT":0.8,"Stale":0.4,"Block":7.6},
    "S4": {"Strong":1.1,"Escrow":0.7,"CRDT":0.2,"Stale":0.1,"Block":2.2},
}

ACTION_DESCRIPTIONS = {
    "Strong": "Full linearisability — every write immediately visible to all nodes",
    "Escrow": "Bounded divergence — writes acknowledged with bounded staleness",
    "CRDT":   "Conflict-free replicated — automatic merge on convergence",
    "Stale":  "Serve cached data — reads may be stale, writes buffered",
    "Block":  "Reject writes — port in crisis, protect consistency",
}

ACTION_COST_LABEL = {
    "Strong": "Highest cost — maximum consistency guarantee",
    "Escrow": "Moderate cost — partial consistency with safety",
    "CRDT":   "Low cost — eventual consistency via CRDT merge",
    "Stale":  "Minimal cost — serve stale reads, queue writes",
    "Block":  "Crisis cost — block writes, protect tier-0",
}

TIER_DESCRIPTIONS = {
    0: "Financial transactions — zero staleness tolerance",
    1: "Inventory allocation — bounded staleness acceptable",
    2: "Route planning — eventual consistency acceptable",
    3: "Reporting & analytics — stale reads acceptable",
}


def get_policy_for_state(state: str) -> dict:
    """Returns full policy for a given disruption state."""
    actions = {}
    for tier in range(4):
        action = POLICY_TABLE.get((state, tier), "Strong")
        cost   = COST_TABLE.get(state, {}).get(action, 1.0)
        strong_cost = COST_TABLE.get(state, {}).get("Strong", 1.0)
        savings     = round((strong_cost - cost) / strong_cost * 100, 1)
        actions[tier] = {
            "tier":        tier,
            "action":      action,
            "cost":        round(cost, 4),
            "savings_pct": savings,
            "description": ACTION_DESCRIPTIONS.get(action, ""),
            "tier_desc":   TIER_DESCRIPTIONS.get(tier, ""),
        }
    return {
        "state":       state,
        "actions":     actions,
        "total_cost":  round(sum(v["cost"] for v in actions.values()), 4),
        "strong_cost": round(sum(COST_TABLE.get(state,{}).get("Strong",1.0)
                                 for _ in range(4)), 4),
    }


def get_full_policy_table() -> list:
    """Returns the complete 5-state × 4-tier policy matrix as list of rows."""
    rows = []
    for state in STATES:
        policy = get_policy_for_state(state)
        for tier, rec in policy["actions"].items():
            rows.append({
                "state":    state,
                "tier":     tier,
                "action":   rec["action"],
                "cost":     rec["cost"],
                "savings":  rec["savings_pct"],
            })
    return rows


def get_safety_report() -> dict:
    """Loads saved MDP safety report from Day 5, or returns defaults."""
    safety_path = os.path.join(COMBINED_DIR, "mdp_safety_report.csv")
    if os.path.exists(safety_path):
        row = pd.read_csv(safety_path).iloc[0].to_dict()
        return {
            "tier0_violations":    int(row.get("tier0_violations", 0)),
            "total_decisions":     int(row.get("total_decisions", 5144)),
            "policy_cost":         round(float(row.get("total_policy_cost", 880999)), 2),
            "strong_cost":         round(float(row.get("total_strong_cost", 1850212)), 2),
            "cost_reduction_pct":  round(float(row.get("cost_reduction_pct", 52.4)), 2),
            "safety_valid":        bool(row.get("safety_claim_valid", True)),
        }
    # Defaults from Day 5 run
    return {
        "tier0_violations":   0,
        "total_decisions":    5144,
        "policy_cost":        880999.34,
        "strong_cost":        1850212.46,
        "cost_reduction_pct": 52.4,
        "safety_valid":       True,
    }


def get_cost_comparison_all_states() -> list:
    """
    For each state: compare policy cost vs AlwaysStrong cost.
    Used for the cost reduction bar chart.
    """
    rows = []
    for state in STATES:
        policy = get_policy_for_state(state)
        rows.append({
            "state":       state,
            "policy_cost": policy["total_cost"],
            "strong_cost": policy["strong_cost"],
            "saving_pct":  round((policy["strong_cost"] - policy["total_cost"])
                                  / policy["strong_cost"] * 100, 1),
        })
    return rows


def get_action_explanation(action: str) -> dict:
    return {
        "action":      action,
        "description": ACTION_DESCRIPTIONS.get(action, "Unknown action"),
        "cost_label":  ACTION_COST_LABEL.get(action, ""),
    }