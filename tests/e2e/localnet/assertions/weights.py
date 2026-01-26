"""Weight emission assertions.

Verifies that weight computation produces valid results:
- Weights sum to 1.0 (or 0 if no miners)
- All weights are non-negative
- Weights match skill score ranking
- No NaN or infinite weights
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass
class WeightAssertionResult:
    """Result of weight assertion checks."""
    
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    details: List[dict] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return self.failed == 0
    
    def add_pass(self, msg: str) -> None:
        self.passed += 1
        self.details.append({"status": "pass", "message": msg})
    
    def add_fail(self, msg: str) -> None:
        self.failed += 1
        self.details.append({"status": "fail", "message": msg})
    
    def add_warning(self, msg: str) -> None:
        self.warnings += 1
        self.details.append({"status": "warning", "message": msg})


async def assert_weights_sum_to_one(db: AsyncEngine) -> WeightAssertionResult:
    """Assert weights sum to 1.0 (within tolerance)."""
    result = WeightAssertionResult()
    
    async with db.connect() as conn:
        # Get latest weights
        rows = await conn.execute(text("""
            SELECT 
                SUM(weight) as total_weight,
                COUNT(*) as weight_count,
                MIN(weight) as min_weight,
                MAX(weight) as max_weight
            FROM miner_weight
            WHERE computed_at = (SELECT MAX(computed_at) FROM miner_weight)
        """))
        row = rows.mappings().fetchone()
        
        if row is None or row["weight_count"] == 0:
            result.add_warning("No weights found in database")
            return result
        
        total = float(row["total_weight"]) if row["total_weight"] else 0
        
        # Check sum is ~1.0 (tolerance of 0.01)
        if 0.99 <= total <= 1.01:
            result.add_pass(f"Weights sum to {total:.6f} (within tolerance)")
        elif total == 0:
            result.add_warning("Weights sum to 0 (no active miners?)")
        else:
            result.add_fail(f"Weights sum to {total:.6f} (expected ~1.0)")
        
        result.details.append({
            "status": "info",
            "message": f"Weight stats: count={row['weight_count']}, min={row['min_weight']}, max={row['max_weight']}"
        })
    
    return result


async def assert_weights_non_negative(db: AsyncEngine) -> WeightAssertionResult:
    """Assert all weights are non-negative."""
    result = WeightAssertionResult()
    
    async with db.connect() as conn:
        rows = await conn.execute(text("""
            SELECT COUNT(*) as negative_count
            FROM miner_weight
            WHERE weight < 0
        """))
        row = rows.mappings().fetchone()
        
        if row and row["negative_count"] > 0:
            result.add_fail(f"Found {row['negative_count']} negative weights")
        else:
            result.add_pass("All weights are non-negative")
    
    return result


async def assert_weights_match_skill_ranking(db: AsyncEngine) -> WeightAssertionResult:
    """Assert weights correlate with skill score ranking.
    
    Higher skill score should generally mean higher weight.
    """
    result = WeightAssertionResult()
    
    async with db.connect() as conn:
        # Get latest weights with corresponding skill scores
        rows = await conn.execute(text("""
            SELECT 
                mw.miner_id,
                mw.weight,
                mrs.skill_score
            FROM miner_weight mw
            LEFT JOIN miner_rolling_score mrs ON mw.miner_id = mrs.miner_id
            WHERE mw.computed_at = (SELECT MAX(computed_at) FROM miner_weight)
              AND mrs.as_of = (SELECT MAX(as_of) FROM miner_rolling_score)
            ORDER BY mw.weight DESC
        """))
        data = rows.mappings().all()
        
        if len(data) < 2:
            result.add_warning("Not enough miners to check ranking correlation")
            return result
        
        # Check if higher weights correlate with higher skill scores
        # Simple check: top weight should have top or near-top skill
        weights = [(d["miner_id"], d["weight"], d["skill_score"]) for d in data]
        
        # Sort by skill
        by_skill = sorted(
            [(m, w, s) for m, w, s in weights if s is not None],
            key=lambda x: x[2],
            reverse=True
        )
        
        if len(by_skill) < 2:
            result.add_warning("Not enough valid skill scores to check ranking")
            return result
        
        # Check if highest weight miner is in top 50% by skill
        top_weight_miner = weights[0][0]
        top_half_by_skill = [x[0] for x in by_skill[:len(by_skill)//2 + 1]]
        
        if top_weight_miner in top_half_by_skill:
            result.add_pass("Highest weight miner is in top half by skill score")
        else:
            result.add_warning("Highest weight miner is not in top half by skill score (may be expected)")
    
    return result


async def assert_no_nan_weights(db: AsyncEngine) -> WeightAssertionResult:
    """Assert no NaN or infinite weights."""
    result = WeightAssertionResult()
    
    async with db.connect() as conn:
        rows = await conn.execute(text("""
            SELECT 
                COUNT(*) FILTER (WHERE weight != weight) as nan_count,
                COUNT(*) FILTER (WHERE weight = 'Infinity'::float OR weight = '-Infinity'::float) as inf_count
            FROM miner_weight
        """))
        row = rows.mappings().fetchone()
        
        if row:
            if row["nan_count"] > 0:
                result.add_fail(f"Found {row['nan_count']} NaN weights")
            else:
                result.add_pass("No NaN weights")
            
            if row["inf_count"] > 0:
                result.add_fail(f"Found {row['inf_count']} infinite weights")
            else:
                result.add_pass("No infinite weights")
    
    return result


async def run_all_weight_assertions(db: AsyncEngine) -> WeightAssertionResult:
    """Run all weight assertions."""
    combined = WeightAssertionResult()
    
    checks = [
        ("Weights sum", assert_weights_sum_to_one),
        ("Non-negative", assert_weights_non_negative),
        ("Ranking correlation", assert_weights_match_skill_ranking),
        ("No NaN/inf", assert_no_nan_weights),
    ]
    
    for name, check_fn in checks:
        try:
            result = await check_fn(db)
            combined.passed += result.passed
            combined.failed += result.failed
            combined.warnings += result.warnings
            combined.details.extend(result.details)
        except Exception as e:
            combined.add_fail(f"{name} check failed with exception: {e}")
    
    return combined
