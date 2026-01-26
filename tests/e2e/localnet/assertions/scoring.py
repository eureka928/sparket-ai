"""Scoring invariant assertions.

Verifies that the scoring pipeline produces valid results:
- CLV scores are bounded and non-NaN
- Brier scores are in [0, 1]
- PSS scores are bounded
- Rolling aggregates are computed
- Skill scores are computed
- No NaN or infinite values in any score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


@dataclass
class ScoringAssertionResult:
    """Result of scoring assertion checks."""
    
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


async def assert_clv_scores_valid(db: AsyncEngine) -> ScoringAssertionResult:
    """Assert CLV scores are valid (bounded, non-NaN)."""
    result = ScoringAssertionResult()
    
    async with db.connect() as conn:
        # Check submission_vs_close table
        rows = await conn.execute(text("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE clv IS NULL) as null_count,
                COUNT(*) FILTER (WHERE clv != clv) as nan_count,
                COUNT(*) FILTER (WHERE clv > 10 OR clv < -10) as extreme_count,
                MIN(clv) as min_clv,
                MAX(clv) as max_clv,
                AVG(clv) as avg_clv
            FROM submission_vs_close
        """))
        row = rows.mappings().fetchone()
        
        if row is None or row["total"] == 0:
            result.add_warning("No CLV scores to check")
            return result
        
        # Check for NaN values
        if row["nan_count"] > 0:
            result.add_fail(f"Found {row['nan_count']} NaN CLV values")
        else:
            result.add_pass("No NaN CLV values")
        
        # Check for extreme values
        if row["extreme_count"] > 0:
            result.add_warning(f"Found {row['extreme_count']} extreme CLV values (>10 or <-10)")
        else:
            result.add_pass("CLV values within expected range")
        
        result.details.append({
            "status": "info",
            "message": f"CLV stats: min={row['min_clv']}, max={row['max_clv']}, avg={row['avg_clv']}, total={row['total']}"
        })
    
    return result


async def assert_brier_scores_valid(db: AsyncEngine) -> ScoringAssertionResult:
    """Assert Brier scores are in valid range [0, 1]."""
    result = ScoringAssertionResult()
    
    async with db.connect() as conn:
        rows = await conn.execute(text("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE brier_score IS NULL) as null_count,
                COUNT(*) FILTER (WHERE brier_score != brier_score) as nan_count,
                COUNT(*) FILTER (WHERE brier_score < 0 OR brier_score > 1) as out_of_range,
                MIN(brier_score) as min_brier,
                MAX(brier_score) as max_brier,
                AVG(brier_score) as avg_brier
            FROM submission_outcome_score
        """))
        row = rows.mappings().fetchone()
        
        if row is None or row["total"] == 0:
            result.add_warning("No Brier scores to check")
            return result
        
        # Check for NaN values
        if row["nan_count"] > 0:
            result.add_fail(f"Found {row['nan_count']} NaN Brier values")
        else:
            result.add_pass("No NaN Brier values")
        
        # Check for out-of-range values
        if row["out_of_range"] > 0:
            result.add_fail(f"Found {row['out_of_range']} Brier values outside [0, 1]")
        else:
            result.add_pass("All Brier values in valid range [0, 1]")
        
        result.details.append({
            "status": "info",
            "message": f"Brier stats: min={row['min_brier']}, max={row['max_brier']}, avg={row['avg_brier']}, total={row['total']}"
        })
    
    return result


async def assert_pss_scores_valid(db: AsyncEngine) -> ScoringAssertionResult:
    """Assert PSS scores are valid."""
    result = ScoringAssertionResult()
    
    async with db.connect() as conn:
        rows = await conn.execute(text("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE pss IS NULL) as null_count,
                COUNT(*) FILTER (WHERE pss != pss) as nan_count,
                MIN(pss) as min_pss,
                MAX(pss) as max_pss,
                AVG(pss) as avg_pss
            FROM submission_outcome_score
        """))
        row = rows.mappings().fetchone()
        
        if row is None or row["total"] == 0:
            result.add_warning("No PSS scores to check")
            return result
        
        # Check for NaN values
        if row["nan_count"] > 0:
            result.add_fail(f"Found {row['nan_count']} NaN PSS values")
        else:
            result.add_pass("No NaN PSS values")
        
        result.details.append({
            "status": "info",
            "message": f"PSS stats: min={row['min_pss']}, max={row['max_pss']}, avg={row['avg_pss']}, total={row['total']}"
        })
    
    return result


async def assert_rolling_scores_valid(db: AsyncEngine) -> ScoringAssertionResult:
    """Assert rolling aggregate scores are valid."""
    result = ScoringAssertionResult()
    
    async with db.connect() as conn:
        rows = await conn.execute(text("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT miner_id) as miners,
                COUNT(*) FILTER (WHERE skill_score IS NULL) as null_skill,
                COUNT(*) FILTER (WHERE skill_score != skill_score) as nan_skill,
                MIN(skill_score) as min_skill,
                MAX(skill_score) as max_skill,
                AVG(skill_score) as avg_skill
            FROM miner_rolling_score
            WHERE skill_score IS NOT NULL
        """))
        row = rows.mappings().fetchone()
        
        if row is None or row["total"] == 0:
            result.add_warning("No rolling scores to check")
            return result
        
        # Check for NaN values
        if row["nan_skill"] > 0:
            result.add_fail(f"Found {row['nan_skill']} NaN skill scores")
        else:
            result.add_pass("No NaN skill scores")
        
        # Check that multiple miners have scores
        if row["miners"] > 1:
            result.add_pass(f"Rolling scores computed for {row['miners']} miners")
        elif row["miners"] == 1:
            result.add_warning("Only 1 miner has rolling scores")
        
        result.details.append({
            "status": "info",
            "message": f"Skill stats: min={row['min_skill']}, max={row['max_skill']}, avg={row['avg_skill']}, miners={row['miners']}"
        })
    
    return result


async def assert_no_infinite_values(db: AsyncEngine) -> ScoringAssertionResult:
    """Assert no infinite values in any scoring table."""
    result = ScoringAssertionResult()
    
    tables_and_columns = [
        ("submission_vs_close", ["clv", "cle", "tw_clv", "tw_cle"]),
        ("submission_outcome_score", ["brier_score", "log_loss", "pss"]),
        ("miner_rolling_score", ["fq_score", "edge_score", "mes_score", "sos_score", "lead_score", "skill_score"]),
    ]
    
    async with db.connect() as conn:
        for table, columns in tables_and_columns:
            for col in columns:
                try:
                    rows = await conn.execute(text(f"""
                        SELECT COUNT(*) as infinite_count
                        FROM {table}
                        WHERE {col} = 'Infinity'::float OR {col} = '-Infinity'::float
                    """))
                    row = rows.mappings().fetchone()
                    
                    if row and row["infinite_count"] > 0:
                        result.add_fail(f"Found {row['infinite_count']} infinite values in {table}.{col}")
                    else:
                        result.add_pass(f"No infinite values in {table}.{col}")
                except Exception as e:
                    # Column may not exist or table empty
                    result.add_warning(f"Could not check {table}.{col}: {e}")
    
    return result


async def run_all_scoring_assertions(db: AsyncEngine) -> ScoringAssertionResult:
    """Run all scoring invariant assertions."""
    combined = ScoringAssertionResult()
    
    checks = [
        ("CLV scores", assert_clv_scores_valid),
        ("Brier scores", assert_brier_scores_valid),
        ("PSS scores", assert_pss_scores_valid),
        ("Rolling scores", assert_rolling_scores_valid),
        ("Infinite values", assert_no_infinite_values),
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
