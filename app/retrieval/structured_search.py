"""
Structured SQL lookup module for quantitative evidence from the metric_facts table.
"""

from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.db.client import get_db_connection
from data.scientific_knowledge import METRIC_FACTS


def search_metric_facts(
    metrics: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    limit: int = 4
) -> List[Dict[str, Any]]:
    """
    Retrieves quantitative, verified scientific facts from the metric_facts table.
    
    If the database is not configured, queries the curated in-memory METRIC_FACTS dataset.
    """
    # 1. Live Supabase structured SQL query if configured
    if settings.is_database_configured:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    conditions = []
                    params = []

                    if metrics:
                        conditions.append("affects_metric = ANY(%s)")
                        params.append(metrics)
                    
                    if keywords:
                        kw_conditions = []
                        for kw in keywords:
                            kw_conditions.append("(intervention ILIKE %s OR affects_metric ILIKE %s)")
                            params.extend([f"%{kw}%", f"%{kw}%"])
                        if kw_conditions:
                            conditions.append(f"({' OR '.join(kw_conditions)})")

                    where_clause = f"WHERE {' OR '.join(conditions)}" if conditions else ""
                    sql = f"""
                        SELECT id, intervention, affects_metric, effect_value, time_horizon, source, source_url
                        FROM metric_facts
                        {where_clause}
                        LIMIT %s;
                    """
                    params.append(limit)
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    if rows:
                        return [
                            {
                                "id": str(r["id"]),
                                "intervention": r["intervention"],
                                "affects_metric": r["affects_metric"],
                                "effect_value": r["effect_value"],
                                "time_horizon": r["time_horizon"],
                                "source": r["source"],
                                "source_url": r["source_url"]
                            }
                            for r in rows
                        ]
        except Exception as e:
            print(f"[WARN] Database structured search fallback activated: {e}")

    # 2. In-memory fallback matching
    results = []
    for fact in METRIC_FACTS:
        match = False
        if metrics and fact["affects_metric"] in metrics:
            match = True
        elif keywords:
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in fact["intervention"].lower() or kw_lower in fact["affects_metric"].lower():
                    match = True
                    break
        elif not metrics and not keywords:
            # If no filter specified, include general facts
            match = True

        if match and fact not in results:
            results.append(fact)

    return results[:limit]
