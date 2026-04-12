from __future__ import annotations
from typing import List, Dict, Any, Tuple
import sqlite3, re

def _cond_sql(by: str, cond: str) -> Tuple[str, List[str]]:
    m = re.match(r"\s*([<>]=?|=)\s*(\d{1,9})\s*$", cond)
    if not m:
        return "", []
    op, val = m.group(1), m.group(2)
    if by == "year":
        return f" AND y.year {op} ?", [val]
    if by == "number":
        return f" AND n.num  {op} ?", [val]
    return "", []

def execute_plan(db_path: str, plan: Dict[str,Any]) -> Dict[str,Any]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    sids = None
    context_rows: List[Dict[str,Any]] = []

    for step in plan.get("steps", []):
        op = step.get("op","").upper()
        if op == "FIND":
            by = step.get("by")
            val = step.get("value","")
            if by == "entity":
                cur.execute(
                    "SELECT s.sid,s.doc_id,s.title,s.text FROM sentences s JOIN entities e ON s.sid=e.sid WHERE e.ent=? LIMIT 500",
                    (val,),
                )
            elif by == "number":
                cur.execute(
                    "SELECT s.sid,s.doc_id,s.title,s.text FROM sentences s JOIN numbers  n ON s.sid=n.sid WHERE n.num=? LIMIT 500",
                    (val,),
                )
            elif by == "year":
                cur.execute(
                    "SELECT s.sid,s.doc_id,s.title,s.text FROM sentences s JOIN years    y ON s.sid=y.sid WHERE y.year=? LIMIT 500",
                    (val,),
                )
            else:
                cur.execute(
                    "SELECT s.sid,s.doc_id,s.title,s.text FROM sentences s WHERE s.text LIKE ? LIMIT 500",
                    (f"%{val}%",),
                )
            rows = cur.fetchall()
            sids = {r[0] for r in rows}
            context_rows = [{"sid": r[0], "doc_id": r[1], "title": r[2], "text": r[3]} for r in rows]

        elif op == "FILTER":
            if sids is None:
                continue
            by, cond = step.get("by"), step.get("cond","")
            sql, args = _cond_sql(by, cond)
            if not sql:
                continue
            q = (
                "SELECT s.sid FROM sentences s "
                "LEFT JOIN numbers n ON s.sid=n.sid "
                "LEFT JOIN years y ON s.sid=y.sid "
                f"WHERE s.sid IN ({','.join(['?']*len(sids))}) {sql}"
            )
            cur.execute(q, list(sids) + args)
            sids = {r[0] for r in cur.fetchall()}
            context_rows = [r for r in context_rows if r["sid"] in sids]

        elif op == "SELECT":
            key = step.get("keyword","")
            if sids is None or not key:
                continue
            sids = {r["sid"] for r in context_rows if key in r["text"]}
            context_rows = [r for r in context_rows if r["sid"] in sids]

        elif op == "AGG":
            typ = step.get("type","COUNT").upper()
            if typ == "COUNT":
                plan["__agg_value"] = len(context_rows)
            elif typ in ("MAX", "MIN"):
                target = step.get("target","year")
                if target == "year":
                    cur.execute(
                        f"SELECT {typ}(CAST(year AS INT)) FROM years WHERE sid IN ({','.join(['?']*len(sids))})",
                        list(sids),
                    )
                else:
                    cur.execute(
                        f"SELECT {typ}(CAST(num  AS FLOAT)) FROM numbers WHERE sid IN ({','.join(['?']*len(sids))})",
                        list(sids),
                    )
                res = cur.fetchone()
                plan["__agg_value"] = res[0] if res and res[0] is not None else None

        elif op == "COMPARE":
            mode = step.get("mode","MAX").upper()
            if sids:
                cur.execute(
                    f"""
                    SELECT s.sid, s.doc_id, s.title, s.text, MAX(CAST(n.num AS FLOAT))
                    FROM sentences s
                    JOIN numbers n ON s.sid=n.sid
                    WHERE s.sid IN ({','.join(['?']*len(sids))})
                    """,
                    list(sids),
                )
                row = cur.fetchone()
                if row:
                    context_rows = [{"sid": row[0], "doc_id": row[1], "title": row[2], "text": row[3]}]

        elif op == "ANSWER":
            tpl = step.get("template","{value}")
            val = plan.get("__agg_value")
            result = tpl.replace("{value}", str(val) if val is not None else "unknown")
            doc_ids = sorted({r["doc_id"] for r in context_rows})[:10]
            con.close()
            return {"answer": result, "evidence_doc_ids": doc_ids, "evidence": context_rows}

    doc_ids = sorted({r["doc_id"] for r in context_rows})[:10] if context_rows else []
    con.close()
    return {"answer": "unknown", "evidence_doc_ids": doc_ids, "evidence": context_rows}
