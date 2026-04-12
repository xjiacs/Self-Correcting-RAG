# src/symbolic/fact_index.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re, sqlite3, os

NUM = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
ENTITY = re.compile(r"[A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff\-_/\.]{1,50}")

def sentence_split(text: str) -> List[str]:
    if not text: return []
    t = text.replace("！","。").replace("？","。").replace("!","。").replace("?","。")
    parts = [p.strip() for p in re.split(r"[。]\s*", t) if p.strip()]
    return parts

def extract_fields(sent: str) -> Dict[str, Any]:
    nums = list({n.replace(",","") for n in NUM.findall(sent)})
    years = list(set(YEAR.findall(sent)))
    ents = list(set(ENTITY.findall(sent)))
    return {"numbers": nums, "years": years, "entities": ents}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sentences(
  sid INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id TEXT, title TEXT, text TEXT
);
CREATE TABLE IF NOT EXISTS entities(
  sid INTEGER, ent TEXT
);
CREATE TABLE IF NOT EXISTS numbers(
  sid INTEGER, num TEXT
);
CREATE TABLE IF NOT EXISTS years(
  sid INTEGER, year TEXT
);
CREATE INDEX IF NOT EXISTS idx_entities ON entities(ent);
CREATE INDEX IF NOT EXISTS idx_numbers  ON numbers(num);
CREATE INDEX IF NOT EXISTS idx_years    ON years(year);
CREATE INDEX IF NOT EXISTS idx_doc      ON sentences(doc_id);
"""

def build_fact_store(corpus: List[Dict[str,Any]], db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript(SCHEMA_SQL)
    for d in corpus:
        doc_id, title, text = str(d.get("doc_id")), d.get("title",""), d.get("text","")
        for s in sentence_split(text):
            cur.execute("INSERT INTO sentences(doc_id,title,text) VALUES(?,?,?)",(doc_id,title,s))
            sid = cur.lastrowid
            f = extract_fields(s)
            for e in f["entities"]: cur.execute("INSERT INTO entities VALUES(?,?)",(sid,e))
            for n in f["numbers"]:  cur.execute("INSERT INTO numbers  VALUES(?,?)",(sid,n))
            for y in f["years"]:    cur.execute("INSERT INTO years    VALUES(?,?)",(sid,y))
    con.commit(); con.close()
