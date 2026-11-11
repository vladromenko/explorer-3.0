"""Local catalogue. Raw episodes are immutable; exclusion never deletes data."""
import json
import sqlite3
import time
from .paths import DATA, prepare

def connect():
    prepare()
    db=sqlite3.connect(DATA/'catalog.sqlite3',timeout=15)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    return db

def initialize():
    with connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS episodes (
          id TEXT PRIMARY KEY, profile TEXT NOT NULL, dataset TEXT NOT NULL,
          episode_index INTEGER NOT NULL, session TEXT NOT NULL, scene TEXT NOT NULL,
          category TEXT NOT NULL, outcome TEXT NOT NULL, frames INTEGER NOT NULL,
          excluded INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, profile TEXT NOT NULL, policy TEXT NOT NULL,
          status TEXT NOT NULL, config TEXT NOT NULL, created REAL NOT NULL,
          error TEXT, checkpoint TEXT);
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY, created REAL NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL);
        ''')
        db.execute("UPDATE jobs SET status='interrupted',error='Application restarted; explicit resume required' WHERE status='running'")

def rows(table: str) -> list[dict]:
    if table not in {'episodes','jobs','events'}:
        raise ValueError('Unknown catalogue')
    with connect() as db:
        return [dict(row) for row in db.execute(f'SELECT * FROM {table} ORDER BY created DESC')]

def event(level: str, message: str):
    with connect() as db:
        db.execute('INSERT INTO events(created,level,message) VALUES(?,?,?)',(time.time(),level,message))

def exclude(episode_id: str, value: bool):
    with connect() as db:
        db.execute('UPDATE episodes SET excluded=? WHERE id=?',(int(value),episode_id))

def add_episode(values: dict):
    keys=('id','profile','dataset','episode_index','session','scene','category','outcome','frames')
    with connect() as db:
        db.execute('INSERT INTO episodes('+','.join(keys)+',created) VALUES('+','.join('?' for _ in range(10))+')',
                   tuple(values[k] for k in keys)+(time.time(),))
