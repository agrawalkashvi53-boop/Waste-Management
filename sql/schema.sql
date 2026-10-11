-- Portable SQL (SQLite here; works on PostgreSQL/MySQL with minor type changes)
CREATE TABLE IF NOT EXISTS waste_classes (
    class_name TEXT PRIMARY KEY,
    recyclable INTEGER NOT NULL CHECK (recyclable IN (0, 1))
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,                 -- UTC, ISO-8601
    line_id         TEXT NOT NULL DEFAULT 'LINE-1',-- sorting line / camera station
    source_name     TEXT,
    predicted_class TEXT NOT NULL REFERENCES waste_classes (class_name),
    confidence      REAL NOT NULL,
    route           TEXT NOT NULL CHECK (route IN ('auto', 'manual')),
    review_state    TEXT NOT NULL DEFAULT 'none'
                    CHECK (review_state IN ('none', 'pending', 'done')),
    reviewed_class  TEXT REFERENCES waste_classes (class_name),
    reviewed_at     TEXT,
    model_version   TEXT NOT NULL,
    thumb           BLOB                           -- kept only while review is pending
);

CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions (created_at);
CREATE INDEX IF NOT EXISTS idx_pred_review  ON predictions (review_state);
