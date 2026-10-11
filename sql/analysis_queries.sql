-- Analytical queries (JOIN / GROUP BY / CASE WHEN). Run against data/waste.db or the CSV export in Power BI.

-- 1. Operator-verified accuracy per true material
SELECT p.reviewed_class AS material, COUNT(*) AS checked,
       ROUND(100.0 * SUM(p.predicted_class = p.reviewed_class) / COUNT(*), 1) AS accuracy_pct
FROM predictions p
WHERE p.review_state = 'done'
GROUP BY p.reviewed_class ORDER BY accuracy_pct;

-- 2. Accuracy by confidence band (justifies the routing threshold)
SELECT CASE WHEN confidence >= 0.8 THEN '>= 0.8' ELSE '< 0.8' END AS band,
       COUNT(*) AS checked,
       ROUND(100.0 * SUM(predicted_class = reviewed_class) / COUNT(*), 1) AS accuracy_pct
FROM predictions WHERE review_state = 'done' GROUP BY band;

-- 3. Daily recyclable share per line (diversion KPI)
SELECT date(p.created_at) AS day, p.line_id,
       COUNT(*) AS items,
       ROUND(100.0 * SUM(c.recyclable) / COUNT(*), 1) AS recyclable_pct
FROM predictions p JOIN waste_classes c ON c.class_name = COALESCE(p.reviewed_class, p.predicted_class)
GROUP BY day, p.line_id ORDER BY day, p.line_id;

-- 4. Most common verified confusions
SELECT predicted_class, reviewed_class, COUNT(*) AS n
FROM predictions
WHERE review_state = 'done' AND predicted_class <> reviewed_class
GROUP BY predicted_class, reviewed_class ORDER BY n DESC;

-- 5. Confident-but-wrong items (highest-risk errors)
SELECT id, line_id, predicted_class, reviewed_class, confidence
FROM predictions
WHERE review_state = 'done' AND predicted_class <> reviewed_class AND confidence >= 0.8
ORDER BY confidence DESC LIMIT 20;
