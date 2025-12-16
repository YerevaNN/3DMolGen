SELECT name, COUNT(*) as count, SUM(dur)/1e6 as total_ms, AVG(dur)/1e3 as avg_us
FROM slices
WHERE name LIKE '%index%' OR name LIKE '%fill%' OR name LIKE '%put%' OR name LIKE '%scatter%'
GROUP BY name
ORDER BY total_ms DESC
LIMIT 30;
