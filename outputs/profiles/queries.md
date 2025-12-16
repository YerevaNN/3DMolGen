
>   SELECT name, COUNT(*) as count, SUM(dur)/1e6 as total_ms  FROM slices  WHERE name LIKE '%logit%' OR name LIKE '%constraint%' OR name LIKE '%mask%'  GROUP BY name  ORDER BY total_ms DESC;
name                 count                total_ms             
-------------------- -------------------- -------------------- 
void at::native::vec                38856            73.851842 
void at::native::vec                 1730             3.115032 

-----------



>   SELECT name, COUNT(*) as launches, AVG(dur)/1e3 as avg_us, SUM(dur)/1e6 as total_ms  FROM slices  WHERE dur < 100000  -- less than 100us  GROUP BY name  HAVING launches > 100  ORDER BY launches DESC  LIMIT 20;
name                 launches             avg_us               total_ms             
-------------------- -------------------- -------------------- -------------------- 
Iteration Start: PyT              6926250             4.991294         34570.948946 

--------------------------------


Here’s a clean Markdown version of what’s visible in the screenshot.

⸻

Query

SELECT
  name,
  ts / 1e9 AS start_sec,
  dur / 1e9 AS duration_sec
FROM slices
WHERE depth = 0 -- top-level events only
ORDER BY ts;


⸻

Query Result (excerpt)

Rows: 6,970,677
Query time: 2,436.1 ms

name	start_sec	duration_sec
PyTorch Profiler (0)	6255670.318873463	57.24404788
Iteration Start: PyTorch Profiler	6255670.318873463	0
cudaMemcpyAsync	6255670.324874711	0.001770822
Memcpy HtoD (Pageable → Device)	6255670.324902735	0.00000128
Activity Buffer Request	6255670.324904904	0.001735618
cudaStreamSynchronize	6255670.326646632	0.000009368
cudaMemcpyAsync	6255670.326689551	0.000014498
cudaStreamSynchronize	6255670.326704218	0.000005942
Memcpy HtoD (Pageable → Device)	6255670.326705171	0.000001248
cudaLaunchKernel	6255670.326993611	0.015772067
Runtime Triggered Module Loading	6255670.326998267	0.010583613
Runtime Triggered Module Loading	6255670.337594021	0.001683452
Lazy Function Loading	6255670.339282399	0.000104694


⸻
.