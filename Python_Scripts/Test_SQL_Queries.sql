-- SELECT group_1_central_tendency_stats_cor1_id, group_2_central_tendency_stats_cor1_id, JSD from KLD_JSD_no_random

SELECT group1, group2, mean_diff from tukey_hsd_stats_cor1;

-- Copy this query into a new table
DROP TABLE IF EXISTS tukey_mean_diff_query_cor1;
CREATE TABLE tukey_mean_diff_query_cor1 AS SELECT group1, group2, mean_diff from tukey_hsd_stats_cor1;


INSERT INTO tukey_mean_diff_query_cor1 VALUES ('COR1', 'COR1', 0);
INSERT INTO tukey_mean_diff_query_cor1 VALUES ('ne', 'ne', 0);
INSERT INTO tukey_mean_diff_query_cor1 VALUES ('ne_LOS', 'ne_LOS', 0);
INSERT INTO tukey_mean_diff_query_cor1 VALUES ('pB', 'pB', 0);
INSERT INTO tukey_mean_diff_query_cor1 VALUES ('random', 'random', 0);

SELECT group1, group2, mean_diff from tukey_mean_diff_query_cor1;

DROP TABLE IF EXISTS tukey_hsd_results_with_JSD;
CREATE TABLE tukey_hsd_results_with_JSD AS SELECT group1, group2, date, mean_diff, lower_bound_ci, upper_bound_ci, KLD, JSD, reject from tukey_hsd_stats_cor1 inner join central_tendency_stats_cor1_new on central_tendency_stats_cor1_new.id = tukey_hsd_stats_cor1.group_1_central_tendency_stats_cor1_id;

DROP TABLE IF EXISTS tukey_hsd_mean_diff_combined_cor1;
CREATE TABLE tukey_hsd_mean_diff_combined_cor1 AS SELECT group1, group2, mean_diff from tukey_hsd_results_with_JSD WHERE date = 'combined';
INSERT INTO tukey_hsd_mean_diff_combined_cor1 VALUES ('COR1', 'COR1', 0);
INSERT INTO tukey_hsd_mean_diff_combined_cor1 VALUES ('ne', 'ne', 0);
INSERT INTO tukey_hsd_mean_diff_combined_cor1 VALUES ('ne_LOS', 'ne_LOS', 0);
INSERT INTO tukey_hsd_mean_diff_combined_cor1 VALUES ('pB', 'pB', 0);
INSERT INTO tukey_hsd_mean_diff_combined_cor1 VALUES ('random', 'random', 0);

DROP TABLE IF EXISTS tukey_hsd_reject_combined_cor1;
CREATE TABLE tukey_hsd_reject_combined_cor1 AS SELECT group1, group2, reject from tukey_hsd_results_with_JSD WHERE date = 'combined';
INSERT INTO tukey_hsd_reject_combined_cor1 VALUES ('COR1', 'COR1', 0);
INSERT INTO tukey_hsd_reject_combined_cor1 VALUES ('ne', 'ne', 0);
INSERT INTO tukey_hsd_reject_combined_cor1 VALUES ('ne_LOS', 'ne_LOS', 0);
INSERT INTO tukey_hsd_reject_combined_cor1 VALUES ('pB', 'pB', 0);
INSERT INTO tukey_hsd_reject_combined_cor1 VALUES ('random', 'random', 0);