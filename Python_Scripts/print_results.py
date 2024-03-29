from functions import print_sql_query, plot_sql_query
dbName = "tutorial.db"
# query = "SELECT * from central_tendency_stats_cor1_new order by mean asc;"
# print_sql_query(dbName, query)

# query = "SELECT * from central_tendency_stats_kcor_new order by mean asc;"
# print_sql_query(dbName, query)

# query = "select * from central_tendency_stats_cor1_new inner join forward_input_variables on forward_input_variables.forward_parameters_id = central_tendency_stats_cor1_new.forward_input_data_id order by mean ASC;"
# print_sql_query(dbName, query)

query = "SELECT data_type, data_source, date, mean, median, standard_deviation, n, round(d_phi,5), d_rho, rot_angle, phi_shift, smooth_xy, smooth_phi_rho_lower, smooth_phi_rho_upper from central_tendency_stats_kcor_new inner join qraft_input_variables on qraft_input_variables.qraft_parameters_id = central_tendency_stats_kcor_new.qraft_parameters_id order by mean ASC;"
print_sql_query(dbName, query)

query = "SELECT data_type, data_source, date, mean, median, standard_deviation, n, round(d_phi,5), d_rho, rot_angle, phi_shift, smooth_xy, smooth_phi_rho_lower, smooth_phi_rho_upper from central_tendency_stats_cor1_new inner join qraft_input_variables on qraft_input_variables.qraft_parameters_id = central_tendency_stats_cor1_new.qraft_parameters_id order by mean ASC;"
print_sql_query(dbName, query)

query = "SELECT data_type, data_source, mean, smooth_xy from central_tendency_stats_kcor_all inner join qraft_input_variables on qraft_input_variables.qraft_parameters_id = central_tendency_stats_kcor_all.qraft_parameters_id where date='combined' order by mean ASC;"
print_sql_query(dbName, query)
# plot_sql_query(dbName, query, 'smooth_xy', 'mean')

query = "SELECT date, data_type, data_source, mean, median, standard_deviation, intensity_removal_coefficient from central_tendency_stats_cor1_all inner join qraft_input_variables on qraft_input_variables.qraft_parameters_id = central_tendency_stats_cor1_all.qraft_parameters_id where intensity_removal_coefficient=0.0 and data_source='COR1' order by mean ASC;"
print_sql_query(dbName, query)

query = "SELECT date, data_type, data_source, mean, median, standard_deviation, intensity_removal_coefficient from central_tendency_stats_cor1_new inner join qraft_input_variables on qraft_input_variables.qraft_parameters_id = central_tendency_stats_cor1_new.qraft_parameters_id where intensity_removal_coefficient=0.0 and data_source='COR1' order by mean ASC;"
print_sql_query(dbName, query)

query = "SELECT date, data_type, data_source, mean, median, standard_deviation, intensity_removal_coefficient from central_tendency_stats_cor1_all_naty_original inner join qraft_input_variables on qraft_input_variables.qraft_parameters_id = central_tendency_stats_cor1_all_naty_original.qraft_parameters_id where intensity_removal_coefficient=0.0 and data_source='COR1' order by mean ASC;"
print_sql_query(dbName, query)

# query = "CREATE TABLE central_tendency_stats_cor1_all_naty_original AS SELECT * FROM central_tendency_stats_cor1_all WHERE 0;"
# print_sql_query(dbName, query)