-- Copyright 2025 Christopher Rura

-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at

--     http://www.apache.org/licenses/LICENSE-2.0

-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

CREATE TABLE IF NOT EXISTS qraft_input_variables (
            qraft_parameters_id INTEGER PRIMARY KEY,
            d_phi REAL,
            d_rho REAL,
            XYCenter_x REAL,
            XYCenter_y REAL,
            rot_angle REAL,
            phi_shift REAL,
            smooth_xy INT,
            smooth_phi_rho_lower INT,
            smooth_phi_rho_upper INT,
            detr_phi INT,
            rho_range_lower REAL,
            rho_range_upper REAL,
            n_rho INT,
            p_range_lower REAL,
            p_range_upper REAL,
            n_p INT,
            n_nodes_min INT,
            intensity_removal_coefficient REAL,
            unique(d_phi, d_rho, XYCenter_x, XYCenter_y, rot_angle, phi_shift, smooth_xy, smooth_phi_rho_lower, smooth_phi_rho_upper, detr_phi, rho_range_lower, rho_range_upper, n_rho, p_range_lower, p_range_upper, n_p, n_nodes_min, intensity_removal_coefficient));

CREATE TABLE IF NOT EXISTS forward_input_variables (
            forward_parameters_id INTEGER PRIMARY KEY,
            crln_obs,
            crlt_obs,
            occlt,
            r_sun_range,
            unique(crln_obs, crlt_obs, occlt, r_sun_range));

DROP TABLE IF EXISTS central_tendency_stats_cor1_new;

CREATE TABLE IF NOT EXISTS central_tendency_stats_cor1_new(
    id INTEGER PRIMARY KEY, 
    data_type, 
    data_source, 
    date, 
    mean, 
    median,
    standard_deviation,
    confidence_interval, 
    n,
    Gaussian_JSD,
    Gaussian_KLD,
    kurtosis,
    skewness,
    qraft_parameters_id INTEGER,
    forward_input_data_id INTEGER,  
    FOREIGN KEY(qraft_parameters_id) REFERENCES qraft_input_variables(qraft_parameters_id),
    FOREIGN KEY(forward_input_data_id) REFERENCES forward_input_variables(forward_input_data_id));

CREATE TABLE IF NOT EXISTS central_tendency_stats_cor1_all(
    id INTEGER PRIMARY KEY, 
    data_type, 
    data_source, 
    date, 
    mean, 
    median,
    standard_deviation,
    confidence_interval, 
    n,
    Gaussian_JSD,
    Gaussian_KLD,
    kurtosis,
    skewness,
    qraft_parameters_id INTEGER,
    forward_input_data_id INTEGER,  
    FOREIGN KEY(qraft_parameters_id) REFERENCES qraft_input_variables(qraft_parameters_id),
    FOREIGN KEY(forward_input_data_id) REFERENCES forward_input_variables(forward_input_data_id),
    unique(data_type, data_source, date, mean, median, confidence_interval, n, qraft_parameters_id, forward_input_data_id));

DROP TABLE IF EXISTS central_tendency_stats_kcor_new;

CREATE TABLE IF NOT EXISTS central_tendency_stats_kcor_new(
            id INTEGER PRIMARY KEY,  
            data_type,  
            data_source,  
            date,  
            mean,  
            median,
            standard_deviation,
            confidence_interval,  
            n,
            Gaussian_JSD,
            Gaussian_KLD,
            kurtosis,
            skewness,
            qraft_parameters_id INTEGER,  
            forward_input_data_id INTEGER,  
            FOREIGN KEY(qraft_parameters_id) REFERENCES qraft_input_variables(qraft_parameters_id),
            FOREIGN KEY(forward_input_data_id) REFERENCES forward_input_variables(forward_input_data_id));


CREATE TABLE IF NOT EXISTS central_tendency_stats_kcor_all(
            id INTEGER PRIMARY KEY,  
            data_type,  
            data_source,  
            date,  
            mean,  
            median,
            standard_deviation,
            confidence_interval,  
            n,
            Gaussian_JSD,
            Gaussian_KLD,
            kurtosis,
            skewness,
            qraft_parameters_id INTEGER,  
            forward_input_data_id INTEGER,  
            FOREIGN KEY(qraft_parameters_id) REFERENCES qraft_input_variables(qraft_parameters_id),
            FOREIGN KEY(forward_input_data_id) REFERENCES forward_input_variables(forward_input_data_id),
            unique(data_type, data_source, date, mean, median, confidence_interval, n, qraft_parameters_id, forward_input_data_id));


DROP TABLE IF EXISTS tukey_hsd_stats_cor1;

CREATE TABLE IF NOT EXISTS tukey_hsd_stats_cor1(
    id INTEGER PRIMARY KEY, 
    group1,
    group2,
    mean_diff,
    p_adj,
    lower_bound_ci,
    upper_bound_ci,
    reject boolean,
    KLD,
    JSD,
    group_1_central_tendency_stats_cor1_id INTEGER,
    group_2_central_tendency_stats_cor1_id INTEGER,
    FOREIGN KEY(group_1_central_tendency_stats_cor1_id) REFERENCES central_tendency_stats_cor1_new(id),
    FOREIGN KEY(group_2_central_tendency_stats_cor1_id) REFERENCES central_tendency_stats_cor1_new(id));

DROP TABLE IF EXISTS tukey_hsd_stats_kcor;

CREATE TABLE IF NOT EXISTS tukey_hsd_stats_kcor(
    id INTEGER PRIMARY KEY, 
    group1,
    group2,
    mean_diff,
    p_adj,
    lower_bound_ci,
    upper_bound_ci,
    reject boolean,
    KLD,
    JSD,
    group_1_central_tendency_stats_kcor_id INTEGER,
    group_2_central_tendency_stats_kcor_id INTEGER,
    FOREIGN KEY(group_1_central_tendency_stats_kcor_id) REFERENCES central_tendency_stats_kcor_new(id),
    FOREIGN KEY(group_2_central_tendency_stats_kcor_id) REFERENCES central_tendency_stats_kcor_new(id));

