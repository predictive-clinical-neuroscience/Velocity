#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 15:42:43 2025

@author: johbay
"""
from scipy.stats import f_oneway, ttest_ind, shapiro, levene
import numpy as np

def compare_distributions_by_column(df, value_col, group_col, measure_name='measure', alpha=0.05):
    """
    Compare distributions from a DataFrame based on a grouping column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    value_col : str
        Column name containing the values to compare (e.g., 'z_gain')
    group_col : str
        Column name containing the group labels (e.g., 'diagnosis_t2')
    measure_name : str
        Name of the measure for reporting
    alpha : float
        Significance level (default: 0.05)
    
    Returns:
    --------
    dict : Results dictionary with all statistical test results
    
    Example:
    --------
    results = compare_distributions_by_column(
        df=gains_df, 
        value_col='z_gain', 
        group_col='diagnosis_t2',
        measure_name='Cortical Thickness'
    )
    """
    
    # Remove rows with missing data
    df_clean = df.dropna(subset=[value_col, group_col]).copy()
    
    
    counts = df_clean.groupby(group_col)[value_col].count()
    valid_groups = counts[counts >= 10].index
    df_clean = df_clean[df_clean[group_col].isin(valid_groups)]
    
    
    # Get unique groups
    groups = df_clean[group_col].unique()
    n_groups = len(groups)
    
    print(f"\n=== Parametric Analysis for {measure_name} ===")
    print(f"Groups found: {list(groups)}")
    print(f"Number of groups: {n_groups}")
    
    if n_groups < 2:
        print("Error: Need at least 2 groups for comparison")
        return None
    
    # Extract data for each group
    group_data = {}
    group_stats = {}
    
    for group in groups:
        group_values = df_clean[df_clean[group_col] == group][value_col].values
        group_data[group] = group_values
        group_stats[group] = {
            'n': len(group_values),
            'mean': np.mean(group_values),
            'std': np.std(group_values, ddof=1),
            'median': np.median(group_values)
        }
        print(f"{group}: n={len(group_values)}, M={np.mean(group_values):.3f}, SD={np.std(group_values, ddof=1):.3f}")
    
    # Check assumptions
    print(f"\n--- Assumption Checks ---")
    
    # 1. Normality test for each group
    normality_results = {}
    for group, values in group_data.items():
        if len(values) >= 3:
            stat, p = shapiro(values)
            normality_results[group] = {'statistic': stat, 'p_value': p}
            normal = "V" if p > alpha else "X"
            print(f"Normality {group}: W={stat:.3f}, p={p:.3e} {normal}")
        else:
            print(f"Normality {group}: Sample too small (n={len(values)})")
    
    # 2. Equal variances test (Levene's test)
    if n_groups >= 2:
        group_values_list = list(group_data.values())
        levene_stat, levene_p = levene(*group_values_list)
        equal_var = "V" if levene_p > alpha else "X"
        print(f"Equal variances (Levene): W={levene_stat:.3f}, p={levene_p:.3e} {equal_var}")
    
    # 3. ANOVA or t-test
    if n_groups == 2:
        print(f"\n--- Two-sample t-test ---")
        group_names = list(groups)
        t_stat, p_value = ttest_ind(group_data[group_names[0]], group_data[group_names[1]])
        significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"t({len(group_data[group_names[0]]) + len(group_data[group_names[1]]) - 2}) = {t_stat:.3f}, p = {p_value:.3e} {significant}")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group_data[group_names[0]])-1) * group_stats[group_names[0]]['std']**2 + 
                             (len(group_data[group_names[1]])-1) * group_stats[group_names[1]]['std']**2) / 
                            (len(group_data[group_names[0]]) + len(group_data[group_names[1]]) - 2))
        cohens_d = (group_stats[group_names[0]]['mean'] - group_stats[group_names[1]]['mean']) / pooled_std
        print(f"Effect size (Cohen's d) = {cohens_d:.3f}")
        
        main_test_results = {
            'test_type': 't-test',
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d
        }
        
    elif n_groups >= 3:
        print(f"\n--- One-way ANOVA ---")
        group_values_list = list(group_data.values())
        f_stat, anova_p = f_oneway(*group_values_list)
        significant = "***" if anova_p < 0.001 else "**" if anova_p < 0.01 else "*" if anova_p < 0.05 else "ns"
        
        total_n = sum(len(values) for values in group_values_list)
        df_between = n_groups - 1
        df_within = total_n - n_groups
        
        print(f"F({df_between}, {df_within}) = {f_stat:.3f}, p = {anova_p:.3e} {significant}")
        
        # Effect size (eta-squared)
        ss_total = sum(len(values) - 1 for values in group_values_list)
        eta_squared = (f_stat * df_between) / (f_stat * df_between + df_within) if df_within > 0 else 0
        print(f"Effect size (η²) = {eta_squared:.3f}")
        
        main_test_results = {
            'test_type': 'ANOVA',
            'f_statistic': f_stat,
            'p_value': anova_p,
            'eta_squared': eta_squared,
            'df_between': df_between,
            'df_within': df_within
        }
        
        # Post-hoc tests if ANOVA is significant
        posthoc_results = {}
        if anova_p < alpha and n_groups >= 3:
            print(f"\n--- Post-hoc Tests (Pairwise t-tests with Bonferroni correction) ---")
            
            group_names = list(groups)
            n_comparisons = n_groups * (n_groups - 1) // 2
            
            for i in range(n_groups):
                for j in range(i + 1, n_groups):
                    group1, group2 = group_names[i], group_names[j]
                    
                    t_stat, t_p = ttest_ind(group_data[group1], group_data[group2])
                    bonferroni_p = min(t_p * n_comparisons, 1.0)
                    
                    mean_diff = group_stats[group1]['mean'] - group_stats[group2]['mean']
                    sig_marker = "***" if bonferroni_p < 0.001 else "**" if bonferroni_p < 0.01 else "*" if bonferroni_p < 0.05 else "ns"
                    
                    comparison_name = f"{group1} vs {group2}"
                    print(f"{comparison_name}: t={t_stat:.3f}, p={bonferroni_p:.3e} {sig_marker}, Δμ={mean_diff:.3f}")
                    
                    posthoc_results[comparison_name] = {
                        't_statistic': t_stat,
                        'p_value': bonferroni_p,
                        'raw_p_value': t_p,
                        'mean_difference': mean_diff,
                        'significant': bonferroni_p < alpha
                    }
            
            main_test_results['posthoc_tests'] = posthoc_results
    
    # Compile all results
    results = {
        'measure': measure_name,
        #'n_groups': n_groups,
        #'groups': list(groups),
        #'group_statistics': group_stats,
        #'normality_tests': normality_results,
        #'levene_test': {'statistic': levene_stat, 'p_value': levene_p} if n_groups >= 2 else None,
        'main_test': main_test_results,
        'assumptions_met': {
            'normality': all(result.get('p_value', 0) > alpha for result in normality_results.values()),
            'equal_variances': levene_p > alpha if n_groups >= 2 else True
        }
    }
    
    return results