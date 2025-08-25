import pandas as pd

df = pd.read_csv('/old/large_dataset.csv')

# only pre-fight data
columns_to_keep = [
"event_name","weight_class","is_title_bout","gender",
"r_wins_total","r_losses_total","r_age","r_height","r_weight","r_reach","r_stance","r_SLpM_total","r_SApM_total","r_sig_str_acc_total","r_td_acc_total","r_str_def_total","r_td_def_total","r_sub_avg","r_td_avg",
"b_wins_total","b_losses_total","b_age","b_height","b_weight","b_reach","b_stance","b_SLpM_total","b_SApM_total","b_sig_str_acc_total","b_td_acc_total","b_str_def_total","b_td_def_total","b_sub_avg","b_td_avg",
"wins_total_diff","losses_total_diff","age_diff","height_diff","weight_diff","reach_diff",
"SLpM_total_diff","SApM_total_diff","sig_str_acc_total_diff","td_acc_total_diff","str_def_total_diff","td_def_total_diff","sub_avg_diff","td_avg_diff"
]

clean_df = df[columns_to_keep].copy()

print(f"Original dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Cleaned dataset: {len(clean_df)} rows, {len(clean_df.columns)} columns")
print(f"Missing values per column:")
print(clean_df.isnull().sum())

clean_df.to_csv('clean_ufc_data.csv', index=False)
print("Saved cleaned data to clean_ufc_data.csv")

