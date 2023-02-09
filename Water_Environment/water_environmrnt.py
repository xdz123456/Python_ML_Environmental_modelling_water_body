import pandas as pd
import function as fc
import matplotlib.pyplot as plt

# Set the work sheet 1
CHLA = pd.read_excel("Cobbosseecontee Lake.xlsx", sheet_name=0)
CHLA = CHLA[['Date', 'Depth', 'CHLA （mg/L）']]

# Set the work sheet 2
Temp = pd.read_excel("Cobbosseecontee Lake.xlsx", sheet_name=1)
Temp.rename(columns={'DEPTH': 'Depth'}, inplace=True)
Temp = Temp[['Date', 'Depth', 'TEMPERATURE（Centrigrade）']]

# Set the work sheet 3
P = pd.read_excel("Cobbosseecontee Lake.xlsx", sheet_name=2)
P = P[['Date', 'Depth', 'Total P （mg/L）']]

# total data without merge
total = (CHLA.append(Temp, ignore_index=True)).append(P, ignore_index=True)
# Get the mode of depth
mode = total['Depth'].mode()[0]

# Selete the the depth which is mode of total data
CHLA = CHLA[(CHLA['Depth'] == mode)]
Temp = Temp[(Temp['Depth'] == mode)]
P = P[(P['Depth'] == mode)]

# Mean the value of the same depth in same day
CHLA = CHLA.groupby("Date").mean().reset_index()
Temp = Temp.groupby("Date").mean().reset_index()
P = P.groupby("Date").mean().reset_index()

# Merge the data
merge_data = pd.merge(CHLA, Temp, how='outer', on=['Date', 'Depth'])
merge_data = pd.merge(merge_data, P, how='outer', on=['Date', 'Depth'])
# Sort with datetime order
merge_data = merge_data.sort_values(by='Date')
merge_data = merge_data.reset_index(drop=True)

# Pre process the data
pre_process_data = fc.pre_process(merge_data, mode)

# Deal data with Mean value
Mean_value_data = fc.delete_useless_year(pre_process_data.copy())
Mean_value_data = fc.fill_mean_value(Mean_value_data.copy())

# Deal data with KNN
KNN_data = fc.fill_knn(pre_process_data.copy())

# Output as excel
fc.output_as_xlsx(Mean_value_data.copy(), KNN_data.copy())

# Obtain the Total data for calculating the correlation
mean_Total = Mean_value_data[['CHLA （mg/L）', 'TEMPERATURE（Centrigrade）', 'Total P （mg/L）']]
KNN_Total = KNN_data[['CHLA （mg/L）', 'TEMPERATURE（Centrigrade）', 'Total P （mg/L）']]


# Percentage
mean_corr_Percentage = fc.cal_corr_with(mean_Total, 'percbend')
knn_corr_Percentage = fc.cal_corr_with(KNN_Total, 'percbend')
# Shepherd’s pi correlation
mean_corr_Shepherd = fc.cal_corr_with(mean_Total, 'shepherd')
knn_corr_Shepherd = fc.cal_corr_with(KNN_Total, 'shepherd')
# Biweight mid correlation
mean_corr_Biweight = fc.cal_corr_with(mean_Total, 'bicor')
knn_corr_Biweight = fc.cal_corr_with(KNN_Total, 'bicor')
# Skipped spearman correlation
mean_corr_Skipped = fc.cal_corr_with(mean_Total, 'skipped')
knn_corr_Skipped = fc.cal_corr_with(KNN_Total, 'skipped')
# Person
mean_corr_person = mean_Total.corr(method='pearson', min_periods=1).values
knn_corr_person = KNN_Total.corr(method='pearson', min_periods=1).values
# Kendall
mean_corr_kendall = mean_Total.corr(method='kendall', min_periods=1).values
knn_corr_kendall = KNN_Total.corr(method='kendall', min_periods=1).values
# Spearman
mean_corr_spearman = mean_Total.corr(method='spearman', min_periods=1).values
knn_corr_spearman = KNN_Total.corr(method='spearman', min_periods=1).values

# print the Correlation Matrix and importance_rank
# print("Correlation Matrix with Mean value data by Percentage: \n", mean_corr_Percentage, "\n")
# fc.print_importance_rank(mean_corr_Percentage)
print("Correlation Matrix with Mean value data by Shepherd: \n", mean_corr_Shepherd, "\n")
fc.print_importance_rank(mean_corr_Shepherd)
print("Correlation Matrix with Mean value data by Biweight: \n", mean_corr_Biweight, "\n")
fc.print_importance_rank(mean_corr_Biweight)
# print("Correlation Matrix with Mean value data by Skipped spearman: \n", mean_corr_Skipped, "\n")
# fc.print_importance_rank(mean_corr_Skipped)
print("Correlation Matrix with Mean value data by Person: \n", mean_corr_person, "\n")
fc.print_importance_rank(mean_corr_person)
print("Correlation Matrix with Mean value data by Kendall: \n", mean_corr_kendall, "\n")
fc.print_importance_rank(mean_corr_kendall)
print("Correlation Matrix with Mean value data by Spearman: \n", mean_corr_spearman, "\n")
fc.print_importance_rank(mean_corr_spearman)

# print("Correlation Matrix with KNN data by Percentage: \n", knn_corr_Percentage, "\n")
# fc.print_importance_rank(knn_corr_Percentage)
print("Correlation Matrix with KNN data by Shepherd: \n", knn_corr_Shepherd, "\n")
fc.print_importance_rank(knn_corr_Shepherd)
print("Correlation Matrix with KNN data by Biweight: \n", knn_corr_Biweight, "\n")
fc.print_importance_rank(knn_corr_Biweight)
# print("Correlation Matrix with KNN data by Skipped spearman: \n", knn_corr_Skipped, "\n")
# fc.print_importance_rank(knn_corr_Skipped)
print("Correlation Matrix with KNN data by Person: \n", knn_corr_person, "\n")
fc.print_importance_rank(knn_corr_person)
print("Correlation Matrix with KNN data by Kendall: \n", knn_corr_kendall, "\n")
fc.print_importance_rank(knn_corr_kendall)
print("Correlation Matrix with KNN data by Spearman: \n", knn_corr_spearman, "\n")
fc.print_importance_rank(knn_corr_spearman)

