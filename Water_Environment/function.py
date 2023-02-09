import numpy as np
import pandas as pd
from datetime import *
from sklearn.impute import KNNImputer
import pingouin as pg
from pandas import ExcelWriter
import matplotlib.pyplot as plt


# Pre-process the data:
# Include: Selete all the data with the depth of mode
#          Each month remain one data
#          If there have more than one data in same day in same depth, mean them
#          If the data with the completed data, just remain it(with CHLA, temperature, Total P)
#          If there is no completed data, get the mean value of all the day in same month
#          Sort them by the Date
def pre_process(data, mode):
    pre_pro_data = pd.DataFrame(columns=('Date', 'Depth', 'CHLA （mg/L）', 'TEMPERATURE（Centrigrade）', 'Total P （mg/L）'))
    # From 1992 to 2018 year
    for i in range(1992, 2019):
        # From 5 to 10 month
        for j in range(5, 11):
            complete_flag = 0
            temp_data = data[(data['Date'].dt.year == i) & (data['Date'].dt.month == j)]
            temp_data = temp_data.reset_index(drop=True)
            # Process the whole empty data
            if temp_data.empty:
                pre_pro_data.loc[-1] = [date(i, j, 1), mode, np.nan, np.nan, np.nan]  # adding a row
                pre_pro_data.index = pre_pro_data.index + 1  # shifting index
                pre_pro_data = pre_pro_data.sort_index()
            else:
                # Process the complete data
                for k in range(0, temp_data.shape[0]):
                    if (not np.isnan(temp_data.loc[k]["CHLA （mg/L）"])) and (
                            not np.isnan(temp_data.loc[k]['TEMPERATURE（Centrigrade）'])) and (
                            not np.isnan(temp_data.loc[k]['Total P （mg/L）'])):
                        complete_flag = 1
                        pre_pro_data.loc[-1] = [date(i, j, 1), mode, temp_data.loc[k]["CHLA （mg/L）"],
                                                temp_data.loc[k]['TEMPERATURE（Centrigrade）'],
                                                temp_data.loc[k]['Total P （mg/L）']]  # adding a row
                        pre_pro_data.index = pre_pro_data.index + 1  # shifting index
                        pre_pro_data = pre_pro_data.sort_index()
                        # Mean the same month of complete data
                        pre_pro_data = pre_pro_data.groupby("Date").mean().reset_index()
                # Mean the other data
                if complete_flag == 0:
                    total_CHLA = 0
                    count_CHLA = 0
                    total_temp = 0
                    count_temp = 0
                    total_P = 0
                    count_P = 0

                    # Store the total data of each value
                    for k in range(0, temp_data.shape[0]):
                        if not np.isnan(temp_data.loc[k]["CHLA （mg/L）"]):
                            total_CHLA += temp_data.loc[k]["CHLA （mg/L）"]
                            count_CHLA += 1
                        if not np.isnan(temp_data.loc[k]["TEMPERATURE（Centrigrade）"]):
                            total_temp += temp_data.loc[k]["TEMPERATURE（Centrigrade）"]
                            count_temp += 1
                        if not np.isnan(temp_data.loc[k]["Total P （mg/L）"]):
                            total_P += temp_data.loc[k]["Total P （mg/L）"]
                            count_P += 1
                    # To calculate the mean value of the other data
                    if count_CHLA == 0:
                        my_CHLA = np.nan
                    else:
                        my_CHLA = total_CHLA / count_CHLA
                    if count_temp == 0:
                        my_temp = np.nan
                    else:
                        my_temp = total_temp / count_temp

                    if count_P == 0:
                        my_P = np.nan
                    else:
                        my_P = total_P / count_P

                    # Add a row
                    pre_pro_data.loc[-1] = [date(i, j, 1), mode, my_CHLA, my_temp, my_P]  # adding a row
                    pre_pro_data.index = pre_pro_data.index + 1  # shifting index
                    pre_pro_data = pre_pro_data.sort_index()
    # Resort the data by date
    pre_pro_data = pre_pro_data.sort_values(by='Date')
    pre_pro_data = pre_pro_data.reset_index(drop=True)
    return pre_pro_data


# For mean value mathod
# If one year has data which is less than 2 delete the whole year
def delete_useless_year(data):
    # To get which value should be deleted
    count_nan_year = np.zeros(27)
    for i in range(1992, 2019):
        count_nan_ChLA = 0
        count_nan_temp = 0
        count_nan_P = 0
        for j in range(5, 11):
            k = (i - 1992) * 6 + (j - 5)
            if np.isnan(data.loc[k]["CHLA （mg/L）"]):
                count_nan_ChLA += 1
            if np.isnan(data.loc[k]["TEMPERATURE（Centrigrade）"]):
                count_nan_temp += 1
            if np.isnan(data.loc[k]["Total P （mg/L）"]):
                count_nan_P += 1
        if count_nan_ChLA >= 5 or count_nan_temp >= 5 or count_nan_P >= 5:
            count_nan_year[i - 1992] = 1

    # Transform the Datetime type to the int type
    # For delete the data easily
    data.insert(data.shape[1], 'Date_with_int', data['Date'])
    for i in range(0, len(data)):
        data.loc[i:i, 'Date_with_int'] = int(datetime.strftime(data.loc[i]['Date'], "%Y%m%d"))

    # Delete the data which one year has data which is less than 2 delete the whole year
    my_data = data.copy()
    my_data = my_data[~((my_data.Date_with_int >= 19990101) & (my_data.Date_with_int <= 19991231))]
    for i in range(0, len(count_nan_year)):
        if count_nan_year[i] == 1:
            my_data = my_data[~((my_data.Date_with_int >= ((1992 + i) * 10000 + 101)) & (
                    my_data.Date_with_int <= ((1992 + i) * 10000 + 1231)))]

    # Resort the data by date
    my_data = my_data.drop(["Date_with_int"], axis=1)
    my_data = my_data.sort_values(by='Date')
    my_data = my_data.reset_index(drop=True)

    return my_data


# Use the mean value to deal with the data
# Mean the data if it is nan
def mean_data(data, name):
    for n in range(0, len(data)):
        if np.isnan(data.loc[n][name]):
            # if last row have the value and next row have the data
            # should not be the 5th or 10th month
            if n - 1 > 0 and n + 1 < len(data) and n % 6 != 0 and n % 6 != 5 and (not np.isnan(data.loc[n - 1][name])) and (
                    not np.isnan(data.loc[n + 1][name])):
                data.loc[n:n, name] = (data.loc[n - 1][name] + data.loc[n + 1][name]) / 2
            # if last two row is not nan, mean them
            # should not be 1th or 2th month
            elif n - 2 > 0 and n % 6 > 1 and (not np.isnan(data.loc[n - 1][name])) and (
                    not np.isnan(data.loc[n - 2][name])):
                if data.loc[n - 1][name] > data.loc[n - 2][name]:
                    data.loc[n:n, name] = abs(data.loc[n - 1][name] - data.loc[n - 2][name]) + data.loc[n - 1][name]
                else:
                    data.loc[n:n, name] = data.loc[n - 1][name] - abs(data.loc[n - 1][name] - data.loc[n - 2][name])
            # if next two row is not nan, mean them
            # should not be 9th or 10th month
            elif n + 2 < len(data) and n % 6 < 4 and (not np.isnan(data.loc[n + 1][name])) and (
                    not np.isnan(data.loc[n + 2][name])):
                if data.loc[n + 1][name] > data.loc[n + 2][name]:
                    data.loc[n:n, name] = data.loc[n + 1][name] + abs(data.loc[n + 1][name] - data.loc[n + 2][name])
                else:
                    data.loc[n:n, name] = data.loc[n + 1][name] - abs(data.loc[n + 1][name] - data.loc[n + 2][name])

    # Make a arithmetic_matrix with the first term and step
    number_year = (int(data.shape[0] / 6))
    arithmetic_matrix = np.zeros((number_year, 2))
    # Use the first term and step complete all the nan data
    # construct the arithmetic matrix
    for i in range(0, number_year):
        first_not_empty = -1
        first_not_empty_index = -1
        second_not_empty = -1
        second_not_empty_index = -1
        for j in range(0, 6):
            if (not np.isnan(data.loc[(i * 6) + j][name])) and first_not_empty_index < 0 and second_not_empty_index < 0:
                first_not_empty = data.loc[(i * 6) + j][name]
                first_not_empty_index = j
                second_not_empty_index = 0
            elif (not np.isnan(data.loc[(i * 6) + j][name])) and second_not_empty_index == 0:
                second_not_empty = data.loc[(i * 6) + j][name]
                second_not_empty_index = j

        step = abs((second_not_empty - first_not_empty) / (second_not_empty_index - first_not_empty_index))
        First_term = first_not_empty + first_not_empty_index * step
        arithmetic_matrix[i, 0] = First_term
        arithmetic_matrix[i, 1] = step
    # complete them
    for n in range(0, len(data)):
        if np.isnan(data.loc[n][name]):
            data.loc[n:n, name] = arithmetic_matrix[int(n / 6), 0] + (n % 6) * arithmetic_matrix[int(n / 6), 1]
    return data


# Fill the data with the method of mean value
def fill_mean_value(final):
    final = mean_data(final, 'CHLA （mg/L）')
    final = mean_data(final, 'TEMPERATURE（Centrigrade）')
    final = mean_data(final, 'Total P （mg/L）')
    return final


# Fill the data with the method of KNN
def fill_knn(final):
    imputer = KNNImputer(n_neighbors=3)
    final_filled = imputer.fit_transform(final[['CHLA （mg/L）', 'TEMPERATURE（Centrigrade）', 'Total P （mg/L）']])
    for n in range(0, len(final)):
        final.loc[n:n, 'CHLA （mg/L）'] = final_filled[n, 0]
        final.loc[n:n, 'TEMPERATURE（Centrigrade）'] = final_filled[n, 1]
        final.loc[n:n, 'Total P （mg/L）'] = final_filled[n, 2]
    return final


# Calculate the Correlation Matrix with different method
def cal_corr_with(data, method):
    name_list = ['CHLA （mg/L）', 'TEMPERATURE（Centrigrade）', 'Total P （mg/L）']
    my_cor_matrix = np.zeros([3, 3])
    for i in range(0, 3):
        for j in range(0, 3):
            if i == j:
                my_cor_matrix[i][j] = 1.
            else:
                my_cor_matrix[i][j] = \
                    pg.corr(data[name_list[i]], data[name_list[j]], method=method).round(3).loc[method]['r']

    return my_cor_matrix


# Give back the useless data
def reprocess(data):
    new = pd.DataFrame(columns=(
        'MIDAS', 'Lake', 'Town(s)', 'Station', 'Date', 'Depth', 'CHLA （mg/L）', 'TEMPERATURE（Centrigrade）',
        'Total P （mg/L）'))
    for i in range(0, len(data)):
        new.loc[-1] = ['5236', 'Cobbosseecontee Lake', 'Litchfield, Manchester, Monmouth, West Gardiner, W', '1',
                       datetime.strftime(data.loc[i]['Date'], "%Y-%m"), data.loc[i]['Depth'],
                       data.loc[i]['CHLA （mg/L）'], data.loc[i]['TEMPERATURE（Centrigrade）'],
                       data.loc[i]['Total P （mg/L）']]  # adding a row
        new.index = new.index + 1  # shifting index
        new = new.sort_index()
    new = new.sort_values(by='Date')
    new = new.reset_index(drop=True)
    return new


# Output by excel
def output_as_xlsx(mean_value_data, knn_data):
    mean_value_data = reprocess(mean_value_data)
    knn_data = reprocess(knn_data)
    writer = ExcelWriter('Cobbosseecontee_Lake_complete.xlsx')
    mean_value_data.to_excel(writer, 'Mean_value_data', index=False)
    knn_data.to_excel(writer, 'KNN_data', index=False)
    wb = writer.book
    ws1 = wb[wb.sheetnames[0]]
    ws1.column_dimensions['B'].width = 25.0
    ws1.column_dimensions['C'].width = 35.0
    ws1.column_dimensions['G'].width = 20.0
    ws1.column_dimensions['H'].width = 20.0
    ws1.column_dimensions['I'].width = 20.0

    ws2 = wb[wb.sheetnames[1]]
    ws2.column_dimensions['B'].width = 25.0
    ws2.column_dimensions['C'].width = 35.0
    ws2.column_dimensions['G'].width = 20.0
    ws2.column_dimensions['H'].width = 20.0
    ws2.column_dimensions['I'].width = 20.0
    writer.save()
    return


# Print the the associated factors importance ranking.
def print_importance_rank(corr_matrix):
    if abs(corr_matrix[0, 1]) >= abs(corr_matrix[0, 2]):
        print("Correlation with CHLA content: Temperature is stronger than phosphorus content.\n")
    else:
        print("Correlation with CHLA content: Phosphorus content is stronger than temperature.\n")

    if abs(corr_matrix[1, 0]) >= abs(corr_matrix[1, 2]):
        print("Correlation with Temperature: CHLA content is stronger than phosphorus content.\n")
    else:
        print("Correlation with Temperature: Phosphorus content is stronger than CHLA content.\n")

    if abs(corr_matrix[2, 0]) >= abs(corr_matrix[2, 1]):
        print("Correlation with Phosphorus content: CHLA content is stronger than temperature.\n")
    else:
        print("Correlation with Phosphorus content: Temperature is stronger than CHLA content.\n")

    print("\n")
    return


# # Plot the value
# def plot_value(data):
#
    # Temp_plot = data.copy()[['Date', 'TEMPERATURE（Centrigrade）']]
    # Temp_plot.rename(columns={'TEMPERATURE（Centrigrade）': 'TEMPERATURE(Centrigrade)'}, inplace=True)
    # Temp_plot.set_index(["Date"], inplace=True)
    # ax = Temp_plot.plot(x_compat=True, grid=True)
    # ax.set_ylabel('TEMPERATURE(Centrigrade)')
    # ax.set_title("Mean_Value_data")
    # plt.show()


