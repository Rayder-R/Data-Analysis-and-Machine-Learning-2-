from ssl import ALERT_DESCRIPTION_ACCESS_DENIED
from matplotlib.font_manager import ttfFontProperty
import pandas as pd
import matplotlib.pyplot as plt
import math

df_telco = pd.read_csv('分群\Telco-Customer-Churn.csv')

# matplotlib 中文設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False


# visualize column names
print(df_telco.columns)

# check unique values of each column
for column in df_telco.columns:
    print('Column: {} - Unique Values: {}'.format(column, df_telco[column].unique()))

print(df_telco.info())

# transform the column TotalCharges into a numeric data type
df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'], errors='coerce')

# null observations of the TotalCharges column
df_telco[df_telco['TotalCharges'].isnull()]

# drop observations with null values
df_telco.dropna(inplace=True)

# drop the customerID column from the dataset
df_telco.drop(columns='customerID', inplace=True)

# unique elements of the PaymentMethod column
df_telco.PaymentMethod.unique()
print(df_telco.PaymentMethod.unique())

# ==================================================================
# create a figure
# fig = plt.figure(figsize=(10, 6)) 
# ax = fig.add_subplot(111)

# # proportion of observation of each class
# prop_response = df_telco['Churn'].value_counts(normalize=True)

# # create a bar plot showing the percentage of churn
# prop_response.plot(kind='bar', 
#                    ax=ax,
#                    color=['springgreen','salmon'])

# # set title and labels
# ax.set_title('Proportion of observations of the response variable',
#              fontsize=18, loc='left')
# ax.set_xlabel('churn',
#               fontsize=14)
# ax.set_ylabel('proportion of observations',
#               fontsize=14)
# ax.tick_params(rotation='auto')

# # eliminate the frame from the plot
# spine_names = ('top', 'right', 'bottom', 'left')
# for spine_name in spine_names:
#     ax.spines[spine_name].set_visible(False)

# plt.show()
# ==================================================================


def percentage_stacked_plot(columns_to_plot, super_title):
    
    '''
    Prints a 100% stacked plot of the response variable for independent variable of the list columns_to_plot.
            Parameters:
                    columns_to_plot (list of string): Names of the variables to plot
                    super_title (string): Super title of the visualization
            Returns:
                    None
    '''
    
    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot)/2)

    # create a figure
    fig = plt.figure(figsize=(12, 5 * number_of_rows)) 
    fig.suptitle(super_title, fontsize=22,  y=.95)
 

    # loop to each column name to create a subplot
    for index, column in enumerate(columns_to_plot, 1):

        # create the subplot
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)

        # calculate the percentage of observations of the response variable for each group of the independent variable
        # 100% stacked bar plot
        prop_by_independent = pd.crosstab(df_telco[column], df_telco['Churn']).apply(lambda x: x/x.sum()*100, axis=1)

        prop_by_independent.plot(kind='bar', ax=ax, stacked=True,
                                 rot=0, color=['springgreen','salmon'])

        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5),
                  title='客戶流失', fancybox=True)

        # set title and labels 
        ax.set_title( column + ' 人數比例分析',
                     fontsize=16, loc='left')

        ax.tick_params(rotation='auto')

        # eliminate the frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)
    plt.show()

# demographic column names 
# ['性別'，'老年人'，'合作夥伴'，'家屬']
demographic_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

# stacked plot of demographic columns
percentage_stacked_plot(demographic_columns, '人口統計分析')


# customer account column names
# （合同、無紙化計費、支付方式）
account_columns = ['Contract', 'PaperlessBilling', 'PaymentMethod']

# stacked plot of customer account columns
percentage_stacked_plot(account_columns, '客戶帳戶分析')
