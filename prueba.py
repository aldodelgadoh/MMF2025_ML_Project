#Modules
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os
from tqdm import tqdm

print(os.listdir("data/"))

#Loading Data
application_train = pd.read_csv('data/application_train.csv')
pos_cash_balance = pd.read_csv('data/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('data/bureau_balance.csv')
previous_application = pd.read_csv('data/previous_application.csv')
installments_payments = pd.read_csv('data/installments_payments.csv')
credit_card_balance = pd.read_csv('data/credit_card_balance.csv')
bureau = pd.read_csv('data/bureau.csv')
application_test = pd.read_csv('data/application_test.csv')
description_df = pd.read_csv('data/HomeCredit_columns_description.csv', encoding='latin1')

#Datasets dictionary for later
datasets = {
    'application_train': application_train,
    'POS_CASH_balance': pos_cash_balance,
    'bureau_balance': bureau_balance,
    'previous_application': previous_application,
    'installments_payments': installments_payments,
    'credit_card_balance': credit_card_balance,
    'bureau': bureau,
    'application_test': application_test
}

#Data Exploration and Data Management
def explore_datasets(datasets, description_df):
    # Print dataset sizes
    for name, data in datasets.items():
        print(f'Size of {name} data: {data.shape}')
    
    # Print dataset columns
    for name, data in datasets.items():
        print(f'{name} Columns: ' + ", ".join(data.columns.values))
    
    summary_list = []
    for name, df in datasets.items():
        print(f"Exploring dataset: {name}")
        for column in df.columns:
            missing_percent = df[column].isnull().mean() * 100
            unique_count = df[column].nunique()
            data_type = df[column].dtype
            
            # Find the description for the column (if it exists)
            description = description_df.loc[
                description_df['Row'] == column, 'Description'
            ].values
            
            description = description[0] if len(description) > 0 else 'No description available'
            
            summary_list.append({
                'Dataset': name,
                'Feature': column,
                '% Missing': round(missing_percent, 2),
                'Unique Count': unique_count,
                'Data Type': data_type,
                'Description': description
            })
    
    summary_df = pd.DataFrame(summary_list)
    return summary_df
summary = explore_datasets(datasets, description_df)
#I created an Excel for better visualization and exploration
summary.to_excel('datasets_summary.xlsx', index=False) 

'''
Exploratory Data Analysis Functions
'''
def plot_stats(df, feature, target='TARGET'):
    """
    Plots the value counts and target percentage for a categorical feature.
    """
    if feature not in df.columns:
        print(f"Feature {feature} not found in the dataset.")
        return
    
    temp = df[feature].value_counts(dropna=True)
    df1 = pd.DataFrame({feature: temp.index, 'Number of Records': temp.values})
    
    if target and target in df.columns:
        # Calculate the percentage of target=1 per category value
        cat_perc = df[[feature, target]].groupby([feature], as_index=False).mean()
        cat_perc.sort_values(by=target, ascending=False, inplace=True)
    
    fig, axes = plt.subplots(ncols=2 if target and target in df.columns else 1, figsize=(14, 6))
    
    sns.set_color_codes("pastel")
    
    # Plot Number of Records
    if target and target in df.columns:
        sns.barplot(ax=axes[0], x=feature, y="Number of Records", data=df1, palette="viridis")
        # Plot TARGET Percentage
        sns.barplot(ax=axes[1], x=feature, y=target, data=cat_perc, palette="magma")
    else:
        sns.barplot(ax=axes, x=feature, y="Number of Records", data=df1, palette="viridis")
    plt.tight_layout()
    #plt.savefig(f"Stats_{feature}.png")    
    plt.show(block=False)
    plt.close()

def plot_distribution(df, feature, target='TARGET'):

    """
    Plots the distribution of a numerical feature, optionally split by TARGET.
    """
    if feature not in df.columns:
        print(f"Feature {feature} not found in the dataset.")
        return
    
    plt.figure(figsize=(10,6))
    
    if target and target in df.columns:
        t1 = df[df[target] != 0]
        t0 = df[df[target] == 0]
        sns.kdeplot(t1[feature].dropna(), bw_adjust=0.5, label="TARGET = 1", shade=True)
        sns.kdeplot(t0[feature].dropna(), bw_adjust=0.5, label="TARGET = 0", shade=True)
        plt.title(f'Distribution of {feature} by TARGET', fontsize=16)
    else:
        sns.kdeplot(df[feature].dropna(), bw_adjust=0.5, label=feature, shade=True)
        plt.title(f'Distribution of {feature} )', fontsize=16)
    
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend()
    #plt.savefig(f"Dist_{feature}.png")
    plt.show(block=False)
    plt.close()

def plot_categorical_distribution(series):
    
    # Ensure the series has a name
    feature_name = series.name if series.name else 'Category'
    
    # Calculate value counts and percentages
    value_counts = series.value_counts()
    percentages = series.value_counts(normalize=True) * 100

    # Combine into a DataFrame
    distribution_df = pd.DataFrame({
        feature_name: value_counts.index,
        'Count': value_counts.values,
        'Percentage': percentages.values
    }).reset_index(drop=True)

    # Plotting with Matplotlib
    plt.figure(figsize=(10, 6))
    bars = plt.bar(distribution_df[feature_name], distribution_df['Percentage'], color='skyblue', edgecolor='black')
    
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title(f'Distribution of {feature_name}', fontsize=14)
    plt.ylim(0, distribution_df['Percentage'].max() + 10)
    
    # Rotate x-axis labels if there are many categories
    if distribution_df[feature_name].nunique() > 10:
        plt.xticks(rotation=45, ha='right')
    else:
        plt.xticks(rotation=0)
    
    # Adding percentage labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height + 0.5, 
            f'{height:.1f}%', 
            ha='center', 
            va='bottom',
            fontsize=10
        )

    plt.tight_layout()
    #plt.savefig(f'Catgorical_{series.name}.png')
    plt.show(block=False)
    plt.close()


'''
Application Train and Test Data Cleaning and preparation
'''
#Checking Target Data
plot_categorical_distribution(application_train['TARGET'])

#If I made some calculated features, I will have to apply to train and test set, so I
#create this database:
application_train['is_train'] = 1
application_test['is_train'] = 0

df = pd.concat([application_train, application_test], ignore_index=True)

#Exploring Amount Income since it will be important
print(application_test['AMT_INCOME_TOTAL'].max())
print(application_train['AMT_INCOME_TOTAL'].max())
plot_distribution(application_train[application_train['AMT_INCOME_TOTAL'] < 5_000_000],'AMT_INCOME_TOTAL')
print(application_train['AMT_INCOME_TOTAL'][application_train['AMT_INCOME_TOTAL'] > 10_000_000].value_counts())
print(application_test['AMT_INCOME_TOTAL'][application_test['AMT_INCOME_TOTAL'] > 10_000_000].value_counts())
print(application_train['AMT_INCOME_TOTAL'][application_train['AMT_INCOME_TOTAL'] > 6_000_000].value_counts())
print(application_test['AMT_INCOME_TOTAL'][application_test['AMT_INCOME_TOTAL'] > 6_000_000].value_counts())

#There are only 5 data points over 6,000,000 in train and 0 data poitns in test
df = df[df['AMT_INCOME_TOTAL'] < 6_000_000] 
(df['AMT_INCOME_TOTAL']/6000000).describe()
plot_distribution(df,'AMT_INCOME_TOTAL')

#Exploring Loan data
loan_amt = ['AMT_CREDIT','AMT_GOODS_PRICE']
for amt in loan_amt:
    print((application_test[amt]/1_000_000).describe())
    print((application_train[amt]/1_000_000).describe())
    plot_distribution(application_train[application_train[amt] < 3_000_000],amt)
    print(application_test[amt][application_test[amt] > 3_000_000].value_counts())
    print(application_train[amt][application_train[amt] > 3_000_000].value_counts())
    plot_distribution(df,amt)
#Seems good, not necesary modification 

print((application_test['AMT_ANNUITY']/100_000).describe())
print((application_train['AMT_ANNUITY']/100_000).describe())
plot_distribution(application_train[application_train['AMT_ANNUITY'] < 150_000],amt)
print(application_test['AMT_ANNUITY'][application_test['AMT_ANNUITY'] > 150_000].value_counts())
print(application_train['AMT_ANNUITY'][application_train['AMT_ANNUITY'] > 150_000].value_counts())
plot_distribution(df,'AMT_ANNUITY')
#Seems good, not necesary modification 


#Exploring Gender
plot_categorical_distribution(application_train['CODE_GENDER'])
plot_stats(df, 'CODE_GENDER', target = 'TARGET')
df['CODE_GENDER'].value_counts()
df = df[df['CODE_GENDER'] != 'XNA'] #very few people has XNA code gender

#Correcting Days Employed since there are anomalous data points
plot_distribution(df,'DAYS_EMPLOYED')
print(df['DAYS_EMPLOYED'][df['DAYS_EMPLOYED']>300000].value_counts())       
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

#There is regulation against using the "age" as a feature for credit assessment. However, to my knowledge that is only
#applicabple to "banks", so in this case, we can use this feature.
plot_distribution(df,'DAYS_BIRTH')
(df['DAYS_BIRTH']/365).describe()
def get_age_label(days_birth):
    """ Return the age group label (int). """
    age_years = -days_birth / 365
    if age_years < 35: return 1
    elif age_years < 45: return 2
    elif age_years < 55: return 3
    elif age_years < 99: return 4
    else: return 0

df['AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: get_age_label(x))
plot_stats(df,'AGE_RANGE')

#EXT_SOURCE_* are scores from external data sources so they can be hepful for this case
ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
for ext in ext_sources:
    plot_distribution(df,ext)

#Let's use some metrics around EXT_SOURCE feautures:
for function_name in ['min', 'max', 'mean']:
        feature_name = 'EXT_SOURCES_{}'.format(function_name.upper())
        df[feature_name] = eval('np.{}'.format(function_name))(
            df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
        


ext_calc = ['EXT_SOURCES_MIN','EXT_SOURCES_MAX','EXT_SOURCES_MEAN']
for ext in ext_calc:
    plot_distribution(df,ext)


docs = [f for f in df.columns if 'FLAG_DOC' in f]
df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)

#Exploring other categorical columns
columns = [
    "NAME_CONTRACT_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_CONT_MOBILE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    "REG_REGION_NOT_LIVE_REGION",
    "REG_REGION_NOT_WORK_REGION",
    "LIVE_REGION_NOT_WORK_REGION",
    "REG_CITY_NOT_LIVE_CITY",
    "REG_CITY_NOT_WORK_CITY",
    "LIVE_CITY_NOT_WORK_CITY"
]

for cat in columns:
    plot_categorical_distribution(df[cat])
    plot_stats(df,cat)

#The rest of the columns are normalized information about building, 
#I am going to live them as presented in the original table.

#Creating new features.
# Credit ratios
df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
# Income ratios
df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']
# Time ratios
df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    

#One-Hot Encode Categorical Features
categorical_features = [col for col in df.columns if df[col].dtype == 'object']
print(f'Categorical features to encode: {categorical_features}')
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)  # drop_first to avoid dummy variable trap


# Preprocess Auxiliary Datasets

#Pre-processing bureau_balance
# Aggregating bureau_balance
buro_balance_grouped = bureau_balance.groupby('SK_ID_BUREAU').agg({
    'MONTHS_BALANCE': ['size', 'min', 'max'],
    'STATUS': lambda x: x.value_counts().to_dict()
})

# Flatten MultiIndex columns
buro_balance_grouped.columns = ['MONTHS_COUNT', 'MONTHS_MIN', 'MONTHS_MAX', 'STATUS_DICT']

# Expand the STATUS_DICT into separate columns
status_df = buro_balance_grouped['STATUS_DICT'].apply(pd.Series).fillna(0)
status_df = status_df.add_prefix('STATUS_')

buro_balance_processed = pd.concat([buro_balance_grouped.drop('STATUS_DICT', axis=1), status_df], axis=1)
bureau = bureau.merge(buro_balance_processed, how='left', on='SK_ID_BUREAU')

# Fill NaNs resulting from the merge
bureau.fillna(0, inplace=True)

# Aggregate bureau features at SK_ID_CURR level
avg_buro = bureau.groupby('SK_ID_CURR').agg({
    'MONTHS_COUNT': 'mean',
    'MONTHS_MIN': 'mean',
    'MONTHS_MAX': 'mean',
    'STATUS_0': 'mean',
    'STATUS_1': 'mean',
    'STATUS_2': 'mean',
    'STATUS_3': 'mean',
    'STATUS_4': 'mean',
    'STATUS_5': 'mean',
    'STATUS_C': 'mean',
    'STATUS_X': 'mean'
})

# Add count of bureau entries per SK_ID_CURR
avg_buro['buro_count'] = bureau.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count()

#Pre-processing previous_application

# One-hot encode categorical features
prev_cat_features = [col for col in previous_application.columns if previous_application[col].dtype == 'object']
previous_application_encoded = pd.get_dummies(previous_application, columns=prev_cat_features, drop_first=True)

# Aggregate features by SK_ID_CURR
avg_prev = previous_application_encoded.groupby('SK_ID_CURR').mean()

# Count number of previous applications per SK_ID_CURR
cnt_prev = previous_application_encoded.groupby('SK_ID_CURR').size().rename('nb_app')

# Combine aggregated features with counts
avg_prev = avg_prev.merge(cnt_prev, left_index=True, right_index=True)

#Pre-processing bureau
# One-hot encode categorical features
buro_cat_features = [col for col in bureau.columns if bureau[col].dtype == 'object']
bureau_encoded = pd.get_dummies(bureau, columns=buro_cat_features, drop_first=True)

# Aggregate features by SK_ID_CURR
avg_bureau = bureau_encoded.groupby('SK_ID_CURR').mean()

# Add count of bureau entries per SK_ID_CURR if not already added
if 'buro_count' not in avg_buro.columns:
    avg_bureau['buro_count'] = bureau_encoded.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count()

#Pre-processing POS_CASH_balance

# Initialize LabelEncoder
le_pos_cash = LabelEncoder()

# Encode 'NAME_CONTRACT_STATUS'
df_POS_CASH = pos_cash_balance.copy()
df_POS_CASH['NAME_CONTRACT_STATUS'] = le_pos_cash.fit_transform(df_POS_CASH['NAME_CONTRACT_STATUS'].astype(str))

# Aggregate features by SK_ID_CURR
nunique_status = df_POS_CASH.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].nunique().rename('POS_NUNIQUE_STATUS')
max_status = df_POS_CASH.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].max().rename('POS_NUNIQUE_STATUS2')

df_POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

# Combine aggregated features
avg_POS_CASH = df_POS_CASH.groupby('SK_ID_CURR').mean().merge(nunique_status, on='SK_ID_CURR').merge(max_status, on='SK_ID_CURR')

#Pre-processing credit_card_balance

# Initialize LabelEncoder
le_credit = LabelEncoder()

# Encode 'NAME_CONTRACT_STATUS'
df_credit = credit_card_balance.copy()
df_credit['NAME_CONTRACT_STATUS'] = le_credit.fit_transform(df_credit['NAME_CONTRACT_STATUS'].astype(str))

# Aggregate features by SK_ID_CURR
nunique_status_credit = df_credit.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].nunique().rename('CC_NUNIQUE_STATUS')
max_status_credit = df_credit.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].max().rename('CC_NUNIQUE_STATUS2')

df_credit.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

# Combine aggregated features
avg_credit = df_credit.groupby('SK_ID_CURR').mean().merge(nunique_status_credit, on='SK_ID_CURR').merge(max_status_credit, on='SK_ID_CURR')

#Pre-processing installments_payments

# Aggregate installments_payments by SK_ID_CURR
avg_payments = installments_payments.groupby('SK_ID_CURR').mean()
avg_payments2 = installments_payments.groupby('SK_ID_CURR').max()
avg_payments3 = installments_payments.groupby('SK_ID_CURR').min()

# Drop SK_ID_PREV if present
if 'SK_ID_PREV' in avg_payments.columns:
    del avg_payments['SK_ID_PREV']
if 'SK_ID_PREV' in avg_payments2.columns:
    del avg_payments2['SK_ID_PREV']
if 'SK_ID_PREV' in avg_payments3.columns:
    del avg_payments3['SK_ID_PREV']


# 4. Merge the Processed Features into the Train and Test Sets
# Split the concatenated DataFrame  into train and test
data = df[df['is_train'] == 1].copy()  # Training set
test = df[df['is_train'] == 0].copy()  # Testing set

data.drop(['is_train'], axis=1, inplace=True)
test.drop(['is_train'], axis=1, inplace=True)

# Merge avg_prev
data = data.merge(avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(avg_prev.reset_index(), how='left', on='SK_ID_CURR')

# Merge avg_buro
data = data.merge(avg_buro.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(avg_buro.reset_index(), how='left', on='SK_ID_CURR')

# Merge avg_POS_CASH
data = data.merge(avg_POS_CASH.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(avg_POS_CASH.reset_index(), how='left', on='SK_ID_CURR')

# Merge avg_credit
data = data.merge(avg_credit.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(avg_credit.reset_index(), how='left', on='SK_ID_CURR')

# Merge avg_payments
data = data.merge(avg_payments.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(avg_payments.reset_index(), how='left', on='SK_ID_CURR')

# Merge avg_payments2
data = data.merge(avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(avg_payments2.reset_index(), how='left', on='SK_ID_CURR')

# Merge avg_payments3
data = data.merge(avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(avg_payments3.reset_index(), how='left', on='SK_ID_CURR')

test = test[test.columns[data.isnull().mean() < 0.80]]
data = data[data.columns[data.isnull().mean() < 0.80]]
print('All databases joined')
