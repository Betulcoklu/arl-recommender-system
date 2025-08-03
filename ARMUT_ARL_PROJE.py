
#########################
# Business Problem
#########################

# Armut, Turkey's largest online service platform, connects service providers with customers who need services.
# The platform enables easy access to services such as cleaning, renovation, and transportation via computer or smartphone.
# Using the dataset containing users and the services/categories they have purchased, a product recommendation system is to be built using Association Rule Learning.


#########################
# Dataset
#########################
# The dataset consists of services received by customers and the categories of these services.
# Each service includes date and time information.

# UserId: Customer number
# ServiceId: Anonymized services under each category. (e.g., upholstery cleaning under the cleaning category)
# A ServiceId can appear under different categories and represent different services under those categories.
# (e.g., Service with CategoryId 7 and ServiceId 4 is radiator cleaning, while CategoryId 2 and ServiceId 4 is furniture assembly)
# CategoryId: Anonymized categories (e.g., cleaning, transportation, renovation)
# CreateDate: Date when the service was purchased


import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

#########################
# TASK 1: Data Preparation
#########################

# Step 1: Read the armut_data.csv file.
df_ = pd.read_csv("/Users/betulcoklu/Documents/arl-recommender-system/armut_data.csv")
df = df_.copy()

# Step 2: ServiceID represents a different service within each CategoryID.
# Create a new variable representing services by concatenating ServiceID and CategoryID with "_".
df["Service"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]

# Step 3: The dataset contains the date and time when services were received; there is no basket (invoice) definition.
# To apply Association Rule Learning, we need to define a basket. Here, a basket is defined as the set of services a customer receives in a given month.
# For example, customer 7256's services 9_4 and 46_4 in August 2017 form one basket; services 9_4 and 38_4 in October 2017 form another basket.
# Baskets should be identified with a unique ID. For this, create a new date variable containing only year and month, and concatenate UserID and this new date variable with "_".

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")
df["BasketID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]

# Example: View the baskets for a specific user
# print(df[df["UserId"] == 7256 ])


#########################
# TASK 2: Generate Association Rules
#########################

# Step 1: Create a basket-service pivot table as shown below.
#
# Service        0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# BasketID
# 0_2017-08       0     0      0     0      0     0     0     0     0     0..
# 0_2017-09       0     0      0     0      0     0     0     0     0     0..
# 0_2018-01       0     0      0     0      0     0     0     0     0     0..
# 0_2018-04       0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08   0     0      0     0      0     0     0     0     0     0..

invoice_product_df = df.groupby(['BasketID', 'Service'])['Service'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)


# Step 2: Generate association rules.
frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)


# Step 3: Use the arl_recommender function to recommend services to a user who last received service '2_0'.

def arl_recommender(rules_df, product_id, rec_count=1):
    """
    Recommends services based on association rules for a given service.
    Args:
        rules_df (pd.DataFrame): Association rules dataframe.
        product_id (str): Service ID to base recommendations on.
        rec_count (int): Number of recommendations to return.
    Returns:
        list: Recommended services.
    """
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    # Remove duplicates and flatten
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

# Example usage:
recommendations = arl_recommender(rules, "2_0", 4)
print(recommendations)