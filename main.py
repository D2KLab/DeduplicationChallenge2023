# -*- coding: utf-8 -*-
"""Deduplication Challenge 2023 - Code

Original file is located at
    https://colab.research.google.com/drive/1VLIkrP552ZQh6Qk-erVu-OmDU4gtV33k
"""

import os
from enum import Enum
from sentence_transformers import SentenceTransformer, util
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import swifter


# Define an Enum class used for identifying each type of duplicate
class TypeEnum(Enum):
  FULL = "FULL"
  SEMANTIC = "SEMANTIC"
  TEMPORAL = "TEMPORAL"
  PARTIAL = "PARTIAL"
  NON_DUPLICATE = "NON_DUPLICATE"

  def __str__(self):
    return f'{self.name}'


# Set the base path as the absolute path of the current directory
BASE_PATH = os.path.abspath('./')

# Read in the dataset file provided for the challenge and store it in a DataFrame
# We set the lineterminator argument to '\n' to avoid issues with line endings
clean_data = pd.read_csv(os.path.join(BASE_PATH, 'wi_dataset.csv'), lineterminator='\n')

# Create a copy of the 'clean_data' dataframe
full_duplicates = clean_data.copy()

# Fill any missing values in the specified columns with empty strings
full_duplicates['title'].fillna('', inplace=True)
full_duplicates['description'].fillna('', inplace=True)
full_duplicates['company_name'].fillna('', inplace=True)
full_duplicates['location'].fillna('', inplace=True)
full_duplicates['country_id'].fillna('', inplace=True)

# Convert the 'title' and 'description' columns to lowercase
full_duplicates['title'] = full_duplicates['title'].str.lower()
full_duplicates['description'] = full_duplicates['description'].str.lower()

# Merge the 'full_duplicates' dataframe with itself based on matching values in the specified columns
# This will result in a dataframe containing all pairs of duplicate rows
full_duplicates = full_duplicates.merge(full_duplicates, on=['title', 'description', 'company_name', 'location', 'country_id'])

# Filter the dataframe to only include rows where 'id_x' is less than 'id_y'
# This ensures that each pair of duplicates is only counted once
full_duplicates = full_duplicates[full_duplicates['id_x'] < full_duplicates['id_y']]

# Keep only the 'id_x' and 'id_y' columns
full_duplicates = full_duplicates[['id_x', 'id_y']]

# Rename the 'id_x' and 'id_y' columns to 'id1' and 'id2', respectively
full_duplicates.rename({ 'id_x': 'id1', 'id_y': 'id2' }, axis=1, inplace=True)

# Add a new 'type' column to the dataframe and set its value to the string representation of 'TypeEnum.FULL'
full_duplicates['type'] = str(TypeEnum.FULL)

# Load the SentenceTransformer model with the LaBSE architecture
model = SentenceTransformer('sentence-transformers/LaBSE')

# Get the 'id' and `title` columns from the 'clean_data' dataframe
ids = clean_data['id'].tolist()
titles = clean_data['title'].fillna('').tolist()

# Use the 'util.paraphrase_mining' function to find paraphrases of the 'titles' list using the SentenceTransformer model
title_paraphrases = util.paraphrase_mining(model, titles, top_k=200, max_pairs=np.inf, show_progress_bar=True)
df_title_paraphrases = pd.DataFrame(title_paraphrases, columns=['score', 'i', 'j'])

# Get the 'title' and 'description' columns from the 'clean_data' dataframe, fill NaN values with empty strings,
# concatenate them with a period separator, and convert them to a list
texts = clean_data[['title', 'description']].fillna('').apply(lambda x: '. '.join(x), axis=1).tolist()

# Use the 'util.paraphrase_mining' function to find paraphrases of the 'texts' list using the SentenceTransformer model
texts_paraphrases = util.paraphrase_mining(model, texts, top_k=200, max_pairs=np.inf, show_progress_bar=True)
df_texts_paraphrases = pd.DataFrame(texts_paraphrases, columns=['score', 'i', 'j'])

# Re-order the indices of the 'df_title_paraphrases' and 'df_texts_paraphrases' dataframes so that 'i' is always less than 'j'
df_title_paraphrases_reordered = df_title_paraphrases.copy()
df_title_paraphrases_reordered.loc[
  df_title_paraphrases_reordered['i'] > df_title_paraphrases_reordered['j'], ['i', 'j']
] = df_title_paraphrases_reordered.loc[df_title_paraphrases_reordered['i'] > df_title_paraphrases_reordered['j'], ['j', 'i']].values

df_texts_paraphrases_reordered = df_texts_paraphrases.copy()
df_texts_paraphrases_reordered.loc[
  df_texts_paraphrases_reordered['i'] > df_texts_paraphrases_reordered['j'], ['i', 'j']
] = df_texts_paraphrases_reordered.loc[df_texts_paraphrases_reordered['i'] > df_texts_paraphrases_reordered['j'], ['j', 'i']].values

# Set weights for merging the two dataframes
df1_weight = 0.7
df2_weight = 0.3

# Merge title and texts dataframes based on their index (i, j) columns with their defined weights
df_merged = pd.merge(df_title_paraphrases_reordered, df_texts_paraphrases_reordered, on=['i', 'j'], how='outer')
df_merged['score'] = df_merged['score_x'] * df1_weight + df_merged['score_y'] * df2_weight
df_weighted_average = df_merged[['i', 'j', 'score', 'score_x', 'score_y']]
df_weighted_average.rename({ 'score_x': 'score_title', 'score_y': 'score_text' }, axis=1, inplace=True)

# Define a function to add ids to dataframe rows
def add_ids(row):
  return (ids[row['i'].astype(np.int64)], ids[row['j'].astype(np.int64)])

# Define a function to create a dataframe of duplicates from the weighted average dataframe and the clean_data dataframe
def create_df_duplicates(paraphrases_df, data):
  df_duplicates = paraphrases_df.copy()
  # Add 'id1' and 'id2' columns to the copied dataframe using the 'add_ids' function
  df_duplicates[['id1', 'id2']] = df_duplicates.swifter.progress_bar(enable=True, desc=None).apply(add_ids, axis=1, result_type='expand')
  # Remove 'i' and 'j' columns from the copied dataframe
  df_duplicates = df_duplicates.drop(['i', 'j'], axis=1)
  # Merge the copied dataframe with the clean_data dataframe using 'id1' and 'id2' columns
  df_duplicates = df_duplicates.merge(data, left_on='id1', right_on='id').merge(data, left_on='id2', right_on='id')
  return df_duplicates

# Create a dataframe of semantic duplicates
df_semantic_duplicates = create_df_duplicates(df_weighted_average, clean_data)

# Filter the semantic duplicates to select only rows with duplicates that meet certain conditions such as
# having a score greater than or equal to 0.7 and matching on company name, location, or country ID.
df = df_semantic_duplicates[
  (df_semantic_duplicates['score'] >= 0.7)
  &
  (
    (df_semantic_duplicates['company_name_x'] == df_semantic_duplicates['company_name_y'])
    |
    (df_semantic_duplicates['location_x'] == df_semantic_duplicates['location_y'])
    |
    (df_semantic_duplicates['country_id_x'] == df_semantic_duplicates['country_id_y'])
  )
].copy()

# Define Boolean masks to select rows with missing values in certain columns and non-missing values in others
mask_company_x = df['company_name_x'].isna() & df['company_name_y'].notna() & (df['country_id_x'] == df['country_id_y']) & (df['location_x'] == df['location_y'])
mask_company_y = df['company_name_y'].isna() & df['company_name_x'].notna() & (df['country_id_x'] == df['country_id_y']) & (df['location_x'] == df['location_y'])
mask_country_x = df['country_id_x'].isna() & df['country_id_y'].notna() & (df['company_name_x'] == df['company_name_y']) & (df['location_x'] == df['location_y'])
mask_country_y = df['country_id_y'].isna() & df['country_id_x'].notna() & (df['company_name_x'] == df['company_name_y']) & (df['location_x'] == df['location_y'])
mask_location_x = df['location_x'].isna() & df['location_y'].notna() & (df['company_name_x'] == df['company_name_y']) & (df['country_id_x'] == df['country_id_y'])
mask_location_y = df['location_y'].isna() & df['location_x'].notna() & (df['company_name_x'] == df['company_name_y']) & (df['country_id_x'] == df['country_id_y'])

# Create a DataFrame for partial duplicates by selecting rows that match any of the Boolean masks defined above
df_partials = df[(mask_company_x | mask_company_y) | (mask_country_x | mask_country_y) | (mask_location_x | mask_location_y)].copy()

# Add a new column to the DataFrame indicating the type of duplicates as "partial"
df_partials['type'] = str(TypeEnum.PARTIAL)

# Define a function to apply semantic/temporal types to each row
def apply_type(row):
  # Check if rows have the same company name, location, country, and retrieval date
  same_company = row['company_name_x'] == row['company_name_y']
  same_location = row['location_x'] == row['location_y']
  same_country = row['country_id_x'] == row['country_id_y']
  same_date = row['retrieval_date_x'] == row['retrieval_date_y']

  # If the score is above 0.7 and the rows have the same company name, location, and country,
  # determine the type based on whether the retrieval dates are the same or not
  if row['score'] >= 0.7 and same_company and same_country and same_location:
    return str(TypeEnum.SEMANTIC) if same_date else str(TypeEnum.TEMPORAL)

  # If the conditions above are not met, we assume it is a non-duplicate
  return str(TypeEnum.NON_DUPLICATE)

# Create a copy of the DataFrame containing semantic duplicates
other_duplicates = df_semantic_duplicates.copy()

# Fill in missing values for company names, country IDs, locations, and scores
other_duplicates['company_name_x'].fillna('', inplace=True)
other_duplicates['company_name_y'].fillna('', inplace=True)
other_duplicates['country_id_x'].fillna('', inplace=True)
other_duplicates['country_id_y'].fillna('', inplace=True)
other_duplicates['location_x'].fillna('', inplace=True)
other_duplicates['location_y'].fillna('', inplace=True)
other_duplicates['score'].fillna(0, inplace=True)

# Apply the type function to each row of the DataFrame, using swifter to enable parallel processing
other_duplicates['type'] = other_duplicates.swifter.allow_dask_on_strings(enable=True).progress_bar(enable=True, desc=None).apply(apply_type, axis=1)

# Drop rows that are classified as non-duplicates
other_duplicates.drop(other_duplicates[other_duplicates['type'] == str(TypeEnum.NON_DUPLICATE)].index, inplace=True)

# Select only the id1, id2, and type columns of the DataFrame, and set the index to be id1 and id2
other_duplicates = other_duplicates[['id1', 'id2', 'type']]
other_duplicates.set_index(['id1', 'id2'])

# Merge full duplicates and other duplicates using 'id1' and 'id2' columns as keys
duplicates_df = pd.merge(full_duplicates, other_duplicates, on=['id1', 'id2'], how='outer', suffixes=('_full', '_other'))

# Combine the values in the 'type_full' and 'type_other' columns into a single column 'type' in the merged dataframe
duplicates_df['type'] = duplicates_df['type_full'].combine_first(duplicates_df['type_other'])

# Drop the 'type_full' and 'type_other' columns from the merged dataframe
duplicates_df = duplicates_df.drop(columns=['type_full', 'type_other'])

# Merge the merged dataframe with clean data to get all the other fields
duplicates_df = pd.merge(duplicates_df, clean_data, left_on='id1', right_on='id')
duplicates_df = pd.merge(duplicates_df, clean_data, left_on='id2', right_on='id')

# Fill missing values in selected columns with empty strings
duplicates_df['company_name_x'].fillna('', inplace=True)
duplicates_df['company_name_y'].fillna('', inplace=True)
duplicates_df['country_id_x'].fillna('', inplace=True)
duplicates_df['country_id_y'].fillna('', inplace=True)
duplicates_df['location_x'].fillna('', inplace=True)
duplicates_df['location_y'].fillna('', inplace=True)

# Assign 'type' as 'TEMPORAL' to rows where the 'retrieval_date_x' and 'retrieval_date_y' columns are not equal
duplicates_df.loc[(duplicates_df['retrieval_date_x'] != duplicates_df['retrieval_date_y']), 'type'] = str(TypeEnum.TEMPORAL)

# Two job advertisements are considered partial duplicates if they describe
# the SAME job position AND one job advertisement is missing a characteristic
# of the other one.
semifull_duplicates = clean_data.copy()
semifull_duplicates['title'] = semifull_duplicates['title'].str.lower()
semifull_duplicates['description'] = semifull_duplicates['description'].str.lower()
semifull_duplicates = semifull_duplicates.merge(semifull_duplicates, on=['title', 'description'])
semifull_duplicates = semifull_duplicates[semifull_duplicates['id_x'] < semifull_duplicates['id_y']]
semifull_duplicates = semifull_duplicates[['id_x', 'id_y']]
semifull_duplicates.rename({ 'id_x': 'id1', 'id_y': 'id2' }, axis=1, inplace=True)
partial_duplicates = pd.merge(semifull_duplicates, other_duplicates, on=['id1', 'id2'], how='outer')
partial_duplicates = pd.merge(partial_duplicates, clean_data, left_on='id1', right_on='id')
partial_duplicates = pd.merge(partial_duplicates, clean_data, left_on='id2', right_on='id')
partial_duplicates.loc[
  (
    ((partial_duplicates['company_name_x'].isna()) & (partial_duplicates['company_name_y'].notna()))
    |
    ((partial_duplicates['company_name_x'].notna()) & (partial_duplicates['company_name_y'].isna()))
  )
  |
  (
    ((partial_duplicates['location_x'].isna()) & (partial_duplicates['location_y'].notna()))
    |
    ((partial_duplicates['location_x'].notna()) & (partial_duplicates['location_y'].isna()))
  )
  |
  (
    ((partial_duplicates['country_id_x'].isna()) & (partial_duplicates['country_id_y'].notna()))
    |
    ((partial_duplicates['country_id_x'].notna()) & (partial_duplicates['country_id_y'].isna()))
  )
, 'type'] = str(TypeEnum.PARTIAL)

partial_duplicates.drop(partial_duplicates[partial_duplicates['type'] != str(TypeEnum.PARTIAL)].index, inplace=True)

# Define a function to append full rows to a group of duplicate rows
# Two identical partial 1 and 2 duplicates of the same record (job posting) 3, are considered full duplicates.
def append_full_rows(group):
  if len(group) > 1:
    full_row = pd.DataFrame({'id1': [group['id1'].iloc[0]], 'id2': [group['id1'].iloc[1]], 'type': str(TypeEnum.FULL) })
    return pd.concat([group, full_row], ignore_index=True)
  else:
    return group

# Group the partial duplicates by their second id and apply the 'append_full_rows' function to each group
partial_duplicates = partial_duplicates.groupby('id2').apply(append_full_rows).reset_index(drop=True)
duplicates_df = pd.concat([duplicates_df, partial_duplicates, df_partials], ignore_index=True)

# Drop all non-duplicate rows from the 'duplicates_df' DataFrame
duplicates_df.drop(duplicates_df[duplicates_df['type'] == str(TypeEnum.NON_DUPLICATE)].index, inplace=True)

# Reorder the columns in the 'duplicates_df' DataFrame to ['id1', 'id2', 'type']
duplicates_df = duplicates_df[['id1', 'id2', 'type']]

# Re-order so that `id1` is always smaller than `id2`
duplicates_df.loc[duplicates_df['id1'] > duplicates_df['id2'], ['id1', 'id2']] = duplicates_df.loc[duplicates_df['id1'] > duplicates_df['id2'], ['id2', 'id1']].values
duplicates_df[['id1', 'id2']] = duplicates_df[['id1', 'id2']].astype(int)

# Sort numerically by id1, id2
duplicates_df.sort_values(['id1', 'id2'], inplace=True)

# Drop duplicate rows if there is any
duplicates_df.drop_duplicates(subset=['id1', 'id2'], inplace=True)

# Assertion: `id1` and `id2` columns should not contain the same value.
assert(duplicates_df[duplicates_df['id1'] == duplicates_df['id2']].empty)

# Assertion: `id1` should always be smaller than `id2`.
assert(duplicates_df[duplicates_df['id1'] > duplicates_df['id2']].empty)

# Assertion: if specific job advertisements are not duplicates, they
# should NOT BE INCLUDED.
assert(duplicates_df[duplicates_df['type'] == str(TypeEnum.NON_DUPLICATE)].empty)

# Assertion: each `id1`,`id2` pair should only appear once.
# If multiple duplicates are found between the same two documents, then
# the `type` column should contain the most specific type of duplicate.
assert(duplicates_df[duplicates_df.duplicated(['id1', 'id2'], keep=False)].empty)

# Write the output to a file `duplicates.csv`
duplicates_df.to_csv('duplicates.csv', index=False, header=False)