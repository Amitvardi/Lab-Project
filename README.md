# Matching courses automatically to LinkedIn users
Amit Vardi, Lihi tal, Ron Shahar

# Overview
Many people with different fields of education use the LinkedIn platform in order to find a job, improve their knowledge and connect with people with common professional interests.

Today LinkedIn does not offer practical professional development such as courses in the relevant educational fields.

Our project will deal with adapting courses to LinkedIn users based on their education.

Our system will allow:
1. Professional development in that the user will be able to find courses that are relevant to him easily

2. Improving efficiency - instead of the user spending a lot of time searching for courses, the system will automatically recommend them

3. Improving the LinkedIn user experience - adding another feature to the system


In addition, adding the feature has a business advantage for LinkedIn in that they will be able to collaborate with sites that offer courses (for example Coursera) and charge a fee for it

# Our plan :
How are we going to implement this idea? 

first define:

Education.field- the key field under the column Education in linkedin/people table.

General subject - the subject according to which we would like to classify the profiles, i.e. we will classify each record if it belongs to any subject and subject according to Education.field.
Examples of General subject:
Mathematics, physics, education and more..

Later for these general subjects we will collect courses on their subject.

We will classify Education.field according to each General subject (for example, mathematics, philosophy, etc.).
We will go over all the General subject and for each General subject we will create two types of labels:

Label 1 - to which all the records that are under the general subject will belong.

Label 0 - records that belong to another general topic will belong to it.

We will do this for any general topic.

How will we categorize the data into the above labels?

We will run an example for the simplicity of the explanation, for example now we would like to classify to the general topic
"medicine" records to 0 or 1
We will distinguish between two types of records:

1. Records that can be unequivocally determined to which general subject they belong
 (for example if 'medicine= 'Education.field ), it obviously belongs to the general subject of medicine.
  We will also add to these records a number of topics included in medicine (for example if 'Education.field'=doctor,nurse.. )
  .
  ![Economy example](C:\Users\amit\Desktop\githb_project\Lab-Projec\python\photos\economy_example.jpeg)

2. Records that cannot be concluded unequivocally and therefore a learning process must be carried out in order to determine which general subject they belong to (for example if 'pharmacist= 'Education.field we understand for sure that this is part of a general subject of medicine but in order for our system to understand this a learning process must be carried out which we will detail later).
We didn't classify the pharmacist as a type 1 record because we can't think of all the subjects in the existing domain, and that's why our model has to use a learning process to understand who those subjects are

For Type 1 records:

Classified as label 1 if field. Education equivalent to the general subject being tested (for example, doctor).
Classified to label 0 if field. Education is equivalent to another general subject (e.g. philosophy).

For type 2 records: 

We will use a language model in order to embed (present as a vector) the field. Education and the general subject
(e.g. medicine).
We will use Education.field and labels of type 1 records for the learning process in that way:
We will compute cosine similarity between fields. Education and the general subject (medicine) presented as vectors,
This result will be used as an explanatory variable. In this way we will train the model based on type 1 records
We will use the trained model to classify type 2 records after embedding and cosine
 .similarity

*Explaining the model intuitively: our goal is for the model to classify type 2 records according to type 1 records, meaning that the analogy between type 1 records and the general subject will be good enough to include type 2 records

# code:

# step 1 - read the /linkedin/people data

   ```bash
    from pyspark.sql.functions import col, size
    profiles = spark.read.parquet('/linkedin/people')
    profiles.display()
   ```
# step 2 - clean, explode and compute word embedding 

We remove all the lines that contain characters that are not letters because afterwards we need to perform word embedding and this removal will prevent the model from trying to perform embedding on words it does not know.
In addition we remove stop words as their word embedding values ​​may bias the result.

In addition we take the field education.field from the education column and do word embedding for it, a problem that has arisen is that some of this information includes more than one word, so we will split the data into different rows so that each row contains one word (later we will return the data to its original structure for this purpose the function also gave for each row an id before splitting) after each row contains one word the function performs word embedding for the words.
   ```bash

from pyspark.sql.functions import split, explode, monotonically_increasing_id, row_number
from pyspark.sql.window import Window
from pyspark.sql.functions import regexp_replace
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *


def compute_word2vec(df):
    windowSpec = Window.orderBy("id")
    df = df.withColumn("id", monotonically_increasing_id())
    exploded_df = df.withColumn("field_exploded", explode(split(df["field"], " ")))
    exploded_df = exploded_df.withColumn('field_exploded', split(col('field_exploded'), '/')[0])
    exploded_df = exploded_df.withColumn("field_exploded", regexp_replace(col("field_exploded"), "[^a-zA-Z]", ""))
    exploded_df = exploded_df.filter(exploded_df["field_exploded"] != "")


    document_assembler = DocumentAssembler().setInputCol("field_exploded").setOutputCol("document")
    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")

    word_embeddings = WordEmbeddingsModel.pretrained("glove_100d") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("word_vector_1")  # You can use different word embeddings models here

    # Create the pipeline
    pipeline = Pipeline(stages=[document_assembler, tokenizer, word_embeddings])

    # Fit the pipeline to the data
    model = pipeline.fit(exploded_df)

    # Transform the data
    exploded_df = model.transform(exploded_df)
    columns_to_drop = ["document", "token"]
    exploded_df = exploded_df.drop(*columns_to_drop)
    exploded_df = exploded_df.select("*", col("word_vector_1.embeddings").alias("word_vector"))
    exploded_df=exploded_df.drop("word_vector_1")
    return exploded_df

def remove_stop_words(df):
    stop_words=["and","in","a","on","of"]
    filtered_df = df[~df['field_exploded'].isin(stop_words)]
    return filtered_df

from pyspark.sql.functions import col, array_contains, lower, expr
from pyspark.sql.functions import explode

filtered_profiles = profiles.filter((col("education").isNotNull()) & (size(col("education")) > 0))
filtered_profiles = filtered_profiles.select("education.field")
filtered_profiles = filtered_profiles.select(explode("field").alias("field"))
filtered_profiles = filtered_profiles.na.drop(subset=["field"])
filtered_profiles = filtered_profiles.withColumn("field", lower(col("field")))
filtered_profiles=compute_word2vec(filtered_profiles)
filtered_profiles=remove_stop_words(filtered_profiles)


   ```
# step 3 - Preprocessing type 1 records:
type 1 records definition:
 Records that can be unequivocally determined to which general subject they belong
 (for example if 'medicine= 'Education.field ), it obviously belongs to the general subject of medicine.
  We will also add to these records a number of topics included in medicine (for example if 'Education.field'=doctor,nurse.. )

 At this step we have filtered the records that are suitable for each general subject according to the definition of type 1 records
   ```bash
math_profiles = filtered_profiles.filter(col("field").like("%math%"))
print(math_profiles.count())


Physics_profiles = filtered_profiles.filter(col("field").like("%physic%"))
print(Physics_profiles.count())

medicine_profiles = filtered_profiles.filter(
    (col("field").like("%doctor%")) |
    (col("field").like("%clinic%")) |
    (col("field").like("%vet%")) |
    (col("field").like("%medical%")) |
    (col("field").like("%medicine%")) |
    (col("field").like("%health%")) |
    (col("field").like("%nurse%")) |
    (col("field").like("%bio%")) |
    (col("field").like("%animal%")) |
    (col("field").like("%care%")) |
    (col("field").like("%pharmacy%"))
)
print(medicine_profiles.count())

marketing_economics_profiles = filtered_profiles.filter(
    (col("field").like("%commerce%")) |
    (col("field").like("%sale%")) |
    (col("field").like("%finance%")) |
    (col("field").like("%business%")) |
    (col("field").like("%investment%")) |
    (col("field").like("%accounting%")) |
    (col("field").like("%economic%"))
)
print(marketing_economics_profiles.count())

social_sciences_profiles = filtered_profiles.filter(
    (col("field").like("%psychology%")) |
    (col("field").like("%theology%")) |
    (col("field").like("%sociology%")) |
    (col("field").like("%history%")) |
    (col("field").like("%philosophy%")) |
    (col("field").like("%therapy%")) |
    (col("field").like("%social%")) |
    (col("field").like("%communication%")) |
    (col("field").like("%heritage%"))
)
print(social_sciences_profiles.count())

teaching_profiles = filtered_profiles.filter(
    (col("field").like("%teach%")) |
    (col("field").like("%education%")) |
    (col("field").like("%school%")) |
    (col("field").like("%university%"))
)
print(teaching_profiles.count())


food_profiles = filtered_profiles.filter(
    (col("field").like("%food%")) |
    (col("field").like("%chef%")) |
    (col("field").like("%restaurant%")) |
    (col("field").like("%nutrition%"))
)

print(food_profiles.count())

computer_profiles = filtered_profiles.filter(
    (col("field").like("%code%")) |
    (col("field").like("%computer%")) |
    (col("field").like("%data%")) |
    (col("field").like("%cyber%")) |
    (col("field").like("%network%"))
)
print(computer_profiles.count())

```
# step 4 - creating type 1 records (the training data) and compute cosine similarity for each general subject:

For each record, we calculated the cosine similarity between the education.field column and the general topic. At the end, we calculated an average of the cosine similarity values ​​for all the words belonging to that education.field on which we performed a split in step 2

Then we prepared the records of type 1 by mixing the different tables and putting labels of 0 and 1 in the required places
```bash

import gensim.downloader as api
word2vec_model = api.load("glove-wiki-gigaword-100")


from scipy.spatial.distance import cosine as scipy_cosine
from pyspark.sql import functions as F

def calculate_cosine_similarity(word_vector_1, word_vector_2):
    try:
        return float(1 - scipy_cosine(word_vector_1, word_vector_2))
    except:
        return 0


def compute_avg_cosine(df):
    cosine_column= [col for col in df.columns if "cosine" in col][0]
    df = df[(df[cosine_column] > 0) & (df[cosine_column] < 1)]
    df = df.groupBy("id").agg(F.mean(cosine_column).alias(cosine_column), *[F.first(col).alias(col) for col in df.columns if col != "id" and col != cosine_column])
    df.display()
    return df


import random
from pyspark.sql.functions import lit
from pyspark.sql.types import DoubleType, ArrayType
import numpy as np
from pyspark.sql import functions as F
def create_labels_compute_cosine(central_df,df2,df3,df4,df5,df6,df7,df8,field):
    central_length=central_df.count()
    sample_length= int(central_length / 7)
    sample_df = df2.limit(sample_length)
    sample_df = sample_df.union(df3.limit(sample_length))
    sample_df = sample_df.union(df4.limit(sample_length))
    sample_df = sample_df.union(df5.limit(sample_length))
    sample_df = sample_df.union(df6.limit(sample_length))
    sample_df = sample_df.union(df7.limit(sample_length))
    sample_df = sample_df.union(df8.limit(sample_length))
    sample_df = sample_df.withColumn("labels", lit(0))
    total_df=central_df.withColumn("labels", lit(1))
    total_df=total_df.union(sample_df)

    field_vector=word2vec_model[field]
    cosine_similarity_udf = udf(calculate_cosine_similarity, DoubleType())
    word_vector_literal = np.array(field_vector).tolist()
    word_vector_col = F.array([F.lit(x) for x in word_vector_literal])
    total_df = total_df.withColumn("cosine_sim_" + field, cosine_similarity_udf(col("word_vector"), word_vector_col))
    
    total_df=compute_avg_cosine(total_df)
    return total_df


math=create_labels_compute_cosine(math_profiles,Physics_profiles,medicine_profiles,marketing_economics_profiles,social_sciences_profiles,teaching_profiles,food_profiles,computer_profiles,"math")

Physics=create_labels_compute_cosine(Physics_profiles,math_profiles,medicine_profiles,marketing_economics_profiles,social_sciences_profiles,teaching_profiles,food_profiles,computer_profiles,"physics")

medicine=create_labels_compute_cosine(medicine_profiles,Physics_profiles,math_profiles,marketing_economics_profiles,social_sciences_profiles,teaching_profiles,food_profiles,computer_profiles,"medicine")

marketing_economics=create_labels_compute_cosine(marketing_economics_profiles,Physics_profiles,medicine_profiles,math_profiles,social_sciences_profiles,teaching_profiles,food_profiles,computer_profiles,"economy")

social_sciences=create_labels_compute_cosine(social_sciences_profiles,Physics_profiles,medicine_profiles,marketing_economics_profiles,math_profiles,teaching_profiles,food_profiles,computer_profiles,"humanities")

teaching=create_labels_compute_cosine(teaching_profiles,Physics_profiles,medicine_profiles,marketing_economics_profiles,social_sciences_profiles,math_profiles,food_profiles,computer_profiles,"teaching")

food=create_labels_compute_cosine(food_profiles,Physics_profiles,medicine_profiles,marketing_economics_profiles,social_sciences_profiles,teaching_profiles,math_profiles,computer_profiles,"food")

computer=create_labels_compute_cosine(computer_profiles,Physics_profiles,medicine_profiles,marketing_economics_profiles,social_sciences_profiles,teaching_profiles,food_profiles,math_profiles,"computer")

```
# step 5 - creating test data:

We do the same process we did for the training data in step 4 but without creating labels.
For each record, we calculated the cosine similarity between the education.field column and the general topic. At the end, we calculated an average of the cosine similarity values ​​for all the words belonging to that education.field .

In the end we filter the data frame to relevant columns.

```bash

def compute_cosine(df,field):
    field_vector=word2vec_model[field]
    cosine_similarity_udf = udf(calculate_cosine_similarity, DoubleType())
    word_vector_literal = np.array(field_vector).tolist()
    word_vector_col = F.array([F.lit(x) for x in word_vector_literal])
    total_df = df.withColumn("cosine_sim_" + field, cosine_similarity_udf(col("word_vector"), word_vector_col))
    return total_df

def compute_avg_cosine_for_all_columns(df):
    cosine_columns= [col for col in df.columns if "cosine" in col]
    for c in cosine_columns:
        df = df[(df[c] > 0) & (df[c] < 1)]
    
    agg_exprs = {col: F.mean(col).alias(col) for col in cosine_columns}
    all_columns = df.columns
    df = df.groupBy("id").agg(*agg_exprs.values(), *[F.first(col).alias(col) for col in all_columns if col not in cosine_columns and col!="id"])
    return df
    
def create_test_data(df):
    fields=["math","physics","medicine","economy","humanities","teaching","food","computer"]
    for f in fields:
        df=compute_cosine(df,f)
    total_result=compute_avg_cosine_for_all_columns(df)
    return total_result

total_df_test=create_test_data(filtered_profiles)
cosine_columns= [col for col in total_df_test.columns if "cosine" in col]
selected_columns = cosine_columns + ["id"]+["field"]
total_df_test = total_df_test.select(selected_columns)
print(total_df_test.count)  


```
# step 6 - Train and Predict:

We perform a training and a test for each general topic (a total of 8 times). We do this separation because we want to allow certain records to belong to more than one number of labels independently.
 (For example, allow a person who is involved in mathematics and computer science to be part of the general subjects of mathematics, computers and even education, since through the knowledge he has acquired, perhaps he can be interested in teaching).
We used RandomForestRegressor for training and prediction because in homework 2 question 4 it worked well on a similar task where we used cosine similraity explanatory variables


```bash
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import round
from functools import reduce

def stay_relevant_columns(df):
    columns_to_stay = [col for col in df.columns if 'cosine' in col or 'id' in col or 'labels' in col]
    total_result=df[columns_to_stay]
    return total_result


def train_and_test(train_df,test_df,cosine_column):
    
    assembler = VectorAssembler(inputCols=[cosine_column], outputCol="features")
    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)

    rf = RandomForestRegressor(featuresCol="features", labelCol="labels", numTrees=100)
    rf_model = rf.fit(train_df)

    predictions = rf_model.transform(test_df)

    substring = cosine_column[len("cosine_sim_"):]
    substring = cosine_column.split("cosine_sim_")[1]
    
    predictions = predictions.withColumn(substring, round(col("prediction")))
    columns_to_drop = ["prediction", "features"]
    predictions = predictions.drop(*columns_to_drop)
    return predictions

total_df_test=total_df_test.dropna()
columns_to_stay = [col for col in total_df_test.columns if 'cosine' in col or 'id' in col or "field" in col]
total_df_test = total_df_test[columns_to_stay]

math_rel=stay_relevant_columns(math)
Physics_rel=stay_relevant_columns(Physics)
medicine_rel=stay_relevant_columns(medicine)
marketing_economics_rel=stay_relevant_columns(marketing_economics)
social_sciences_rel=stay_relevant_columns(social_sciences)
teaching_rel=stay_relevant_columns(teaching)
food_rel=stay_relevant_columns(food)
computer_rel=stay_relevant_columns(computer)
math_pred=train_and_test(math_rel,total_df_test["id","cosine_sim_math","field"],"cosine_sim_math")

physics_pred=train_and_test(Physics_rel,total_df_test["id","cosine_sim_physics"],"cosine_sim_physics")

medicine_pred=train_and_test(medicine_rel,total_df_test["id","cosine_sim_medicine"],"cosine_sim_medicine")

marketing_economics_pred=train_and_test(marketing_economics_rel,total_df_test["id","cosine_sim_economy"],"cosine_sim_economy")

social_sciences_pred=train_and_test(social_sciences_rel,total_df_test["id","cosine_sim_humanities"],"cosine_sim_humanities")

teaching_pred=train_and_test(teaching_rel,total_df_test["id","cosine_sim_teaching"],"cosine_sim_teaching")

food_pred=train_and_test(food_rel,total_df_test["id","cosine_sim_food"],"cosine_sim_food")

computers_pred=train_and_test(computer_rel,total_df_test["id","cosine_sim_computer"],"cosine_sim_computer")


```
# step 7 - Results:
We will present an example from the table that matches general subjects to a person according to his educaiton.field:


In the above table, where there is a 1, it means that we want to show that person courses in the relevant columns, and a 0 means that we do not want to show him the courses on these subjects.

The code that appears groups all the data into one table and helps us present the results in a readable and understandable way
```bash

from functools import reduce
from pyspark.sql import DataFrame
dfs = [math_pred, physics_pred, medicine_pred, marketing_economics_pred, social_sciences_pred,teaching_pred,food_pred,computers_pred]

# Define the join condition
join_condition = "id"

# Perform a series of left joins to combine the DataFrames
combined_df = reduce(lambda df1, df2: df1.join(df2, on=join_condition, how='left'), dfs)

# Show the combined DataFrame
combined_df=combined_df.select("id", "field","math","physics","medicine","economy","teaching","food","computer")
combined_df.display()



```
# step 8 - Evaluate our results:

We calculated F1 on the training and validation data (80-20 split) the results are below:



Code for train data evaluation:
```bash



from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import round
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def compute_f1(df,title):
  evaluator = MulticlassClassificationEvaluator(labelCol='labels', predictionCol=title, metricName='f1')

  # Step 2: Compute F1 score
  f1_score = evaluator.evaluate(df)
  print("F1 Score:", f1_score)

def stay_relevant_columns(df):
    columns_to_stay = [col for col in df.columns if 'cosine' in col or 'id' in col or 'labels' in col]
    total_result=df[columns_to_stay]
    return total_result


def train_and_test(train_df,test_df,cosine_column):
    
    assembler = VectorAssembler(inputCols=[cosine_column], outputCol="features")
    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)

    rf = RandomForestRegressor(featuresCol="features", labelCol="labels", numTrees=100)
    rf_model = rf.fit(train_df)

    predictions = rf_model.transform(test_df)

    substring = cosine_column[len("cosine_sim_"):]
    substring = cosine_column.split("cosine_sim_")[1]

    predictions = predictions.withColumn(substring, round(col("prediction")))
    predictions = predictions.drop("features")

    print("f1 score on "+substring)
    compute_f1(predictions,substring)

total_df_test=total_df_test.dropna()
columns_to_stay = [col for col in total_df_test.columns if 'cosine' in col or 'id' in col]
total_df_test = total_df_test[columns_to_stay]

math_rel=stay_relevant_columns(math)
Physics_rel=stay_relevant_columns(Physics)
medicine_rel=stay_relevant_columns(medicine)
marketing_economics_rel=stay_relevant_columns(marketing_economics)
social_sciences_rel=stay_relevant_columns(social_sciences)
teaching_rel=stay_relevant_columns(teaching)
food_rel=stay_relevant_columns(food)
computer_rel=stay_relevant_columns(computer)

math_pred=train_and_test(math_rel,math_rel,"cosine_sim_math")

physics_pred=train_and_test(Physics_rel,Physics_rel,"cosine_sim_physics")

medicine_pred=train_and_test(medicine_rel,medicine_rel,"cosine_sim_medicine")

marketing_economics_pred=train_and_test(marketing_economics_rel,marketing_economics_rel,"cosine_sim_economy")

social_sciences_pred=train_and_test(social_sciences_rel,social_sciences_rel,"cosine_sim_humanities")

teaching_pred=train_and_test(teaching_rel,teaching_rel,"cosine_sim_teaching")

food_pred=train_and_test(food_rel,food_rel,"cosine_sim_food")

computers_pred=train_and_test(computer_rel,computer_rel,"cosine_sim_computer")



```
Code for validation data evaluation:
```bash

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import round
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def compute_f1(df,title):
  evaluator = MulticlassClassificationEvaluator(labelCol='labels', predictionCol=title, metricName='f1')

  # Step 2: Compute F1 score
  f1_score = evaluator.evaluate(df)
  df.show()
  print("F1 Score:", f1_score)

def stay_relevant_columns(df):
    columns_to_stay = [col for col in df.columns if 'cosine' in col or 'id' in col or 'labels' in col]
    total_result=df[columns_to_stay]
    return total_result


def train_and_test(train_df,test_df,cosine_column):
    
    assembler = VectorAssembler(inputCols=[cosine_column], outputCol="features")
    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)

    rf = RandomForestRegressor(featuresCol="features", labelCol="labels", numTrees=100)
    rf_model = rf.fit(train_df)

    predictions = rf_model.transform(test_df)

    substring = cosine_column[len("cosine_sim_"):]
    substring = cosine_column.split("cosine_sim_")[1]

    predictions = predictions.withColumn(substring, round(col("prediction")))
    predictions = predictions.drop("features")

    print("f1 score on "+substring)
    compute_f1(predictions,substring)

total_df_test=total_df_test.dropna()
columns_to_stay = [col for col in total_df_test.columns if 'cosine' in col or 'id' in col]
total_df_test = total_df_test[columns_to_stay]

math_rel=stay_relevant_columns(math)
Physics_rel=stay_relevant_columns(Physics)
medicine_rel=stay_relevant_columns(medicine)
marketing_economics_rel=stay_relevant_columns(marketing_economics)
social_sciences_rel=stay_relevant_columns(social_sciences)
teaching_rel=stay_relevant_columns(teaching)
food_rel=stay_relevant_columns(food)
computer_rel=stay_relevant_columns(computer)

train_math, test_math = math_rel.randomSplit([0.8, 0.2], seed=55) 
math_pred=train_and_test(train_math,test_math,"cosine_sim_math")

train_physics, test_physics = Physics_rel.randomSplit([0.8, 0.2], seed=55) 
physics_pred=train_and_test(train_physics,test_physics,"cosine_sim_physics")

train_medicine, test_medicine = medicine_rel.randomSplit([0.8, 0.2], seed=55) 
medicine_pred=train_and_test(train_medicine,test_medicine,"cosine_sim_medicine")

train_marketing_economics, test_marketing_economics =  marketing_economics_rel.randomSplit([0.8, 0.2], seed=55) 
marketing_economics_pred=train_and_test(train_marketing_economics,test_marketing_economics,"cosine_sim_economy")

train_social_sciences ,test_social_sciences =  social_sciences_rel.randomSplit([0.8, 0.2], seed=55) 
social_sciences_pred=train_and_test(train_social_sciences,test_social_sciences,"cosine_sim_humanities")

train_teaching, test_teaching =  teaching_rel.randomSplit([0.8, 0.2], seed=55) 
teaching_pred=train_and_test(train_teaching,test_teaching,"cosine_sim_teaching")

train_food, test_food =  food_rel.randomSplit([0.8, 0.2], seed=55) 
food_pred=train_and_test(train_food,test_food,"cosine_sim_food")

train_computer, test_computer =  computer_rel.randomSplit([0.8, 0.2], seed=55) 
computers_pred=train_and_test(train_computer,test_computer,"cosine_sim_computer")



```
# step 9 - scrapping Coursera website:

We scraped the Coursera website according to the general subjects we tested. First we checked a large amount of courses on which we performed scraping and came to the conclusion that it would be better to show the user only selected and recommended courses, therefore we classified the site according to the most recommended courses and for each field we took the three most recommended courses Below are the results and the code that performs the scraping:

```bash

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd


def insert_more_pages(org_path,num_of_pages):
    total_pathes=[]
    for i in range(num_of_pages):
        if i==0:
            total_pathes.append(org_path)
        else:
            total_pathes.append(org_path+"&page="+str(i)+"&sortBy=BEST_MATCH")
    return total_pathes

def itraetion(course_param,param):
    total_list_of_param=[]
    for element in course_param:
        if param=="titles":
            total_list_of_param.append(element.text)
        elif param=="ratings":
            total_list_of_param.append(element.text)
        else:
            total_list_of_param.append(element.text)
    return total_list_of_param

def itreate_and_get_data(pages_pathes,driver):
    titles,skills,ratings,level=[],[],[],[]
    for path in pages_pathes:
        driver.get(path)
        course_title = driver.find_elements(By.CLASS_NAME, "cds-CommonCard-title")
        course_skills= driver.find_elements(By.CLASS_NAME,"cds-Typography-base")
        course_ratings= driver.find_elements(By.CLASS_NAME,"cds-CommonCard-ratings")
        titles.extend(itraetion(course_title,"titles"))
        skills.extend(itraetion(course_skills,"skills"))
        ratings.extend(itraetion(course_ratings,"ratings"))
    return titles,skills,ratings

def delete_above3(courses_dict):
    for key, lists in courses_dict.items():
        new_values = []
        for old_list in lists:
            truncated_list = old_list[:3]  
            new_values.append(truncated_list)
        courses_dict[key] = new_values 
    return courses_dict


def save_as_xlsx(courses_dict):
    top3_courses=delete_above3(courses_dict)
    file_names=["math","Physics","medicine","economy","humanities","teaching","food","computer"]
    for index,(p,value) in enumerate(top3_courses.items()):
        if p=="https://www.coursera.org/search?query=humanities" or p=="https://www.coursera.org/search?query=food":
            empty_list = ["empty"] * len(value[0])
            df = pd.DataFrame({
                "Course Titles": value[0],
                "Skills you'll gain": empty_list,
                "Course Rating": value[2]
            })
        else:
            df = pd.DataFrame({
            "Course Titles": value[0],
            "Skills you'll gain": value[1],
            "Course Rating": value[2]
            })
        index = p.find('=')
        name = p[index + 1:]
        df.to_excel("C:/Users/amit/PycharmProjects/gradio/Lib/site-packages/scrapping_data/"+name+".xlsx", index=False)


if __name__ == '__main__':
    path = "C:/Users/amit/PycharmProjects/gradio/Lib/site-packages/chromedriver.exe"
    num_of_pages=3
    chrome_options = Options()
    service = Service(path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    pathes=["https://www.coursera.org/search?query=math","https://www.coursera.org/search?query=physics",
            "https://www.coursera.org/search?query=medicine","https://www.coursera.org/search?query=economy",
            "https://www.coursera.org/search?query=humanities","https://www.coursera.org/search?query=teaching",
            "https://www.coursera.org/search?query=food","https://www.coursera.org/search?query=computers"
            ]


    dic_field_to_data={}
    for p in pathes:
        print(p)
        pages_pathes=insert_more_pages(p,num_of_pages)
        dic_field_to_data[p]= itreate_and_get_data(pages_pathes,driver)

    save_as_xlsx(dic_field_to_data)







    
    





    
    









    
    


  




    





```