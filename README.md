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

# Our plan:
How are we going to implement this idea?

first define:
Education.field- the key field under the column Education in linkedin/people table
General subject - the subject according to which we would like to classify the profiles, i.e. we will classify each record if it belongs to any subject and subject according to Education.field
Examples of General subject:
Mathematics, physics, education and more..
Later for these general subjects we will collect courses on their subject.

We will classify field.Education according to each General subject (for example, mathematics, philosophy, etc.).
We will go over all the General subject and for each General subject we will create two types of labels:
Label 1 - to which all the records that are under the general subject will belong.
Label 0 - records that belong to another general topic will belong to it.
We will do this for any general topic.

How will we categorize the data into the above labels?
We will run an example for the simplicity of the explanation, for example now we would like to classify to the general topic
"medicine" records to 0 or 1
We will distinguish between two types of records:
1. Records that can be unequivocally determined to which general subject they belong
 (for example if 'medicine= 'field.Education ), it obviously belongs to the general subject of medicine.
  We will also add to these records a number of topics included in medicine (for example if 'field.Education'=doctor,nurse.. )
  .

2. Records that cannot be concluded unequivocally and therefore a learning process must be carried out in order to determine which general subject they belong to (for example if 'pharmacist= 'field.Education we understand for sure that this is part of a general subject of medicine but in order for our system to understand this a learning process must be carried out which we will detail later).
We didn't classify the pharmacist as a type 1 record because we can't think of all the subjects in the existing domain, and that's why our model has to use a learning process to understand who those subjects are

For Type 1 records:
Classified as label 1 if field. Education equivalent to the general subject being tested (for example, doctor).
Classified to label 0 if field. Education is equivalent to another general subject (e.g. philosophy).
For type records: 2
We will use a language model in order to embed (present as a vector) the field. Education and the general subject
(e.g. medicine).
We will use field.education and labels of type 1 records for the learning process in that way:
We will compute cosine similarity between fields. Education and the general subject (medicine) presented as vectors,
This result will be used as an explanatory variable. In this way we will train the model based on type 1 records
We will use the trained model to classify type 2 records after embedding and cosine
 .similarity

*Explaining the model intuitively: our goal is for the model to classify type 2 records according to type 1 records, meaning that the analogy between type 1 records and the general subject will be good enough to include type 2 records

# code:

# step 1: read the /linkedin/people data

   ```bash
    from pyspark.sql.functions import col, size
    profiles = spark.read.parquet('/linkedin/people')
    profiles.display()
   ```
# step 2: number ,explode ,clean and compute word embedding 
function
:compute_word2vec
In this function we would like to take the field education.field from the education column and do word embedding for it, a problem that has arisen is that some of this information includes more than one word, so we will split the data into different rows so that each row contains one word (later we will return the data to its original structure for this purpose the function also gave for each row an id before splitting) after each row contains one word the function performs word embedding for the words
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

