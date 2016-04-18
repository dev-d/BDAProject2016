
# Script to demostrate how to run co-clustering using spark and scala.
# Assumption: we assume that sample data is loaded in a Hive table called datase.
# Use the loader script to load this data if you have not already done so 
# To execute use, spark-shell

val df_newsgroup = sqlContext.sql("select * from dataset")

val splitData = df_newsgroup.randomSplit(Array(0.8, 0.2))
val df_test = splitData(0)
val df_train = splitData(1)

val tokenizer = new org.apache.spark.ml.feature.Tokenizer
tokenizer.setInputCol("news")
tokenizer.setOutputCol("news_words")

val df_train_words = tokenizer.transform(df_train)