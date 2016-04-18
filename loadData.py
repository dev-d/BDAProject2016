# sample script to load some data from scikit-learn and write to a Hive table
# we'll use this data later to demonstrate co-clustering

from pyspark.sql import SQLContext
from sklearn.datasets import fetch_20newsgroups

# uncomment if you will execute using spark-submit <filename.py>
# from pyspark import SparkConf, SparkContext
# conf = SparkConf().setAppName("Data loader app")
# sc = SparkContext(conf=conf)
# sqlContext = SQLContext(sc)


categories = ['rec.autos', 'rec.sport.baseball', 'comp.graphics', 'comp.sys.mac.hardware',
              'sci.space', 'sci.crypt', 'talk.politics.guns', 'talk.religion.misc']
newsgroup = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# Create pandas DataFrames for values and targets
import pandas as pd
pdf_newsgroup = pd.DataFrame(data=newsgroup.data, columns=['news']) # Texts
pdf_newsgroup_target = pd.DataFrame(data=newsgroup.target, columns=['target'])

df_newsgroup = sqlContext.createDataFrame(pd.concat([pdf_newsgroup, pdf_newsgroup_target], axis=1))

sqlContext.createDataFrame(newsgroup)
sqlContext.sql("create table dataset as select * from df_newsgroup");

# Now a table called datasets exists in Hive (in the default database)