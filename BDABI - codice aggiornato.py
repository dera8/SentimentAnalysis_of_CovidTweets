##############################################1-CONNECTING MONGODB ATLAS WITH Spark MONGODB###################################################################
import pyspark
from pyspark.sql import SparkSession

my_spark = SparkSession \
    .builder \
    .appName("myApp") \
    .config("spark.mongodb.input.uri", "spark.mongodb.input.uri mongodb+srv://Debora:ciao@cluster0.dwkae.gcp.mongodb.net/covid19geo?retryWrites=true&w=majority") \
    .config("spark.mongodb.output.uri", "mongodb+srv://Debora:ciao@cluster0.dwkae.gcp.mongodb.net/covid19geo?retryWrites=true&w=majority") \
    .getOrCreate()

  #.config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.11:2.3.1')\
months = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("uri","mongodb+srv://Debora:ciao@cluster0.dwkae.gcp.mongodb.net/covid19geo.mesi?retryWrites=true&w=majority").load()

months.show(60)

############################################################2-PREPROCESSING###################################################################################

import re
import pyspark
from pyspark.sql.functions import regexp_replace
import pyspark.sql.functions as f
import emoji
from emoji.unicode_codes import UNICODE_EMOJI

df_clean = months.select('created_at','full_text',"retweet_count")

#NEW TEXT COLUMN FILTERED 
df_cleantext = df_clean.withColumn("text", regexp_replace('full_text', 'RT @[\w]*:',' ')) #elimino gli RT e i relativi nickname
df_cleantext2 = df_cleantext.withColumn("text", regexp_replace('text', '@[\w]*',' ')) #elimino tutti i nickname
df_cleantext3 = df_cleantext2.withColumn("text", regexp_replace('text', 'https?://[A-Za-z0-9./]*',' ')) #elimino i link
df_cleantext4 = df_cleantext3.withColumn("text", regexp_replace('text', '\n',' '))
df_cleantext5 = df_cleantext4.withColumn("text", regexp_replace('text', '[^ a-zA-Zà-ú''\❤️\❣️'
                            '\😭\👎\🔝\✌\✊]',' '))

df_cleantext6 = df_cleantext5.withColumn("text", regexp_replace('text', '❤️', '||positive||'))
df_cleantext7 = df_cleantext6.withColumn("text", regexp_replace('text', '❣️', '||positive||'))
df_cleantext8 = df_cleantext7.withColumn("text", regexp_replace('text', '✌', '||positive||'))
df_cleantext9 = df_cleantext8.withColumn("text", regexp_replace('text', '✊', '||positive||'))
df_cleantext10 = df_cleantext9.withColumn("text", regexp_replace('text', '🔝', '||positive||'))
df_cleantext11 = df_cleantext10.withColumn("text", regexp_replace('text', '😭', '||negative||'))
df_cleantext12 = df_cleantext11.withColumn("text", regexp_replace('text', '👎', '||negative||'))

#CANCELLO I DUPLICATI

df_cleantext13 = df_cleantext12.dropDuplicates()
display(df_cleantext13)              

##############################################################3-VADER############################################################################################
#ETICHETTO I TWEETS CON VADER

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sentiment = udf(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
spark.udf.register("sentiment", sentiment)
df_cleantext14 = df_cleantext13.withColumn("sentiment",sentiment("text").cast("double"))

#df_cleantext13['polarity'] = df_cleantext13['text'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])

display(df_cleantext14)

# Definisco una funzione che calcola la sentiment in base ai valori degli scores
def evaluateSentiment(sentiment):
      if sentiment >= 0.05:
        return "positive"
      elif sentiment <= -0.05: 
        return "negative"
      elif sentiment > -0.05 and sentiment< 0.05:
        return "neutral"

label = udf(lambda x: evaluateSentiment(x))
spark.udf.register("label", label)
df_cleantext15 = df_cleantext14.withColumn("label",label("sentiment"))
#df_cleantext15['result'] = df_cleantext14['sentiment'].apply(lambda x: evaluateSentiment(x))
display(df_cleantext15)
df_cleantext16 = df_cleantext15.where(df_cleantext15.label != "neutral")

###########################################################3-SPARK NLP SENTIMENT ANALYSIS####################################################################

import sparknlp
from pyspark.ml import Pipeline, PipelineModel

from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

from sparknlp.pretrained import PretrainedPipeline

from pyspark.sql.functions import explode

from pyspark.ml.evaluation import MulticlassClassificationEvaluator# convert text column to spark nlp document

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

eng_stopwords = stopwords.words('english')

spark = sparknlp.start()

documentAssembler = DocumentAssembler() \
     .setInputCol("text") \
     .setOutputCol('document')

tokenizer = Tokenizer() \
     .setInputCols(['document']) \
     .setOutputCol('tokenized')

normalizer = Normalizer() \
     .setInputCols(['tokenized']) \
     .setOutputCol('normalized') \
     .setLowercase(True)

stopwords_cleaner = StopWordsCleaner() \
     .setInputCols(['normalized']) \
     .setOutputCol('cleanTokens') \
     .setStopWords(eng_stopwords)

word_embeddings = WordEmbeddingsModel().pretrained()\
      .setInputCols("document", "cleanTokens") \
      .setOutputCol("embeddings")

use = UniversalSentenceEncoder.pretrained() \
            .setInputCols("document","embeddings") \
            .setOutputCol("use_embeddings")

sentimentdl = SentimentDLApproach()\
      .setInputCols("use_embeddings")\
      .setOutputCol("result")\
      .setMaxEpochs(5)\
      .setValidationSplit(0.2)\
      .setLabelColumn("label")\
      .setEnableOutputLogs(True)


#DEFINISCO UNA PIPELINE NECESSARIA ALL'ESECUZIONE DELL'ALGORITMO DI SENTIMENT ANALYSIS:
pipeline = Pipeline() \
     .setStages([documentAssembler,                  
                 tokenizer,
                 normalizer,                                   
                 stopwords_cleaner,
                 word_embeddings,
                 use,
                 sentimentdl])

pipelineModel = pipeline.fit(df_cleantext16)

#salvo il modello
pipelineModel.stages[-1].write().overwrite().save('dbfs:/tmp_sentimentdl_model')

#CREO UNA NUOVA PIPELINE CON IL MODELLO SALVATO
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer
from sparknlp.annotator import Normalizer
from sparknlp.annotator import StopWordsCleaner
from sparknlp.annotator import *
from pyspark.ml import Pipeline
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

eng_stopwords = stopwords.words('english')

documentAssembler = DocumentAssembler() \
     .setInputCol("text") \
     .setOutputCol('document')

tokenizer = Tokenizer() \
     .setInputCols(['document']) \
     .setOutputCol('tokenized')

normalizer = Normalizer() \
     .setInputCols(['tokenized']) \
     .setOutputCol('normalized') \
     .setLowercase(True)

stopwords_cleaner = StopWordsCleaner() \
     .setInputCols(['normalized']) \
     .setOutputCol('cleanTokens') \
     .setStopWords(eng_stopwords)

word_embeddings = WordEmbeddingsModel().pretrained()\
      .setInputCols("document", "cleanTokens") \
      .setOutputCol("embeddings")

use = UniversalSentenceEncoder.pretrained() \
            .setInputCols("document","embeddings") \
            .setOutputCol("use_embeddings")

sentimentdl = SentimentDLModel.load("dbfs:/tmp_sentimentdl_model") \
  .setInputCols(["use_embeddings"])\
  .setOutputCol("class")

pipeline = Pipeline() \
     .setStages([documentAssembler,                  
                 tokenizer,
                 normalizer,                                   
                 stopwords_cleaner,
                 word_embeddings,
                 use,
                 sentimentdl])

#CLASSIFICAZIONE DEI TWEETS (ASSEGNO UN'ETICHETTA DI POSITIVITà, NEGATIVITà O NEUTRALITà):
#MOSTRO LA TABELLA A CLASSIFICAZIONE AVVENUTA:

df_final = pipeline.fit(df_cleantext13).transform(df_cleantext13).select("text","sentiment.result") 
#display(df_final)
#pipeline.fit(df_cleantext6).transform(df_cleantext6).printSchema()

#select only "SentimentText" and "Sentiment" column, 
#and cast "Sentiment" column data into integer
df_final2 = df_final.select("text",explode(df_final.result).alias("result"))

#####################################################################4 EVALUATION##########################################################################

#carico un dataset di test
filePath = "dbfs:/FileStore/tables/21_04_20.csv"
file_type = "csv"

# CSV options
first_row_is_header = "true"
multiline = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
dftest = spark.read.format(file_type) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .option("multiline", multiline) \
  .option("escape", "\"") \
  .load(filePath)

dftest2 = dftest.select('created_at','full_text')
#display(df)
dftest2.show(60)

#2-PREPROCESSING

import re
import pyspark
from pyspark.sql.functions import regexp_replace
import pyspark.sql.functions as f


df_cleantest = dftest2.select('full_text')

#CREO UNA NUOVA COLONNA TEXT IN CUI METTO IL FULL_TEXT RIPULITO:
df_cleantexttest = df_cleantest.withColumn("text", regexp_replace('full_text', 'RT @[\w]*:','')) #elimino gli RT e i relativi nickname
df_cleantext2test = df_cleantexttest.withColumn("text", regexp_replace('text', '@[\w]*','')) #elimino tutti i nickname
df_cleantext3test = df_cleantext2test.withColumn("text", regexp_replace('text', 'https?://[A-Za-z0-9./]*','')) #elimino i link
df_cleantext4test = df_cleantext3test.withColumn("text", regexp_replace('text', '\n',''))
df_cleantext5test = df_cleantext4test.withColumn("text", regexp_replace('text', '[^ a-zA-Zà-ú''\❤️\❣️'
                            '\😭\👎\🔝\✌\✊]',''))

df_cleantext6test = df_cleantext5test.withColumn("text",f.lower(f.col("text")))

df_cleantext7test = df_cleantext6test.withColumn("text", regexp_replace('text', '❤️', '||positive||'))
df_cleantext8test = df_cleantext7test.withColumn("text", regexp_replace('text', '❣️', '||positive||'))
df_cleantext9test = df_cleantext8test.withColumn("text", regexp_replace('text', '✌', '||positive||'))
df_cleantext10test = df_cleantext9test.withColumn("text", regexp_replace('text', '✊', '||positive||'))
df_cleantext11test = df_cleantext10test.withColumn("text", regexp_replace('text', '🔝', '||positive||'))
df_cleantext12test = df_cleantext11test.withColumn("text", regexp_replace('text', '😭', '||negative||'))
df_cleantext13test = df_cleantext12test.withColumn("text", regexp_replace('text', '👎', '||negative||'))

display(df_cleantext13test)

df_cleantext14test = df_cleantext13test.dropDuplicates()
display(df_cleantext14test)
#df_cleantext8 = df_cleantext7.dropDuplicates(['text'])

#VADER

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sentiment = udf(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
spark.udf.register("sentiment", sentiment)
df_cleantext15test = df_cleantext14test.withColumn("sentiment",sentiment("text").cast("double"))

#df_cleantext13['polarity'] = df_cleantext13['text'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])

display(df_cleantext15test)

# Definisco una funzione che calcola la sentiment in base ai valori degli scores
def evaluateSentiment(sentiment):
      if sentiment >= 0.05:
        return "positive"
      elif sentiment <= -0.05: 
        return "negative"
      elif sentiment > -0.05 and sentiment< 0.05:
        return "neutral"

label = udf(lambda x: evaluateSentiment(x))
spark.udf.register("label", label)
df_cleantext16test = df_cleantext15test.withColumn("label",label("sentiment"))
#df_cleantext15['result'] = df_cleantext14['sentiment'].apply(lambda x: evaluateSentiment(x))

display(df_cleantext16test)

df_cleantext17test = df_cleantext16test.where(df_cleantext16test.label != "neutral")

preds = pipelineModel.transform(df_cleantext17test)

predsfinal = preds.select('label','text',"result.result")
predsfinal.show(50, truncate=50)

from pyspark.sql.functions import explode
preds2 = predsfinal.select("label","text",explode(predsfinal.result).alias("predicted"))
preds3 = preds2.where(preds2.predicted != "neutral")
preds4 = preds3.toPandas()

display(preds4)

#CALCOLO ACCURACY
from sklearn.metrics import accuracy_score
print ("Accuracy:", accuracy_score(preds4['predicted'], preds4['label']))


#############################################################5 CONTEGGIO TWEETS PER MESE#####################################################################

from pyspark.sql import functions as f
import re
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import col, expr, when

df_final = pipeline.fit(df_duplicates).transform(df_duplicates).select("created_at","sentiment.result")

split_columns = f.split(df_final["created_at"], " ")
df_final = df_final.withColumn("day", split_columns.getItem(2))
df_final = df_final.withColumn("month", split_columns.getItem(1))
df_final = df_final.withColumn("sentiment", df_final.result.getItem(0))

df_final = df_final.withColumn("positive_0" , regexp_replace('sentiment','positive', '1'))
df_final = df_final.withColumn("positive_1" , regexp_replace('positive_0','negative', '0'))
df_final = df_final.withColumn("positive_2" , regexp_replace('positive_1','neutral', '0'))

df_final = df_final.withColumn("negative_0" , regexp_replace('sentiment','negative', '1'))
df_final = df_final.withColumn("negative_1" , regexp_replace('negative_0','positive', '0'))
df_final = df_final.withColumn("negative_2" , regexp_replace('negative_1','neutral', '0'))

df_final = df_final.withColumn("neutral_0" , regexp_replace('sentiment','neutral', '1'))
df_final = df_final.withColumn("neutral_1" , regexp_replace('neutral_0','positive', '0'))
df_final = df_final.withColumn("neutral_2" , regexp_replace('neutral_1','negative', '0'))

df_final = df_final.withColumn("positive", expr('CAST(positive_2 AS INTEGER)'))
df_final = df_final.withColumn("negative" , expr('CAST(negative_2 AS INTEGER)'))
df_final = df_final.withColumn("neutral" , expr('CAST(neutral_2 AS INTEGER)'))

df_final_sum = df_final.groupby("day","month").sum("positive","negative","neutral").orderBy("day")


display(df_final_sum)

############################################################6 SCRITTURA IN MONGODB#############################################################################

mesi3 = df_final_sum.write.format("com.mongodb.spark.sql.DefaultSource").mode("append").option("database","covid19geo").option("collection","mesi_sum1mln").save()

#############################################################7 TWEETS PIU' RETWEETATI###########################################################################

from pyspark.sql import functions as f
import re
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import col, expr, when
#MOSTRO LA TABELLA A CLASSIFICAZIONE AVVENUTA:

df_final = pipeline.fit(df_cleantext5).transform(df_cleantext5).select("created_at","text","sentiment.result","retweet_count")


split_columns = f.split(df_final["created_at"], " ")
df_final = df_final.withColumn("day", split_columns.getItem(2))
df_final= df_final1.withColumn("month", split_columns.getItem(1))
df_final = df_final2.withColumn("sentiment", df_final2.result.getItem(0))
display(df_final)

############################################################8 HASHTAGS PIU' UTILIZZATI#########################################################################

from pyspark.sql.functions import explode_outer, explode, posexplode_outer
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import desc

df_final = hashtag.select("entities.hashtags.text")
df_final.printSchema()
df_final2 = df_final.select(explode(df_final.text).alias("hashtags"))
df_final3 =  df_final2.select("hashtags")
df_final_sum = df_final3.groupby("hashtags").count()
df_final_sum2 = df_final_sum.sort(desc("count"))
#df_final_sum2 = df_final_sum.withColumn("count")
display(df_final_sum2)

###########################################################9 SENTIMENT MEDIO 20-26 APRILE#####################################################################

#pipeline.fit(df_clean).transform(df_clean).select("sentiment").show(truncate = False)
from pyspark.sql.functions import explode
from pyspark.sql.functions import expr, col
from pyspark.sql.types import DoubleType

df_final = pipeline.fit(df_duplicates).transform(df_duplicates).select("created_at","sentiment.result","sentiment.metadata")
split_columns = f.split(df_final["created_at"], " ")
df_final = df_final.withColumn("day", split_columns.getItem(2))
df_final = df_final.withColumn("month", split_columns.getItem(1))

df_final2 = df_final.select("result","day","month",explode(df_final.metadata))
df_final3 =  df_final2.select("result","day","month","col.positive","col.negative")

df_final4 = df_final3.withColumn("positive", df_final3["positive"].cast(DoubleType()))
df_final5 = df_final4.withColumn("negative", df_final3["negative"].cast(DoubleType()))

#df_final_sum = df_final5.groupby("day").avg("positive","negative").orderBy("day")
df_final6 = df_final5.orderBy("day")
display(df_final6)

df_final6.write.format("com.mongodb.spark.sql.DefaultSource").mode("append").option("database","covid19geo").option("collection","averageSentimentAvg").save()

###############################################10 SELEZIONE TWEETS CON LOCAZIONE NON NULLA######################################################################

df_cleanPlace = df_clean.where(col("place.country").isNotNull())

############################################################11 CONFRONTO PYSPARK-PYTHON########################################################################
import pandas as pd		
import re
from pyspark.sql import functions as f
from pyspark.sql.functions import col, explode, regexp_replace, split
from pyspark.sql.functions import *

filePath = "dbfs:/FileStore/tables/mesi250mila.csv"
file_type = "csv"

# CSV options
first_row_is_header = "false"
multiline = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .option("multiline", multiline) \
  .option("escape", "\"") \
  .load(filePath)

#display(df)
#df2.show(60)
#df.printSchema()

df1 = df.select('_c1')

df2 = df1.filter(df1._c1 != 'entities.hashtags')
df3 = df2.withColumn('_c1',explode(split('_c1','\\},\\{')))
df4 = df3.filter(df3._c1 != '[]')
split_columns = f.split(df4["_c1"], ",")
df5 = df4.withColumn("hashtags", split_columns.getItem(0))
df6 = df5.drop('_c1')
df7 = df6.withColumn('hashtags', regexp_replace('hashtags', '[\s\S]*"text":"', ''))
df8 = df7.withColumn('hashtags', regexp_replace('hashtags', '"', ''))
df_final_sum = df8.groupby("hashtags").count().sort(desc("count"))
display(df_final_sum)



import pandas as pd
from datetime import datetime
startTime = datetime.now()

filePath = "/dbfs/FileStore/tables/mesi250mila.csv"
file_type = "csv"

dfhash = pd.read_csv(filePath)
dtype={'_id': int}
low_memory=False
dfhash.dropna(inplace = True) 
print(dfhash.head(4))

df1 = dfhash['entities.hashtags']
df2 = pd.DataFrame(df1.str.split('},{', expand=True))
df3 = df2[df2 != '[]']
df4 = df3.melt()
df5 = df4.dropna()
df6 = pd.DataFrame(df5.value.str.split('","', expand=True))
df7 = df6.drop(columns=1)	
df8 = df7[0].str.replace('[\s\S]*"text":"','', regex = True)
df_final_sum = df8.value_counts()

print(datetime.now()-startTime)