### HDFS vs. S3 ###

There are two ways that s3 can be used with hadoop's Map/Reduce ([source](https://www.datasciencecentral.com/profiles/blogs/s3-as-input-or-output-for-hadoop-mr-jobs)):
1. As a replacement for HDFS using the s3 block filesystem (i.e. using it as a reliable distributed filesystem with support for very large files)  
2.  As a convenient repository for data input to and output from MapReduce. In the second case HDFS is still used for the Map/Reduce phase. Note also, that by using s3 as an input to MapReduce you lose the data locality optimization, which may be significant.

Regarding the first option ([source](https://databricks.com/blog/2017/05/31/top-5-reasons-for-choosing-s3-over-hdfs.html)):
- When using HDFS and getting perfect data locality, it is possible to get ~3GB/node local read throughput on some of the instance types (e.g. i2.8xl, roughly 90MB/s per core). Databrick's DBIO (their cloud I/O optimization module) provides optimized connectors to S3 and can sustain ~600MB/s read throughput on i2.8xl (roughly 20MB/s per core). 
- However, a big benefit with S3 is we can separate storage from compute, and as a result, we can just launch a larger cluster for a smaller period of time to increase throughput, up to allowable physical limits.
- Most of the big data systems (e.g., Spark, Hive) rely on HDFS' *atomic rename* feature to support atomic writes: that is, the output of a job is observed by the readers in an "all or nothing" fashion. This is important for data integrity because when a job fails, no partial data should be written out to corrupt the dataset.  
S3's lack of atomic directory renames has been a critical problem for guaranteeing data integrity.

### Hadoop vs. SQL ###
In big data (volume [Terabyte+], variety [unstructured, ...], velocity [extreme data arrival rates]) Hadoop is either faster or cheaper.  
In non-big data Hadoop can be useful: ([source](https://community.hortonworks.com/questions/57073/hadoop-versus-sql-server-or-odi.html))  
- Easy ingestion (schema-on-read)
- Cheap storage compared to SQL, so a good option for storig staging data of ETL jobs.  


### Data Lake ###
- A **Data Lake** uses a flat architecture to store data (ELT), i.e. data is stored in its native format.
- While data warehouses use schema-on-write (the structure of the data is enforced at the time of entering into the database), data lakes have schema-on-read.
- While data warehouses can be used for Batch reporting, BI, and visualizations, data lakes can be used for predictive analytics, data discovery, and profiling.
- The main difference between Data Lakes and Data Warehouses seems to be that Data Lakes store unprocessed data while Data Warehoues store processed data.

### Services ###
#### Amazon **Redshift** 
- Uses a columnar storage
- Has two types of storage DC2 and RA3. In DC2 storage and compure are NOT decoupled. It's suitable for data sizes under 1 TB. For more than 1 TB RA3 should be used where storage and compute are decoupled.
- Using Redshift Spectrum we can query S3 data (like Athena)
#### Kinesis Firehose 
- Moves the incoming data to S3, Redshift or Elasticsearch.  
#### Kinesis Streams
- Collects and process incoming data (holds for 24 hours). The processed data then can be stored in S3 or others. A common use-case is the real-time aggregation of data followed by loading the aggregate data into a data warehouse or map-reduce cluster.  
#### Kinesis Analytics
- To process and analyze the streaming data continuously  
#### Amazon DynamoDB
- It is a NoSQL database. NoSQL DB's can be grouped into the following categories: 
  - *columnar* (`Cassandra`, `HBase`)
  - *key-value store* (`DynamoDB`, `Riak`) 
  - *document-store* (`MongoDB`, `CouchDB`)
  - *graph* (`Neo4j`, `OrientDB`)  
  
- DynamoDB uses three basic data model units: Tables, Items, and Attributes. Tables are collections of Items, and Items are collections of Attributes. Attributes are basic units of information, like key-value pairs. Items are like rows in an RDBMS table, except that DynamoDB requires a Primary Key. Tables are like tables in relational databases, except that in DynamoDB, tables do not have fixed schemas associated with them. 

### ML services ###
- Amazon **Rekognition** for images and video
- Amazon **Lex** for chatbot integration
- Amazon **Polly** for text to speech
- Amazon **Translate** for natural language translation
- Amazon **Transcribe** for speech recognition
- Amazon **Personalize** for individualized recommendations for customers
- Amazon **Forecast** for accurate time-series forecasting
- Amazon **Textract** to extract text and data from virtually any document
- Amazon **Comprehend** to find relationships in text
- Amazon **Comprehend Medical** for detecting useful information in unstructured clinical text
