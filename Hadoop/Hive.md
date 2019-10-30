HiveQL automatically translates SQL-like queries into MapReduce jobs (if the query needs any mapping or reducing).  

In master node EC2 instance shell, typing `hive` starts Hive. 


- Create Table Using Data in S3  
This creates a table in the specified bucket using the data that is in the bucket.
````SQL
CREATE EXTERNAL TABLE testTable 
( aDate STRING, time STRING, store STRING, item  STRING, cost FLOAT, payment STRING )
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' 
LOCATION 's3://PATH/TO/BUCKET/'; # for hdfs location note that the home directory of hdfs is /user/hadoop 
````
- Information on a table  
````SQL
show create table <TableName>;
show tblproperties <TableName>;
describe formatted <TableName>;
````
- Information on the job (the steps)
````
explain <QUERY>
````
### Configuring Hive ###
Hive configuration file is `/etc/hive/conf.dist/hive-site.xml`. We can change it for the current session with `--hiveconf` when envoking `hive` (e.x. `$ hive --hiveconf hive.execution.engine=spark`).

- `hive.execution.engine` determines the execution engine (`tez` (default), `spark`, `mr` [=MapReduce] (depreciated in Hive 2.x)).  
For spark integration to work we need to do [this](https://www.linkedin.com/pulse/hive-spark-configuration-common-issues-mohamed-k) configuration (for EMR, the mentioned file in the link is `/usr/lib/hive/bin/hive`).

**Adding column names when showing query results**  

     set hive.cli.print.header=true;
     set hive.resultset.use.unique.column.names=false;  # To prevent adding table name prefix to columns

To make this permanent add them to `.hiverc` (create one if not exists) or `hive-site.xml` with value `true`. 
