Sources: [Udacity Course (Intro to Hadoop and MapReduce)](https://classroom.udacity.com/courses/ud617), [Hadoop Wiki](https://wiki.apache.org/hadoop/)  

### Concepts ###

- HDFS: Hadoop Distributed File Systems  
As the files are uploaded to HDFS, each block (64MB or 128MB in newer versions) will get stored in three (the default) *data node* in the cluster. Two *NameNodes* (one for backup) keep track of what is where (meta-data).

- A *TaskTracker* is a service that manages tasks in a DataNode. It accespts Map, Reduce and Shuffle operations from a JobTracker.

- The *JobTracker* is the service within Hadoop that farms out MapReduce tasks to specific nodes in the cluster, ideally the nodes that have the data, or at least are in the same rack. After jobs are submitted to the Job tracker, it talks to the NameNode to determine the location of the data, and it locates DataNodes at or near the data which their *TaskTracker* has free slots. The JobTracker submits the work to the chosen TaskTrackers.

### Commands ###
- All the commands that interact with HDFS start with `Hadoop fs`.  
````
hadoop fs -ls/-cat/-mv/-cp/-rm/-tail/-mkdir ...
hadoop fs -put localFile [newName]  # places localFile into HDFS (note that the home dir is /user/hadoop )
hadoop fs -get hdfsFile [newName]   # retrieves data from HDFS and put it in local disc

````

- The mappers filter the data so we have only the data we need. For example the store name and sale amount among 6 columns
````Python
#! /usr/bin/python2.7

# mapper.py
import sys

# read standard input line by line
for line in sys.stdin:
    # strip off extra whitespace, split on tab and put the data in an array
    data = line.strip().split("\t")

    if len(data) == 6:
        date, time, store, item, cost, payment = data
        
        # print out the data that will be passed to the reducer
        print "{0}\t{1}".format(store, cost)
````

- 'Shuffle and sort' is the phase between mapper and reducer. 
It ensures the values for any particular key are collected together, then sends the keys and the list of values to the reducer.

- The reducers gets data like this (sorted on keys):
````
key1  val1
key1  val2  
key1  val3  
key2  val4  
key2  val5
...
````  
It is necessary that for any key, the destination partition is the same. If the key "cat" is generated in two separate (key, value) pairs, they must both be reduced together.  As a result it is not guaranteed that reducers get the same amount of jobs. In fact some reducers may get no keys. More details [here](https://developer.yahoo.com/hadoop/tutorial/module5.html#partitioning).
 
````Python
#! /usr/bin/python2.7

# reducer.py
import sys

salesTotal = 0
oldKey = None

for line in sys.stdin:
    data = line.strip().split("\t")
    if len(data) != 2:
        continue

    thisKey, thisSale = data
    if oldKey and oldKey != thisKey:  # if key changed
        print oldKey, "\t", salesTotal
        oldKey = thisKey
        salesTotal = 0

    salesTotal += float(thisSale)
    oldKey = thisKey

if oldKey != None:
    print oldKey, "\t", salesTotal

````
* In Python 2 It is important to remember to cast to float or int. Because `2 < 'a'` doesn't produce error (evaluates to True)**.

- Hadoop streaming makes it possible to write mappers and reducers in any programming languages rather than just Java:  
````BASH
hadoop jar PATH/TO/JAR -files mapper.py,reducer.py -mapper mapper.py -reducer reducer.py  -input InputDir -output OutputDirName
# mapper.py and reducer.py are read from the local directory, while InputDir is read from HDFS. Output directory will be created in HDFS after the programs execution.
# jar file location: `/usr/lib/hadoop-mapreduce/hadoop-streaming.jar`

# To Make an alias for the above command:
run_mapreduce() { hadoop jar PATH/TO/JAR -files $1,$2 -mapper $1 -reducer $2 -input $3 -output $4 
                }
alias hs=run_mapreduce
````



- To test our mapper and reducer (we should always test with small dataset before running the code on a huge dataset):
````cat testInput | ./mapper.py | sort | ./reducer.py````
- Use `lynx` together with the link provided when the job is running to monitor the job (can be used after job completion as well).

### Hadoop and S3 ###
To transfer data from S3 to Hadoop: ([source](https://hortonworks.github.io/hdp-aws/s3-copy-data/index.html))
````
hadoop distcp s3://PATH hadoopDir
````
