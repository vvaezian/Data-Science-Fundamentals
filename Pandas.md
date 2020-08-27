### DataFrame

A **DataFrame** is a table of data.
````Python
x = np.arange(0,50)
df = pd.DataFrame({'x':x})

df = pd.DataFrame({'a':[True, False, True], 'b':[4, 5, 6], 'c':[7, 8, 9]})
# defining row by row: pd.DataFrame([[True, 4, 7], [False, 5, 8], [True, 6, 9]], columns=['a', 'b', 'c'])
# we can change index numbering by providing index list.
print(df)
#        a  b  c
# 0   True  4  7
# 1  False  5  8
# 2   True  6  9 
````
creating a dataframe with random data
````Python
df = pd.DataFrame(np.arange(12).reshape(3,4), columns=['A', 'B', 'C', 'D'])
````
stacking dataframes
````Python
pd.concat([a, b, c, d], ignore_index=True)
````
### Attributes ###
````
df.columns    # Index(['a', 'b', 'c'], dtype='object')
df.index      # RangeIndex(start=0, stop=3, step=1)
df.values     # [[True  4 7]
              #  [False 5 8]
              #  [True  6 9]]
````
### Methods ###
- `df.isnull()` converts values of df into true/false based on whether values are missing.  
- `df.any()` produces a Series where its index is the columns of df and its values are true/false based on whether there is at least one true in the column. `s.any()` produces one true/false value.
- `df.all()` produces a Series where its index is the columns of df and its values true/false based on whether all cell of column are true. `s.any()` produces one true/false value.  
- `df.dtypes`

Choosing some columns:
````Python
df[column_list] # e.x. reduced_by_cols = df[['a', 'c']]
````
Choosing some rows:
````Python
# index-based:
# if index is not the default (i.e. user provided index for the df), `loc` chooses rows by index names. 
df.iloc[row_index] # e.x. reduced_by_rows = df.iloc[[0, 2]]  

# value-based:
df.loc[[boolean expression to restrict rows], [list of columns to return]]
df.loc[df.myCol=='ldb', df.columns]

# to show rows that don't have null in a specific column:
df.loc[ df.myCol.notnull(), df.columns ]
````
* If we dont use double-brackets, the behaviour is different. `df.iloc[0, 1]` returns the element at row 0 column 1.  
`df.iloc[0]` returns a Series containing the row at index 0. `df['a']` returns a Series containing the column 'a'.

### Join, Merge
- `df1.join(df2)` by default performs left join. and keeps all columns
- `pd.merge(df1, df2, left_on='col_l', right_on='col_r')`

### Series
Making a column titlecase:
````python
df.a = df.a.str.title()
````

Casting to int for a column that contains Null values:
```python
df_bc['myCol'] = df_bc['myCol'].astype('Int64') 
```
Distinct Values together with their counts (excludes NaN by default)
```python
df.col.value_counts()
```

### SQL Connection ###
````python
### pyodbc
# SQL Server
import pyodbc
args = '''
        Driver={ODBC Driver 13 for SQL Server};
        Server=myServer;
        Database=myDB;
        UID=*****; # system user
        PWD=*****;
       '''
conn_mssql = pyodbc.connect(args)
cursor_mssql = conn_mssql.cursor()

# PostgreSQL on RDS
import psycopg2
connection_psql = psycopg2.connect( user="vvAdmin"
                                  , password="***"
                                  , port="5432"
                                  , database="myDB"
                                  , host="[ENDPOINT]")
cursor_psql = connection_psql.cursor()

### sqlalchemy
import sqlalchemy as sa
import urllib
params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};"
                                 "SERVER=xxx;"
                                 "DATABASE=yyy;"
                                 "Trusted_Connection=yes")
engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
# engine_rds_pgsql = sa.create_engine("postgresql+psycopg2://[USER]:[ENDPOINT]/[TABLE_NAME]")
````
### Read from SQL
```python
### small data
df_read = pd.read_sql(sql='tbl_name',  # or a query
                      con=engine,
                      columns=['x','y'],
                      chunksize=1000000
                     )
# If a column had int data type and had NULL values in SQL Server, it gets imported as float. 
# We can use the Int64 data type which is introduced in 0.24+ for nullables integer:
df_read.myCol = df_read.myCol.astype('Int64')

### big data
# Pandas data storage is not as efficient as H2O, so after importing each chunk, 
# we convert it to H2O dataframe (we cannot do it directly because H2O doesn't have ODBC connection yet)
# note that when we use chunksize, Pandas sometimes reads a few more lines (see: https://github.com/pandas-dev/pandas/issues/28153)
# so we need to take that into consideration

pd_frame_generator = pd.read_sql_table(table_name='tbl_name', con=engine, chunksize=500000)

pd_df_chunk = next(pd_frame_generator)
h2o_df_all = h2o.H2OFrame(pd_df_chunk, column_types={'col3':'enum', 'col5':'enum'})

for pd_df_chunk in pd_frame_generator:
  h2o_df_chunk = h2o.H2OFrame(pd_df_chunk, column_types={'col3':'enum', 'col5':'enum'})
  h2o_df_all = h2o_df_all.rbind(h2o_df_chunk)
```
### Write to SQL
```python
### small data
df_write.to_sql('tbl_name', con=engine, index=False, if_exists='append')

### big data
# SQL Server 
pd_df.to_csv('test.csv', sep='\t', header=False, index=False)
subprocess.call('bcp {t} in {f} -S {s} -U {u} -P {p} -c -t "{sep}" '.format(t='db_name.dbo.tbl_name',   # to
                                                                            f='/PATH/TO/FILE/test.csv', # from
                                                                            s='Server_Name', # to server
                                                                            u='XXX', 
                                                                            p="YYY", 
                                                                            sep='\t'), 
                shell=True)

# Postgres 
conn_pgsql = engine_pgsql.raw_connection()
cursor_pgsql = conn_pgsql.cursor()
output = io.StringIO()
df.to_csv(output, sep='\t', na_rep='Nan', header=False, index=False)
output.seek(0)
contents = output.getvalue()
cursor_pgsql.copy_from(output, 'destination_table', null='Nan')
conn_pgsql.commit()
````
### Write to CSV
```python
conn = pyodbc.connect(' ... ')
cursor = conn.cursor()

cursor.execute('SELECT ...')

col_headers = [ i[0] for i in cursor.description ]
rows = [ list(i) for i in cursor.fetchall()]  # Pandas require list of lists for rows. cursor returns list of tuples. So we cast to list.
df = pd.DataFrame(rows, columns=col_headers)

df.to_csv("test.csv", index=False)
```


## Misc. ##
````Python
print(df)
#        a  b  c
# 0   True  4  7
# 1  False  5  8
# 2   True  6  9 

print(df.b[df.a])
0    4
2    6
Name: b, dtype: int64
    
print(df.b[~df.a])
1    5
Name: b, dtype: int64

print((df.a==0).mean())   # 0.333333333333
````
