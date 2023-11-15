# deep-learning-challenge

<div style="display: inline_block"><br/>
  <img align="center" alt="Colaboratory" src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" />

<div style="display: inline_block"><br/>
  <img align="center" alt="python" src="http://ForTheBadge.com/images/badges/made-with-python.svg" />


  ## Introduction

In this project, it use SparkSQL to analyze home sales data. The goal is to determine key metrics, create temporary views, partition data, and perform various operations using Spark. 

## Libraries

The project utilizes the following libraries:

`PySpark`: PySpark is the Python API for Apache Spark, a powerful open-source distributed computing system. It is used for processing large-scale data.

`!pip install pyspark` Run the provided code in a Spark environment, ensuring that the Spark session is properly configured.

Execute each query sequentially, observing the results and runtimes.

## How the Code Works

The project involves analyzing home sales data using SparkSQL. Key tasks include reading data from an AWS S3 bucket, creating temporary views, running SQL queries on the data, caching and uncaching tables, and working with partitioned data.

Let's break down how the code works step by step:

### 1. Import Libraries

```diff
+ import findspark
+ findspark.init()
+ from pyspark.sql import SparkSession
+ import time
```

* `findspark.init()`: Initializes the findspark module to locate the Spark installation.
* `SparkSession`: Initializes a Spark session, which is the entry point for reading data and executing operations in Spark.
* `time`: Imports the time module for measuring query runtimes.
  
### 2. Create a SparkSession

```diff
+ spark = SparkSession.builder.appName("SparkSQL").getOrCreate()
```

* Initiates a Spark session with the application name "SparkSQL."

### 3. Read Data from AWS S3 Bucket
  
```diff
+ from pyspark import SparkFiles
+ url = "https://2u-data-curriculum-team.s3.amazonaws.com/dataviz-classroom/v1.2/22-big-data/home_sales_revised.csv"
+ spark.sparkContext.addFile(url)
+ home_sales_df = spark.read.csv(SparkFiles.get("home_sales_revised.csv"), sep=",", header=True)
```
* Reads home sales data from an AWS S3 bucket into a Spark DataFrame.

### 4. Create a Temporary View
```diff
+ home_sales_df.createOrReplaceTempView('home_sales')
```
* Creates a temporary view named 'home_sales' for the DataFrame, allowing the use of SparkSQL queries.

### 5. SparkSQL Queries

* The code contains several SparkSQL queries that answer specific questions about the home sales data. These queries include filtering, grouping, and ordering operations.
  * Query 1: Calculate the average price for a four-bedroom house sold in each year rounded to two decimal places.
  * Query 2: Determine the average price of a home for each year it was built with 3 bedrooms and 3 bathrooms rounded to two decimal places.
  * Query 3: Find the average price of a home for each year built with 3 bedrooms, 3 bathrooms, two floors, and a size greater than or equal to 2,000 square feet rounded to two decimal places.
  * Query 4: Calculate the "view" rating for the average price of a home where the price is greater than or equal to $350,000. Measure the runtime of this query.
  * Query 5: Run a query using the cached data, filtering out view ratings with an average price greater than or equal to $350,000. Compare the runtime with the uncached version.
  * Query 6: Run a query on the parquet data, filtering out view ratings with an average price greater than or equal to $350,000. Compare the runtime with the cached version.

### 6.  Caching

```diff
+ spark.sql("cache table home_sales")
+ spark.catalog.isCached('home_sales')
```
* Caches the 'home_sales' temporary table to improve query performance and checks if it is cached.

### 7. Query on Partitioned Parquet Data

```diff
home_sales_df.write.partitionBy("date_built").mode("overwrite").parquet("home_sales_data")
df_p = spark.read.parquet('home_sales_data')
df_p.createOrReplaceTempView("temp_table_parquet")
```
* Partitions the home sales data by the "date_built" field in the parquet format.
* Reads the partitioned parquet data and creates a temporary view.

### 8. Uncaching

```diff
spark.sql("UNCACHE TABLE home_sales")
spark.catalog.isCached('home_sales')
```
* Uncaches the 'home_sales' temporary table and checks if it is no longer cached.

## Conclusion

The code provides a comprehensive example of using SparkSQL to analyze and manipulate large-scale home sales data efficiently. It covers various Spark operations, caching strategies, and demonstrates the impact of caching on query runtimes.

## Developer

[<img src="https://avatars.githubusercontent.com/u/133066908?v=4" width=115><br><sub>Ricardo De Los Rios</sub>](https://github.com/ricardodelosrios) 
