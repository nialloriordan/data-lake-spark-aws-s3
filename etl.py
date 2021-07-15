import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import (
    year,
    month,
    dayofmonth,
    hour,
    weekofyear,
    date_format,
)
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import monotonically_increasing_id, col
import logging

# set up logging to file and stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s:%(lineno)s - %(message)s",
    handlers=[logging.FileHandler("spark_job.log"), logging.StreamHandler()],
)

config = configparser.ConfigParser()
config.read("dl.cfg")

os.environ["AWS_ACCESS_KEY_ID"] = config["AWS"]["AWS_ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = config["AWS"]["AWS_SECRET_ACCESS_KEY"]


def create_spark_session():
    """initialise a spark session with hadoop aws for s3 access

    Returns:
        spark session
    """
    spark = SparkSession.builder.config(
        "spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0"
    ).getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """Process song data from S3 using spark and
    load back to S3 as a dimensional table

    Args:
        spark (pyspark.sql.SparkSession): spark session
        input_data (str): s3 bucket path to input data
        output_data (str): s3 bucket path to output data
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, "song_data", "*/*/*", "*.json")

    # read song data file
    logging.info("Loading song data files from s3 with spark...")
    df = spark.read.json(song_data)
    logging.info("Finished loading song data file from s3 with spark\n")

    # extract columns to create songs table
    logging.info("Selecting columns to create songs table...")
    songs_table = df.select(
        "song_id", "title", "artist_id", "year", "duration"
    ).dropDuplicates()
    logging.info("Finished selecting columns to create songs table")

    # songs output file path
    songs_output_path = os.path.join(output_data, "songs", "songs.parquet")
    # write songs table to parquet files partitioned by year and artist
    logging.info("Saving processed song data to s3...")
    songs_table.write.partitionBy("year", "artist_id").mode("overwrite").parquet(
        songs_output_path
    )
    logging.info("Finished saving song data to s3\n")

    # extract columns to create artists table
    logging.info("Selecting columns to create artists table...")
    artists_table = df.select(
        col("artist_id").alias("artist_id"),
        col("artist_name").alias("name"),
        col("artist_location").alias("location"),
        col("artist_latitude").alias("latitude"),
        col("artist_longitude").alias("longitude"),
    ).dropDuplicates()
    logging.info("Finished selecting columns to create artists table")

    # songs output file path
    artists_output_path = os.path.join(output_data, "artists", "artists.parquet")

    # write artists table to parquet files
    logging.info("Saving processed artists data to s3...")
    artists_table.write.mode("overwrite").parquet(artists_output_path)
    logging.info("Finished saving processed artists data to s3\n")


def process_log_data(spark, input_data, output_data):
    """Process log data from S3 using spark and load
    back into S3 as a dimensional table

    Args:
        spark (pyspark.sql.SparkSession): spark session
        input_data (str): s3 bucket path to input data
        output_data (str): s3 bucket path to output data
    """
    # get filepath to log data file
    log_data = os.path.join(input_data, "log_data", "*/*", "*.json")

    # read log data file
    logging.info("Loading log data files from s3 with spark...")
    df = spark.read.json(log_data)
    logging.info("Finished loading log data files from s3\n")

    # filter by actions for song plays
    logging.info(
        "Filtering log data to only include records associated with song plays..."
    )
    df = df.where(df.page == "NextSong")
    logging.info("Finished filtering log data\n")

    # users output file path
    users_output_path = os.path.join(output_data, "users", "users.parquet")

    # extract columns for users table
    logging.info("Selecting relevant columns for users table...")
    users_table = df.select(
        "userId", "firstName", "lastName", "gender", "level"
    ).dropDuplicates()
    logging.info("Finished selecting relevant columns for users table")

    # write users table to parquet files
    logging.info("Saving users table to s3...")
    users_table.write.mode("overwrite").parquet(users_output_path)
    logging.info("Finished saving users table to s3\n")

    # create timestamp column from original timestamp column
    logging.info("Converting datetime column to be of type timestamp...")
    get_datetime = udf(
        lambda x: datetime.fromtimestamp(int(x) / 1000.0), TimestampType()
    )
    df = df.withColumn("dt", get_datetime("ts"))
    logging.info("Finished converting datetime column")

    # time output file path
    time_output_path = os.path.join(output_data, "time", "time.parquet")

    # extract columns to create time table
    logging.info("Selecting columns for time table...")
    time_table = df.select(
        col("dt").alias("start_time"),
        hour("dt").alias("hour"),
        dayofmonth("dt").alias("day"),
        weekofyear("dt").alias("week"),
        month("dt").alias("month"),
        year("dt").alias("year"),
        date_format("dt", "E").alias("weekday"),
    ).dropDuplicates()
    logging.info("Finished selecting columns for time table")

    # write time table to parquet files partitioned by year and month
    logging.info("Saving time table to s3...")
    time_table.write.partitionBy("year", "month").mode("overwrite").parquet(
        time_output_path
    )
    logging.info("Saved time table to s3\n")

    # songs output file path
    songs_output_path = os.path.join(output_data, "songs", "songs.parquet")

    # read in song data to use for songplays table
    logging.info("Loading song data to use for songplays table...")
    song_df = spark.read.parquet(songs_output_path)
    logging.info("Finished loading song data\n")

    # extract columns from joined song and log datasets to create songplays table
    logging.info("Joining song and log datasets to create songplays table...")
    songplays_table = df.join(song_df, df.song == song_df.title)
    logging.info("Finished joining song and log datasets")

    # select relevant columns for songplays table i.e the fact table
    logging.info("Selecting relevant columns for songplays table...")
    songplays_table = songplays_table.select(
        monotonically_increasing_id().alias("songplay_id"),
        col("dt").alias("start_time"),
        col("userId").alias("user_id"),
        "level",
        "song_id",
        "artist_id",
        col("sessionId").alias("session_id"),
        "location",
        col("userAgent").alias("user_agent"),
        month("dt").alias("month"),
        year("dt").alias("year"),
    ).dropDuplicates()
    logging.info("Finished selecting relevant columns for songplays table")

    # songplays output file path
    songplays_output_path = os.path.join(output_data, "songplays", "songplays.parquet")

    # write songplays table to parquet files partitioned by year and month
    logging.info("Saving songplays table to s3...")
    songplays_table.write.partitionBy("year", "month").mode("overwrite").parquet(
        songplays_output_path
    )
    logging.info("Saved songplays table to s3\n")


def main():
    """Main function to build ETL pipeline
    1. Extracts data from S3
    2. Process data into analytical tables using spark
    3. Loads data back into S3
    """
    logging.info("Starting spark job...")
    logging.info("Creating spark session...")
    spark = create_spark_session()
    # To improve the performance of Spark with S3,
    # use version 2 of the output committer algorithm
    spark.sparkContext._jsc.hadoopConfiguration().set(
        "mapreduce.fileoutputcommitter.algorithm.version", "2"
    )
    logging.info("Spark session intialised\n\n")
    input_data = config["S3"]["INPUT_DATA"]
    output_data = config["S3"]["OUTPUT_DATA"]
    logging.info("Processing song data...\n")
    process_song_data(spark, input_data, output_data)
    logging.info("Processed song data\n\n")
    logging.info("Processing log data...\n")
    process_log_data(spark, input_data, output_data)
    logging.info("Processed log data\n")
    logging.info("Spark job completed\n'n")


if __name__ == "__main__":
    main()
