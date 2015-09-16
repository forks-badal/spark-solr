package com.lucidworks.spark;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.Test;
import scala.Predef;
import scala.Tuple2;
import scala.collection.JavaConverters;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import static org.junit.Assert.assertTrue;
import java.util.Map;
import java.util.HashMap;
import scala.Predef;
import scala.Tuple2;
import scala.collection.JavaConverters;

/**
 * Tests for the SolrRelation implementation.
 */
public class SolrRelationTest extends RDDProcessorTestBase {

  protected transient SQLContext sqlContext;

  @Test
  public void testFilterSupport() throws Exception {

    SQLContext sqlContext = new SQLContext(jsc);

    String[] testData = new String[] {
      "1,a,x,1000",
      "2,b,y,2000",
      "3,c,z,3000",
      "4,a,x,4000"
    };

    String zkHost = cluster.getZkServer().getZkAddress();
    String testCollection = "testFilterSupport";
    deleteCollection(testCollection);
    deleteCollection("testFilterSupport2");
    buildCollection(zkHost, testCollection, testData, 2);

    Map<String, String> options = new HashMap<String, String>();
    options.put("zkhost", zkHost);
    options.put("collection", testCollection);

    DataFrame df = sqlContext.read().format("solr").options(options).load();

    df.show();

    long count = df.count();
    assertCount(testData.length, count, "*:*");

    count = df.filter(df.col("field1_s").equalTo("a")).count();
    assertCount(2, count, "field1_s == a");

    count = df.filter(df.col("field3_i").gt(3000)).count();
    assertCount(1, count, "field3_i > 3000");

    count = df.filter(df.col("field3_i").geq(3000)).count();
    assertCount(2, count, "field3_i >= 3000");

    count = df.filter(df.col("field3_i").lt(2000)).count();
    assertCount(1, count, "field3_i < 2000");

    count = df.filter(df.col("field3_i").leq(1000)).count();
    assertCount(1, count, "field3_i <= 1000");

    count = df.filter(df.col("field3_i").gt(2000).and(df.col("field2_s").equalTo("z"))).count();
    assertCount(1, count, "field3_i > 2000 AND field2_s == z");

    count = df.filter(df.col("field3_i").lt(2000).or(df.col("field1_s").equalTo("a"))).count();
    assertCount(2, count, "field3_i < 2000 OR field1_s == a");

    count = df.filter(df.col("field1_s").isNotNull()).count();
    assertCount(4, count, "field1_s IS NOT NULL");

    count = df.filter(df.col("field1_s").isNull()).count();
    assertCount(0, count, "field1_s IS NULL");

    // write to another collection to test writes
    String confName = "testConfig";
    File confDir = new File("src/test/resources/conf");
    int numShards = 2;
    int replicationFactor = 1;
    createCollection("testFilterSupport2", numShards, replicationFactor, 2, confName, confDir);

    options = new HashMap<String, String>();
    options.put("zkhost", zkHost);
    options.put("collection", "testFilterSupport2");

    df.write().format("solr").options(options).mode(SaveMode.Overwrite).save();
    Thread.sleep(1000);

    DataFrame df2 = sqlContext.read().format("solr").options(options).load();
    df2.show();

    deleteCollection(testCollection);
    deleteCollection("testFilterSupport2");
  }

  @Test
  public void testNestedDataFrames() throws Exception {
    SQLContext sqlContext = new SQLContext(jsc);
    String confName = "testConfig";
    File confDir = new File("src/test/resources/conf");
    int numShards = 2;
    int replicationFactor = 1;
    deleteCollection("testNested");
    createCollection("testNested", numShards, replicationFactor, 2, confName, confDir);
    List<StructField> fields = new ArrayList<StructField>();
    List<StructField> fields1 = new ArrayList<StructField>();
    List<StructField> fields2 = new ArrayList<StructField>();
    fields.add(DataTypes.createStructField("id", DataTypes.StringType, true));
    fields.add(DataTypes.createStructField("testing_s", DataTypes.StringType, true));
    fields1.add(DataTypes.createStructField("test1_s", DataTypes.StringType, true));
    fields1.add(DataTypes.createStructField("test2_s", DataTypes.StringType, true));
    fields2.add(DataTypes.createStructField("test11_s", DataTypes.StringType, true));
    fields2.add(DataTypes.createStructField("test12_s", DataTypes.StringType, true));
    fields2.add(DataTypes.createStructField("test13_s", DataTypes.StringType, true));
    fields1.add(DataTypes.createStructField("testtype_s", DataTypes.createStructType(fields2) , true));
    fields.add(DataTypes.createStructField("test_s", DataTypes.createStructType(fields1), true));
    StructType schema = DataTypes.createStructType(fields);
    Row dm = RowFactory.create("7", "test", RowFactory.create("test1", "test2", RowFactory.create("test11", "test12", "test13")));
    List<Row> list = new ArrayList<Row>();
    list.add(dm);
    JavaRDD<Row> rdd = jsc.parallelize(list);
    DataFrame df = sqlContext.createDataFrame(rdd, schema);
    HashMap<String, String> options = new HashMap<String, String>();
    String zkHost = cluster.getZkServer().getZkAddress();
    options = new HashMap<String, String>();
    options.put("zkhost", zkHost);
    options.put("collection", "testNested");
    options.put("preserveschema", "Y");
    df.write().format("solr").options(options).mode(SaveMode.Overwrite).save();
    Thread.sleep(1000);
    DataFrame df2 = sqlContext.read().format("solr").options(options).load();
    df2 = sqlContext.createDataFrame(df2.javaRDD(),df2.schema());
    df.show();
    df2.show();
    df2.registerTempTable("DFTEST");
    sqlContext.sql("SELECT test_s.testtype_s FROM DFTEST").show();
    deleteCollection("testNested");

  }

  protected void assertCount(long expected, long actual, String expr) {
    assertTrue("expected count == "+expected+" but got "+actual+" for "+expr, expected == actual);
  }
}
