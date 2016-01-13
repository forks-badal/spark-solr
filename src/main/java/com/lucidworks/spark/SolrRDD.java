package com.lucidworks.spark;

import com.lucidworks.spark.SolrQuerySupport.PivotField;
import com.lucidworks.spark.SolrQuerySupport.QueryResultsIterator;
import com.lucidworks.spark.SolrQuerySupport.SolrFieldMeta;
import com.lucidworks.spark.SolrQuerySupport.TermVectorIterator;
import com.lucidworks.spark.query.*;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.impl.CloudSolrClient;
import org.apache.solr.client.solrj.response.FacetField;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrDocumentList;
import org.apache.solr.common.params.ModifiableSolrParams;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

public class SolrRDD implements Serializable {

  public static Logger log = Logger.getLogger(SolrRDD.class);

  public static final int DEFAULT_PAGE_SIZE = 1000;

  protected String zkHost;
  protected String collection;
  protected transient JavaSparkContext sc;

  public SolrRDD(String collection) {
    this("localhost:9983", collection); // assume local embedded ZK if not supplied
  }

  public SolrRDD(String zkHost, String collection) {
    this.zkHost = zkHost;
    this.collection = collection;
  }

  public String getCollection() {
    return collection;
  }


  public void setSc(JavaSparkContext jsc){
    sc = jsc;
  }
  public JavaSparkContext getSc() {
    return sc;
  }

  /**
   * Get a document by ID using real-time get
   */
  public JavaRDD<SolrDocument> get(JavaSparkContext jsc, final String docId) throws SolrServerException {
    CloudSolrClient cloudSolrServer = SolrSupport.getSolrClient(zkHost);
    ModifiableSolrParams params = new ModifiableSolrParams();
    params.set("collection", collection);
    params.set("qt", "/get");
    params.set("id", docId);
    QueryResponse resp = null;
    try {
      resp = cloudSolrServer.query(params);
    } catch (Exception exc) {
      if (exc instanceof SolrServerException) {
        throw (SolrServerException)exc;
      } else {
        throw new SolrServerException(exc);
      }
    }
    SolrDocument doc = (SolrDocument) resp.getResponse().get("doc");
    List<SolrDocument> list = (doc != null) ? Arrays.asList(doc) : new ArrayList<SolrDocument>();
    return jsc.parallelize(list, 1);
  }

  public JavaRDD<SolrDocument> query(JavaSparkContext jsc, final SolrQuery query, boolean useDeepPagingCursor) throws SolrServerException {
    if (useDeepPagingCursor)
      return queryDeep(jsc, query);

    query.set("collection", collection);
    CloudSolrClient cloudSolrServer = SolrSupport.getSolrClient(zkHost);
    List<SolrDocument> results = new ArrayList<SolrDocument>();
    Iterator<SolrDocument> resultsIter = new QueryResultsIterator(cloudSolrServer, query, null);
    while (resultsIter.hasNext()) results.add(resultsIter.next());
    return jsc.parallelize(results, 1);
  }

  /**
   * Makes it easy to query from the Spark shell.
   */
  public JavaRDD<SolrDocument> query(SparkContext sc, String queryStr) throws SolrServerException {
    return queryShards(new JavaSparkContext(sc), SolrQuerySupport.toQuery(queryStr));
  }

  public JavaRDD<SolrDocument> query(SparkContext sc, SolrQuery solrQuery) throws SolrServerException {
    return queryShards(new JavaSparkContext(sc), solrQuery);
  }

  public JavaRDD<SolrDocument> queryShards(JavaSparkContext jsc, final SolrQuery origQuery) throws SolrServerException {
    // first get a list of replicas to query for this collection
    List<String> shards = SolrSupport.buildShardList(SolrSupport.getSolrClient(zkHost), collection);

    final SolrQuery query = origQuery.getCopy();

    // we'll be directing queries to each shard, so we don't want distributed
    query.set("distrib", false);
    query.set("collection", collection);
    query.setStart(0);
    if (query.getRows() == null)
      query.setRows(DEFAULT_PAGE_SIZE); // default page size

    // parallelize the requests to the shards
    JavaRDD<SolrDocument> docs = jsc.parallelize(shards, shards.size()).flatMap(
      new FlatMapFunction<String, SolrDocument>() {
        public Iterable<SolrDocument> call(String shardUrl) throws Exception {
          return new StreamingResultsIterator(SolrSupport.getHttpSolrClient(shardUrl), query, "*");
        }
      }
    );
    return docs;
  }

  public JavaRDD<Row> toRows(StructType schema, JavaRDD<SolrDocument> docs) {
    final String[] queryFields = schema.fieldNames();
    JavaRDD<Row> rows = docs.map(new Function<SolrDocument, Row>() {
      public Row call(SolrDocument doc) throws Exception {
        Object[] vals = new Object[queryFields.length];
        for (int f = 0; f < queryFields.length; f++) {
          Object fieldValue = doc.getFieldValue(queryFields[f]);
          if (fieldValue != null) {
            if (fieldValue instanceof Collection) {
              vals[f] = ((Collection) fieldValue).toArray();
            } else if (fieldValue instanceof Date) {
              vals[f] = new java.sql.Timestamp(((Date) fieldValue).getTime());
            } else {
              vals[f] = fieldValue;
            }
          }
        }
        return RowFactory.create(vals);
      }
    });
    return rows;
  }

  public JavaRDD<SolrDocument> queryDeep(JavaSparkContext jsc, final SolrQuery origQuery) throws SolrServerException {
    return queryDeep(jsc, origQuery, 36);
  }

  public JavaRDD<SolrDocument> queryDeep(JavaSparkContext jsc, final SolrQuery origQuery, final int maxPartitions) throws SolrServerException {

    final SolrClient solrClient = SolrSupport.getSolrClient(zkHost);
    final SolrQuery query = origQuery.getCopy();
    query.set("collection", collection);
    query.setStart(0);
    if (query.getRows() == null)
      query.setRows(DEFAULT_PAGE_SIZE); // default page size

    long startMs = System.currentTimeMillis();
    List<String> cursors = SolrQuerySupport.collectCursors(solrClient, query, true);
    long tookMs = System.currentTimeMillis() - startMs;
    log.info("Took "+tookMs+"ms to collect "+cursors.size()+" cursor marks");
    int numPartitions = Math.min(maxPartitions,cursors.size());

    JavaRDD<String> cursorJavaRDD = jsc.parallelize(cursors, numPartitions);
    // now we need to execute all the cursors in parallel
    JavaRDD<SolrDocument> docs = cursorJavaRDD.flatMap(
      new FlatMapFunction<String, SolrDocument>() {
        public Iterable<SolrDocument> call(String cursorMark) throws Exception {
          return SolrQuerySupport.querySolr(SolrSupport.getSolrClient(zkHost), query, 0, cursorMark).getResults();
        }
      }
    );
    return docs;
  }


  public JavaRDD<SolrDocument> queryShards(JavaSparkContext jsc, final SolrQuery origQuery, final String splitFieldName, final int splitsPerShard) throws SolrServerException {
    // if only doing 1 split per shard, then queryShards does that already
    if (splitFieldName == null || splitsPerShard <= 1)
      return queryShards(jsc, origQuery);

    long timerDiffMs = 0L;
    long timerStartMs = 0L;

    // first get a list of replicas to query for this collection
    List<String> shards = SolrSupport.buildShardList(SolrSupport.getSolrClient(zkHost), collection);

    timerStartMs = System.currentTimeMillis();

    // we'll be directing queries to each shard, so we don't want distributed
    JavaRDD<ShardSplit> splitsRDD = SolrQuerySupport.splitShard(jsc, origQuery, shards, splitFieldName, splitsPerShard, collection);
    List<ShardSplit> splits = splitsRDD.collect();
    timerDiffMs = (System.currentTimeMillis() - timerStartMs);
    log.info("Collected " + splits.size() + " splits, took " + timerDiffMs + "ms");

    // parallelize the requests to the shards
    JavaRDD<SolrDocument> docs = jsc.parallelize(splits, splits.size()).flatMap(
      new FlatMapFunction<ShardSplit, SolrDocument>() {
        public Iterable<SolrDocument> call(ShardSplit split) throws Exception {
          return new StreamingResultsIterator(SolrSupport.getHttpSolrClient(split.getShardUrl()), split.getSplitQuery(), "*");
        }
      }
    );
    return docs;
  }

  public JavaRDD<Vector> queryTermVectors(JavaSparkContext jsc, final SolrQuery query, final String field, final int numFeatures) throws SolrServerException {
    // first get a list of replicas to query for this collection
    List<String> shards = SolrSupport.buildShardList(SolrSupport.getSolrClient(zkHost), collection);

    if (query.getRequestHandler() == null) {
      query.setRequestHandler("/tvrh");
    }
    query.set("shards.qt", query.getRequestHandler());

    query.set("tv.fl", field);
    query.set("fq", field + ":[* TO *]"); // terms field not null!
    query.set("tv.tf_idf", "true");

    // we'll be directing queries to each shard, so we don't want distributed
    query.set("distrib", false);
    query.set("collection", collection);
    query.setStart(0);
    if (query.getRows() == null)
      query.setRows(DEFAULT_PAGE_SIZE); // default page size

    // parallelize the requests to the shards
    JavaRDD<Vector> docs = jsc.parallelize(shards, shards.size()).flatMap(
      new FlatMapFunction<String, Vector>() {
        public Iterable<Vector> call(String shardUrl) throws Exception {
          return new TermVectorIterator(SolrSupport.getHttpSolrClient(shardUrl), query, "*", field, numFeatures);
        }
      }
    );
    return docs;
  }

  public DataFrame asTempTable(SQLContext sqlContext, String queryString, String tempTable) throws Exception {
    SolrQuery solrQuery = SolrQuerySupport.toQuery(queryString);
    DataFrame rows = applySchema(sqlContext, solrQuery, query(sqlContext.sparkContext(), solrQuery));
    rows.registerTempTable(tempTable);
    return rows;
  }

  public DataFrame queryForRows(SQLContext sqlContext, String queryString) throws Exception {
    SolrQuery solrQuery = SolrQuerySupport.toQuery(queryString);
    return applySchema(sqlContext, solrQuery, query(sqlContext.sparkContext(), solrQuery));
  }

  public DataFrame applySchema(SQLContext sqlContext, SolrQuery query, JavaRDD<SolrDocument> docs) throws Exception {
    // now convert each SolrDocument to a Row object
    StructType schema = getQuerySchema(query);
    JavaRDD<Row> rows = toRows(schema, docs);
    return sqlContext.applySchema(rows, schema);
  }

  public StructType getQuerySchema(SolrQuery query) throws Exception {
    CloudSolrClient solrServer = SolrSupport.getSolrClient(zkHost);
    // Build up a schema based on the fields requested
    String fieldList = query.getFields();
    String[] fields = null;
    if (fieldList != null) {
      fields = query.getFields().split(",");
    } else {
      // just go out to Solr and get 10 docs and extract a field list from that
      SolrQuery probeForFieldsQuery = query.getCopy();
      probeForFieldsQuery.remove("distrib");
      probeForFieldsQuery.set("collection", collection);
      probeForFieldsQuery.set("fl", "*");
      probeForFieldsQuery.setStart(0);
      probeForFieldsQuery.setRows(10);
      QueryResponse probeForFieldsResp = solrServer.query(probeForFieldsQuery);
      SolrDocumentList hits = probeForFieldsResp.getResults();
      Set<String> fieldSet = new TreeSet<String>();
      for (SolrDocument hit : hits)
        fieldSet.addAll(hit.getFieldNames());
      fields = fieldSet.toArray(new String[0]);
    }

    if (fields == null || fields.length == 0)
      throw new AnalysisException("Query ("+query+") does not specify any fields needed to build a schema!", null, null);

    Set<String> liveNodes = solrServer.getZkStateReader().getClusterState().getLiveNodes();
    if (liveNodes.isEmpty())
      throw new RuntimeException("No live nodes found for cluster: "+zkHost);
    String solrBaseUrl = solrServer.getZkStateReader().getBaseUrlForNodeName(liveNodes.iterator().next());
    if (!solrBaseUrl.endsWith("?"))
      solrBaseUrl += "/";

    Map<String,SolrFieldMeta> fieldTypeMap = SolrQuerySupport.getFieldTypes(fields, solrBaseUrl, collection);
    List<StructField> listOfFields = new ArrayList<StructField>();
    for (String field : fields) {
      SolrFieldMeta fieldMeta = fieldTypeMap.get(field);
      DataType dataType = (fieldMeta != null) ? SolrQuerySupport.solrDataTypes.get(fieldMeta.fieldTypeClass) : null;
      if (dataType == null) dataType = DataTypes.StringType;

      if (fieldMeta != null && fieldMeta.isMultiValued) {
        dataType = new ArrayType(dataType, true);
      }

      boolean nullable = !"id".equals(field);

      listOfFields.add(DataTypes.createStructField(field, dataType, nullable));
    }

    return DataTypes.createStructType(listOfFields);
  }

  public Map<String,Double> getLabels(String labelField) throws SolrServerException {
    SolrQuery solrQuery = new SolrQuery("*:*");
    solrQuery.setRows(0);
    solrQuery.set("collection", collection);
    solrQuery.addFacetField(labelField);
    solrQuery.setFacetMinCount(1);
    QueryResponse qr = SolrQuerySupport.querySolr(SolrSupport.getSolrClient(zkHost), solrQuery, 0, null);
    List<String> values = new ArrayList<>();
    for (FacetField.Count f : qr.getFacetField(labelField).getValues()) {
      values.add(f.getName());
    }

    Collections.sort(values);
    final Map<String,Double> labelMap = new HashMap<>();
    double d = 0d;
    for (String label : values) {
      labelMap.put(label, new Double(d));
      d += 1d;
    }

    return labelMap;
  }

  /**
   * Allows you to pivot a categorical field into multiple columns that can be aggregated into counts, e.g.
   * a field holding HTTP method (http_verb=GET) can be converted into: http_method_get=1, which is a common
   * task when creating aggregations.
   */
  public DataFrame withPivotFields(final DataFrame solrData, final PivotField[] pivotFields) throws IOException, SolrServerException {

    final StructType schemaWithPivots = toPivotSchema(solrData.schema(), pivotFields);

    JavaRDD<Row> withPivotFields = solrData.javaRDD().map(new Function<Row, Row>() {
      @Override
      public Row call(Row row) throws Exception {
        Object[] fields = new Object[schemaWithPivots.size()];
        for (int c=0; c < row.length(); c++)
          fields[c] = row.get(c);

        for (PivotField pf : pivotFields)
          SolrQuerySupport.fillPivotFieldValues(row.getString(row.fieldIndex(pf.solrField)), fields, schemaWithPivots, pf.prefix);

        return RowFactory.create(fields);
      }
    });

    return solrData.sqlContext().createDataFrame(withPivotFields, schemaWithPivots);
  }

  public StructType toPivotSchema(final StructType baseSchema, final PivotField[] pivotFields) throws IOException, SolrServerException {
    List<StructField> pivotSchemaFields = new ArrayList<>();
    pivotSchemaFields.addAll(Arrays.asList(baseSchema.fields()));
    for (PivotField pf : pivotFields) {
      for (StructField sf : getPivotSchema(pf.solrField, pf.maxCols, pf.prefix, pf.otherSuffix)) {
        pivotSchemaFields.add(sf);
      }
    }
    return DataTypes.createStructType(pivotSchemaFields);
  }

  public List<StructField> getPivotSchema(String fieldName, int maxCols, String fieldPrefix, String otherName) throws IOException, SolrServerException {
    final List<StructField> listOfFields = new ArrayList<StructField>();
    SolrQuery q = new SolrQuery("*:*");
    q.set("collection", collection);
    q.setFacet(true);
    q.addFacetField(fieldName);
    q.setFacetMinCount(1);
    q.setFacetLimit(maxCols);
    q.setRows(0);
    FacetField ff = SolrQuerySupport.querySolr(SolrSupport.getSolrClient(zkHost), q, 0, null).getFacetField(fieldName);
    for (FacetField.Count f : ff.getValues()) {
      listOfFields.add(DataTypes.createStructField(fieldPrefix+f.getName().toLowerCase(), DataTypes.IntegerType, false));
    }
    if (otherName != null) {
      listOfFields.add(DataTypes.createStructField(fieldPrefix+otherName, DataTypes.IntegerType, false));
    }
    return listOfFields;
  }

}
