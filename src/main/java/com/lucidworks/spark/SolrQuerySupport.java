package com.lucidworks.spark;

import com.lucidworks.spark.query.*;
import com.lucidworks.spark.util.SolrJsonSupport;
import org.apache.http.NameValuePair;
import org.apache.http.client.utils.URLEncodedUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.StreamingResponseCallback;
import org.apache.solr.client.solrj.impl.CloudSolrClient;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrDocumentList;
import org.apache.solr.common.SolrException;
import org.apache.solr.common.params.ModifiableSolrParams;
import org.apache.solr.common.util.NamedList;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.*;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.io.Serializable;
import java.net.ConnectException;
import java.net.SocketException;
import java.nio.charset.StandardCharsets;
import java.util.*;


public class SolrQuerySupport implements Serializable{

  public static Logger log = Logger.getLogger(SolrQuerySupport.class);

  public static final int DEFAULT_PAGE_SIZE = 1000;

  public static final Map<String,DataType> solrDataTypes = new HashMap<String, DataType>();
  static {
    solrDataTypes.put("solr.StrField", DataTypes.StringType);
    solrDataTypes.put("solr.TextField", DataTypes.StringType);
    solrDataTypes.put("solr.BoolField", DataTypes.BooleanType);
    solrDataTypes.put("solr.TrieIntField", DataTypes.IntegerType);
    solrDataTypes.put("solr.TrieLongField", DataTypes.LongType);
    solrDataTypes.put("solr.TrieFloatField", DataTypes.FloatType);
    solrDataTypes.put("solr.TrieDoubleField", DataTypes.DoubleType);
    solrDataTypes.put("solr.TrieDateField", DataTypes.TimestampType);
  }

  /**
   * Iterates over the entire results set of a query (all hits).
   */
  public static class QueryResultsIterator extends PagedResultsIterator<SolrDocument> {

    public QueryResultsIterator(SolrClient solrServer, SolrQuery solrQuery, String cursorMark) {
      super(solrServer, solrQuery, cursorMark);
    }

    protected List<SolrDocument> processQueryResponse(QueryResponse resp) {
      return resp.getResults();
    }
  }

  /**
   * Returns an iterator over TermVectors
   */
  public static class TermVectorIterator extends PagedResultsIterator<Vector> {

    private String field = null;
    private HashingTF hashingTF = null;

    public TermVectorIterator(SolrClient solrServer, SolrQuery solrQuery, String cursorMark, String field, int numFeatures) {
      super(solrServer, solrQuery, cursorMark);
      this.field = field;
      hashingTF = new HashingTF(numFeatures);
    }

    protected List<Vector> processQueryResponse(QueryResponse resp) {
      NamedList<Object> response = resp.getResponse();

      NamedList<Object> termVectorsNL = (NamedList<Object>)response.get("termVectors");
      if (termVectorsNL == null)
        throw new RuntimeException("No termVectors in response! " +
          "Please check your query to make sure it is requesting term vector information from Solr correctly.");

      List<org.apache.spark.mllib.linalg.Vector> termVectors = new ArrayList<Vector>(termVectorsNL.size());
      Iterator<Map.Entry<String, Object>> iter = termVectorsNL.iterator();
      while (iter.hasNext()) {
        Map.Entry<String, Object> next = iter.next();
        String nextKey = next.getKey();
        Object nextValue = next.getValue();
        if (nextValue instanceof NamedList) {
          NamedList nextList = (NamedList) nextValue;
          Object fieldTerms = nextList.get(field);
          if (fieldTerms != null && fieldTerms instanceof NamedList) {
            termVectors.add(SolrTermVector.newInstance(nextKey, hashingTF, (NamedList<Object>) fieldTerms));
          }
        }
      }

      SolrDocumentList docs = resp.getResults();
      totalDocs = docs.getNumFound();

      return termVectors;
    }
  }

  public static SolrQuery toQuery(String queryString) {

    if (queryString == null || queryString.length() == 0)
      queryString = "*:*";

    SolrQuery q = new SolrQuery();
    if (queryString.indexOf("=") == -1) {
      // no name-value pairs ... just assume this single clause is the q part
      q.setQuery(queryString);
    } else {
      NamedList<Object> params = new NamedList<Object>();
      for (NameValuePair nvp : URLEncodedUtils.parse(queryString, StandardCharsets.UTF_8)) {
        String value = nvp.getValue();
        if (value != null && value.length() > 0) {
          String name = nvp.getName();
          if ("sort".equals(name)) {
            if (value.indexOf(" ") == -1) {
              q.addSort(SolrQuery.SortClause.asc(value));
            } else {
              String[] split = value.split(" ");
              q.addSort(SolrQuery.SortClause.create(split[0], split[1]));
            }
          } else {
            params.add(name, value);
          }
        }
      }
      q.add(ModifiableSolrParams.toSolrParams(params));
    }

    Integer rows = q.getRows();
    if (rows == null)
      q.setRows(DEFAULT_PAGE_SIZE);

    List<SolrQuery.SortClause> sorts = q.getSorts();
    if (sorts == null || sorts.isEmpty())
      q.addSort(SolrQuery.SortClause.asc("id"));

    return q;
  }

  public static QueryResponse querySolr(SolrClient solrServer, SolrQuery solrQuery, int startIndex, String cursorMark) throws SolrServerException {
    return querySolr(solrServer, solrQuery, startIndex, cursorMark, null);
  }

  public static QueryResponse querySolr(SolrClient solrServer, SolrQuery solrQuery, int startIndex, String cursorMark, StreamingResponseCallback callback) throws SolrServerException {
    QueryResponse resp = null;
    try {
      if (cursorMark != null) {
        solrQuery.setStart(0);
        solrQuery.set("cursorMark", cursorMark);
      } else {
        solrQuery.setStart(startIndex);
      }

      if (callback != null) {
        resp = solrServer.queryAndStreamResponse(solrQuery, callback);
      } else {
        resp = solrServer.query(solrQuery);
      }
    } catch (Exception exc) {

      log.error("Query ["+solrQuery+"] failed due to: "+exc);

      // re-try once in the event of a communications error with the server
      Throwable rootCause = SolrException.getRootCause(exc);
      boolean wasCommError =
        (rootCause instanceof ConnectException ||
          rootCause instanceof IOException ||
          rootCause instanceof SocketException);
      if (wasCommError) {
        try {
          Thread.sleep(2000L);
        } catch (InterruptedException ie) {
          Thread.interrupted();
        }

        try {
          if (callback != null) {
            resp = solrServer.queryAndStreamResponse(solrQuery, callback);
          } else {
            resp = solrServer.query(solrQuery);
          }
        } catch (Exception excOnRetry) {
          if (excOnRetry instanceof SolrServerException) {
            throw (SolrServerException)excOnRetry;
          } else {
            throw new SolrServerException(excOnRetry);
          }
        }
      } else {
        if (exc instanceof SolrServerException) {
          throw (SolrServerException)exc;
        } else {
          throw new SolrServerException(exc);
        }
      }
    }

    return resp;
  }

   public static final int[] getPivotFieldRange(StructType schema, String pivotPrefix) {
    StructField[] schemaFields = schema.fields();
    int startAt = -1;
    int endAt = -1;
    for (int f=0; f < schemaFields.length; f++) {
      String name = schemaFields[f].name();
      if (startAt == -1 && name.startsWith(pivotPrefix)) {
        startAt = f;
      }
      if (startAt != -1 && !name.startsWith(pivotPrefix)) {
        endAt = f-1; // we saw the last field in the range before this field
        break;
      }
    }
    return new int[]{startAt,endAt};
  }

  public static final void fillPivotFieldValues(String rawValue, Object[] row, StructType schema, String pivotPrefix) {
    int[] range = getPivotFieldRange(schema, pivotPrefix);
    for (int i=range[0]; i <= range[1]; i++) row[i] = 0;
    try {
      row[schema.fieldIndex(pivotPrefix+rawValue.toLowerCase())] = 1;
    } catch (IllegalArgumentException ia) {
      row[range[1]] = 1;
    }
  }
  public static DataType getsqlDataType(String s) {
    if (s.toLowerCase().equals("double")) {
      return DataTypes.DoubleType;
    }
    if (s.toLowerCase().equals("byte")) {
      return DataTypes.ByteType;
    }
    if (s.toLowerCase().equals("short")) {
      return DataTypes.ShortType;
    }
    if (((s.toLowerCase().equals("int")) || (s.toLowerCase().equals("integer")))) {
      return DataTypes.IntegerType;
    }
    if (s.toLowerCase().equals("long")) {
      return DataTypes.LongType;
    }
    if (s.toLowerCase().equals("String")) {
      return DataTypes.StringType;
    }
    if (s.toLowerCase().equals("boolean")) {
      return DataTypes.BooleanType;
    }
    if (s.toLowerCase().equals("timestamp")) {
      return DataTypes.TimestampType;
    }
    if (s.toLowerCase().equals("date")) {
      return DataTypes.DateType;
    }
    if (s.toLowerCase().equals("vector")) {
      return new VectorUDT();
    }
    if (s.toLowerCase().equals("matrix")) {
      return new MatrixUDT();
    }
    if (s.contains(":") && s.split(":")[0].toLowerCase().equals("array")) {
      return getArrayTypeRecurse(s,0);
    }
    return DataTypes.StringType;
  }

  public static DataType getArrayTypeRecurse(String s, int fromIdx) {
    if (s.contains(":") && s.split(":")[1].toLowerCase().equals("array")) {
      fromIdx = s.indexOf(":", fromIdx);
      s = s.substring(fromIdx+1, s.length());
      return DataTypes.createArrayType(getArrayTypeRecurse(s,fromIdx));
    }
    return DataTypes.createArrayType(getsqlDataType(s.split(":")[1]));
  }

  public static final class PivotField implements Serializable {
    public final String solrField;
    public final String prefix;
    public final String otherSuffix;
    public final int maxCols;

    public PivotField(String solrField, String prefix) {
      this(solrField, prefix, 10);
    }

    public PivotField(String solrField, String prefix, int maxCols) {
      this(solrField, prefix, maxCols, "other");
    }

    public PivotField(String solrField, String prefix, int maxCols, String otherSuffix) {
      this.solrField = solrField;
      this.prefix = prefix;
      this.maxCols = maxCols;
      this.otherSuffix = otherSuffix;
    }
  }

 public static class SolrFieldMeta {
    String fieldType;
    boolean isMultiValued;
    String fieldTypeClass;
  }

  public static Map<String,SolrFieldMeta> getFieldTypes(String[] fields, String solrBaseUrl, String collection) {

    // specific field list
    StringBuilder sb = new StringBuilder();
    for (int f=0; f < fields.length; f++) {
      if (f > 0) sb.append(",");
      sb.append(fields[f]);
    }
    String fl = sb.toString();

    String fieldsUrl = solrBaseUrl+collection+"/schema/fields?showDefaults=true&includeDynamic=true&fl="+fl;
    List<Map<String, Object>> fieldInfoFromSolr = null;
    try {
      Map<String, Object> allFields =
              SolrJsonSupport.getJson(SolrJsonSupport.getHttpClient(), fieldsUrl, 2);
      fieldInfoFromSolr = (List<Map<String, Object>>)allFields.get("fields");
    } catch (Exception exc) {
      String errMsg = "Can't get field metadata from Solr using request "+fieldsUrl+" due to: " + exc;
      log.error(errMsg);
      if (exc instanceof RuntimeException) {
        throw (RuntimeException)exc;
      } else {
        throw new RuntimeException(errMsg, exc);
      }
    }

    // avoid looking up field types more than once
    Map<String,String> fieldTypeToClassMap = new HashMap<String,String>();

    // collect mapping of Solr field to type
    Map<String,SolrFieldMeta> fieldTypeMap = new HashMap<String,SolrFieldMeta>();
    for (String field : fields) {

      if (fieldTypeMap.containsKey(field))
        continue;

      SolrFieldMeta tvc = null;
      for (Map<String,Object> map : fieldInfoFromSolr) {
        String fieldName = (String)map.get("name");
        if (field.equals(fieldName)) {
          tvc = new SolrFieldMeta();
          tvc.fieldType = (String)map.get("type");

          Object multiValued = map.get("multiValued");
          if (multiValued != null && multiValued instanceof Boolean) {
            tvc.isMultiValued = ((Boolean)multiValued).booleanValue();
          } else {
            tvc.isMultiValued = "true".equals(String.valueOf(multiValued));
          }
        }
      }

      if (tvc == null || tvc.fieldType == null) {
        String errMsg = "Can't figure out field type for field: " + field + ". Check you Solr schema and retry.";
        log.error(errMsg);
        throw new RuntimeException(errMsg);
      }

      String fieldTypeClass = fieldTypeToClassMap.get(tvc.fieldType);
      if (fieldTypeClass != null) {
        tvc.fieldTypeClass = fieldTypeClass;
      } else {
        String fieldTypeUrl = solrBaseUrl+collection+"/schema/fieldtypes/"+tvc.fieldType;
        try {
          Map<String, Object> fieldTypeMeta =
                  SolrJsonSupport.getJson(SolrJsonSupport.getHttpClient(), fieldTypeUrl, 2);
          tvc.fieldTypeClass = SolrJsonSupport.asString("/fieldType/class", fieldTypeMeta);
          fieldTypeToClassMap.put(tvc.fieldType, tvc.fieldTypeClass);
        } catch (Exception exc) {
          String errMsg = "Can't get field type metadata for "+tvc.fieldType+" from Solr due to: " + exc;
          log.error(errMsg);
          if (exc instanceof RuntimeException) {
            throw (RuntimeException)exc;
          } else {
            throw new RuntimeException(errMsg, exc);
          }
        }
      }

      fieldTypeMap.put(field, tvc);
    }

    return fieldTypeMap;
  }



  protected static List<String> collectCursors(final SolrClient solrClient, final SolrQuery origQuery) throws SolrServerException {
    return collectCursors(solrClient, origQuery, false);
  }

  protected static List<String> collectCursors(final SolrClient solrClient, final SolrQuery origQuery, final boolean distrib) throws SolrServerException {
    List<String> cursors = new ArrayList<String>();

    final SolrQuery query = origQuery.getCopy();
    // tricky - if distrib == false, then set the param, otherwise, leave it out (default is distrib=true)
    if (!distrib) {
      query.set("distrib", false);
    } else {
      query.remove("distrib");
    }
    query.setFields("id");

    String nextCursorMark = "*";
    while (true) {
      cursors.add(nextCursorMark);
      query.set("cursorMark", nextCursorMark);

      QueryResponse resp = null;
      try {
        resp = solrClient.query(query);
      } catch (Exception exc) {
        if (exc instanceof SolrServerException) {
          throw (SolrServerException)exc;
        } else {
          throw new SolrServerException(exc);
        }
      }

      nextCursorMark = resp.getNextCursorMark();
      if (nextCursorMark == null || resp.getResults().isEmpty())
        break;
    }

    return cursors;
  }



   public static JavaRDD<ShardSplit> splitShard(JavaSparkContext jsc, final SolrQuery origQuery, List<String> shards, final String splitFieldName, final int splitsPerShard, String collection) {
    final SolrQuery query = origQuery.getCopy();
    query.set("distrib", false);
    query.set("collection", collection);
    query.setStart(0);
    if (query.getRows() == null)
      query.setRows(DEFAULT_PAGE_SIZE); // default page size

    // get field type of split field
    final DataType fieldDataType;
    if ("_version_".equals(splitFieldName)) {
      fieldDataType = DataTypes.LongType;
    } else {
      Map<String,SolrFieldMeta> fieldMetaMap = SolrQuerySupport.getFieldTypes(new String[]{splitFieldName}, shards.get(0), collection);
      SolrFieldMeta solrFieldMeta = fieldMetaMap.get(splitFieldName);
      if (solrFieldMeta != null) {
        String fieldTypeClass = solrFieldMeta.fieldTypeClass;
        fieldDataType = SolrQuerySupport.solrDataTypes.get(fieldTypeClass);
      } else {
        log.warn("No field metadata found for "+splitFieldName+", assuming it is a String!");
        fieldDataType = DataTypes.StringType;
      }
      if (fieldDataType == null)
        throw new IllegalArgumentException("Cannot determine DataType for split field "+splitFieldName);
    }

    JavaRDD<ShardSplit> splitsRDD = jsc.parallelize(shards, shards.size()).flatMap(new FlatMapFunction<String, ShardSplit>() {
      public Iterable<ShardSplit> call(String shardUrl) throws Exception {

        ShardSplitStrategy splitStrategy = null;
        if (fieldDataType == DataTypes.LongType || fieldDataType == DataTypes.IntegerType) {
          splitStrategy = new NumberFieldShardSplitStrategy();
        } else if (fieldDataType == DataTypes.StringType) {
          splitStrategy = new StringFieldShardSplitStrategy();
        } else {
          throw new IllegalArgumentException("Can only split shards on fields of type: long, int, or string!");
        }
        List<ShardSplit> splits =
          splitStrategy.getSplits(shardUrl, query, splitFieldName, splitsPerShard);

        log.info("Found " + splits.size() + " splits for " + splitFieldName + ": " + splits);

        return splits;
      }
    });
    return splitsRDD;
  }
}
