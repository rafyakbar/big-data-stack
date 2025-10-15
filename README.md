```mermaid
flowchart TD
    A[Data Source] --> B["Apache NiFi<br>(Ingest data)"]
    B --> C["HDFS<br>(Store raw data)"]
    C --> D["Apache Spark<br>(Preprocessing: Tokenize, Clean, TF-IDF Vectorization)"]
    D --> E["HDFS<br>(Store preprocessed data)"]
    E --> F["Spark MLlib<br>(Train Supervised Model)"]
    F --> G["HDFS<br>(Store trained model)"]
    G --> H["Python + Matplotlib / Seaborn via PySpark<br>(Evaluation & Visualization)"]
```

| Komponen                 | URL Akses                                      | Fungsi                                             |
| ------------------------ | ---------------------------------------------- | -------------------------------------------------- |
| **Hadoop NameNode**      | [http://localhost:9870](http://localhost:9870) | Monitoring HDFS (lihat file yang diupload ke HDFS) |
| **YARN ResourceManager** | [http://localhost:8088](http://localhost:8088) | Melihat job Spark yang sedang jalan                |
| **Spark Master**         | [http://localhost:8080](http://localhost:8080) | Dashboard cluster Spark                            |
| **NiFi**                 | [http://localhost:8081](http://localhost:8081) | Data ingestion pipeline                            |
| **Jupyter Notebook**     | [http://localhost:8888](http://localhost:8888) | Analisis dan model machine learning                |

