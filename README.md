```mermaid
flowchart LR
    %% Definisi gaya warna berdasarkan komponen
    classDef source fill:#e0e0e0,stroke:#555,color:#000,rx:12,ry:12;
    classDef nifi fill:#ff6f00,stroke:#d14d00,color:#fff,rx:12,ry:12;
    classDef hdfs fill:#1f77b4,stroke:#0d4a6b,color:#fff,rx:12,ry:12;
    classDef spark fill:#e31c3d,stroke:#a0001a,color:#fff,rx:12,ry:12;
    classDef mllib fill:#7e318e,stroke:#5a1a66,color:#fff,rx:12,ry:12;
    classDef viz fill:#2ca02c,stroke:#1b661b,color:#fff,rx:12,ry:12;

    A[Data Source] --> B["Apache NiFi<br>(Ingest data)"]
    B --> C["HDFS<br>(Store raw data)"]
    C --> D["Apache Spark<br>(Preprocessing:<br>Tokenize, Clean, TF-IDF Vectorization)"]
    D --> E["HDFS<br>(Store preprocessed data)"]
    E --> F["Spark MLlib<br>(Train Supervised Model)"]
    F --> G["HDFS<br>(Store trained model)"]
    G --> H["Python + Matplotlib / Seaborn via PySpark<br>(Evaluation & Visualization)"]

    class A source;
    class B nifi;
    class C,E,G hdfs;
    class D spark;
    class F mllib;
    class H viz;
```

| Komponen                 | URL Akses                                      | Fungsi                                             |
| ------------------------ | ---------------------------------------------- | -------------------------------------------------- |
| **Hadoop NameNode**      | [http://localhost:9870](http://localhost:9870) | Monitoring HDFS (lihat file yang diupload ke HDFS) |
| **YARN ResourceManager** | [http://localhost:8088](http://localhost:8088) | Melihat job Spark yang sedang jalan                |
| **Spark Master**         | [http://localhost:8080](http://localhost:8080) | Dashboard cluster Spark                            |
| **NiFi**                 | [http://localhost:8081](http://localhost:8081) | Data ingestion pipeline                            |
| **Jupyter Notebook**     | [http://localhost:8888](http://localhost:8888) | Analisis dan model machine learning                |

