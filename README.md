```mermaid
flowchart TD
    %% Definisi gaya warna berdasarkan komponen
    classDef source fill:#e0e0e0,stroke:#555,color:#000,rx:12,ry:12;
    classDef nifi fill:#ff6f00,stroke:#d14d00,color:#fff,rx:12,ry:12;
    classDef hdfs fill:#1f77b4,stroke:#0d4a6b,color:#fff,rx:12,ry:12;
    classDef spark fill:#e31c3d,stroke:#a0001a,color:#fff,rx:12,ry:12;
    classDef mllib fill:#7e318e,stroke:#5a1a66,color:#fff,rx:12,ry:12;
    classDef viz fill:#2ca02c,stroke:#1b661b,color:#fff,rx:12,ry:12;
    classDef eda fill:#ff9800,stroke:#d16e00,color:#fff,rx:12,ry:12;
    classDef intermediate fill:#9c27b0,stroke:#6a0080,color:#fff,rx:12,ry:12;

    A[Data Source] --> B["Apache NiFi<br>(Simulate Streaming)"]
    B --> C["HDFS<br>(Store raw data)"]
    C --> D["Apache Spark<br>(Filter Data)"]

    %% Cabang Train
    D --> H1["Train Data"]
    H1 --> I1["Apache Spark MLlib<br>(Preprocessing & Feature Extraction)"]
    I1 --> K["Apache Spark MLlib<br>(Train Model)"]
    K --> L["HDFS<br>(Store Trained model)"]

    %% Cabang Test
    D --> H2["Test Data"]
    H2 --> I2["Apache Spark MLlib<br>(Preprocessing & Feature Extraction)"]

    %% Gabungan untuk evaluasi
    I2 --> M["Evaluation"]
    L --> M

    %% Terapkan kelas warna
    class A source;
    class B nifi;
    class C,L hdfs;
    class D,I1,I2 spark;
    class K mllib;
    class M viz;
    class H1,H2 intermediate;

```

| Komponen                 | URL Akses                                      | Fungsi                                             |
| ------------------------ | ---------------------------------------------- | -------------------------------------------------- |
| **Hadoop NameNode**      | [http://localhost:9870](http://localhost:9870) | Monitoring HDFS (lihat file yang diupload ke HDFS) |
| **YARN ResourceManager** | [http://localhost:8088](http://localhost:8088) | Melihat job Spark yang sedang jalan                |
| **Spark Master**         | [http://localhost:8080](http://localhost:8080) | Dashboard cluster Spark                            |
| **NiFi**                 | [http://localhost:8081/nifi/](http://localhost:8081/nifi/) | Data ingestion pipeline                            |
| **Jupyter Notebook**     | [http://localhost:8888](http://localhost:8888) | Analisis dan model machine learning                |

