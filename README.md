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

    A[Data Source] --> B["Apache NiFi<br>(Pecah dataset menjadi<br>beberapa bagian untuk<br>simulasi stream data)"]
    B --> C["HDFS<br>(Store raw data)"]
    C --> D["Apache Spark<br>(Exploratory Data Analysis<br>— EDA —)"]
    D --> E["Apache Spark<br>(Filter: 'generation'<br>panjang > 10 karakter)"]
    E --> F["HDFS<br>(Store filtered data)"]
    F --> G["Apache Spark<br>(Split: 70% train,<br>30% test)"]
    
    %% Cabang Train
    G --> H1["HDFS<br>(Store train data)"]
    H1 --> I1["Apache Spark NLP<br>(Ekstraksi Fitur BERT<br>pada data train)"]
    I1 --> J1["HDFS<br>(Store train features)"]
    J1 --> K["Spark MLlib<br>(Train: SVM / SVC)"]
    K --> L["HDFS<br>(Store trained model)"]
    
    %% Cabang Test
    G --> H2["HDFS<br>(Store test data)"]
    H2 --> I2["Apache Spark NLP<br>(Ekstraksi Fitur BERT<br>pada data test)"]
    I2 --> J2["HDFS<br>(Store test features)"]
    
    %% Gabung untuk evaluasi
    K --> M
    J2 --> M
    L --> M
    M["Python + Matplotlib / Seaborn via PySpark<br>(Evaluation & Visualization)"]

    %% Terapkan kelas warna
    class A source;
    class B nifi;
    class C,F,H1,H2,J1,J2,L hdfs;
    class D,E,G,I1,I2 spark;
    class K mllib;
    class M viz;
    class D eda;
```

| Komponen                 | URL Akses                                      | Fungsi                                             |
| ------------------------ | ---------------------------------------------- | -------------------------------------------------- |
| **Hadoop NameNode**      | [http://localhost:9870](http://localhost:9870) | Monitoring HDFS (lihat file yang diupload ke HDFS) |
| **YARN ResourceManager** | [http://localhost:8088](http://localhost:8088) | Melihat job Spark yang sedang jalan                |
| **Spark Master**         | [http://localhost:8080](http://localhost:8080) | Dashboard cluster Spark                            |
| **NiFi**                 | [http://localhost:8081/nifi/](http://localhost:8081/nifi/) | Data ingestion pipeline                            |
| **Jupyter Notebook**     | [http://localhost:8888](http://localhost:8888) | Analisis dan model machine learning                |

