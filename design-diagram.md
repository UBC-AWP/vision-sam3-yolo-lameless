flowchart TB
    subgraph frontend [Frontend Interfaces]
        Upload[Data Upload]
        VideoProc[Video Processing UI]
        Pairwise[Pairwise Comparison]
        Triplet[Triplet Comparison]
        Viz[Visualization Dashboard]
        Training[Training Module]
    end
    
    subgraph backend [Backend Pipelines]
        Curation[Clip Curation]
        YOLO[YOLO Detection]
        SAM3[SAM3 Segmentation]
        DINO[DINOv3 Embeddings]
        TLEAP[T-LEAP Pose]
        TCN[TCN Pipeline]
        Transformer[Transformer Pipeline]
        GraphT[Graph Transformer]
        ML[Tabular ML]
        Fusion[Ensemble Fusion]
    end
    
    subgraph hitl [Human-in-the-Loop]
        Consensus[Human Consensus]
        RaterRel[Rater Reliability]
        ActiveLearn[Active Learning]
    end
    
    Upload --> Curation
    Curation --> YOLO & SAM3 & DINO & TLEAP
    TLEAP --> TCN & Transformer
    DINO --> GraphT
    YOLO & SAM3 & DINO & TLEAP & TCN & Transformer & GraphT & ML --> Fusion
    Consensus --> Fusion
    Fusion --> Viz
    ActiveLearn --> Pairwise & Triplet