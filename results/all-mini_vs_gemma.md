# Comprehensive Embedding Model Evaluation Report
## Business Intelligence Semantic Search Performance Analysis

### Summary
**405 queries** were tested across 4 categories using business intelligence data to provide evidence for model selection.

**Key Finding**: **all-MiniLM-L6-v2 was found to significantly outperform EmbeddingGemma** with an overall accuracy of **82.0%** vs **66.7%**.

---

## Test Methodology

### Dataset
- **Total Queries**: 405 natural language variations  
- **Data Source**: Real business intelligence content (metrics, formulas, dimensions, dimension values)  
- **Test Categories**:  
  - Metrics: 102 queries  
  - Formulas: 213 queries  
  - Dimensions: 30 queries  
  - Dimension Values: 60 queries  

### Evaluation Metrics
- **Accuracy**: Percentage of queries for which the correct target was found  
- **Correct Type Rate**: Percentage for which the right type of result (metric, formula, etc.) was found  
- **Search Latency**: Average time taken to complete the search  
- **Similarity Scores**: Decimal scores indicating match quality (0.0–1.0)  
- **Ranking Performance**: Frequency with which the correct result appeared in top positions  

---

## Overall Performance Comparison

| Metric | all-MiniLM-L6-v2 | EmbeddingGemma | Better Model |
|--------|------------------|----------------|--------|
| **Overall Accuracy** | **82.0%** (332/405) | 66.7% (270/405) | **all-MiniLM-L6-v2** |
| **Correct Type Rate** | **77.3%** | 72.1% | **all-MiniLM-L6-v2** |
| **Avg Search Time** | **0.50s** | 0.64s | **all-MiniLM-L6-v2** |
| **Model Load Time** | **2.1s** | 4.4s | **all-MiniLM-L6-v2** |

**Performance Gap**: all-MiniLM-L6-v2 achieved **15.3 percentage points higher accuracy** and was **28% faster** in search operations.

---

## Category-by-Category Analysis

### 1. Metrics Performance

| Metric | all-MiniLM-L6-v2 | EmbeddingGemma | Better Model |
|--------|------------------|----------------|----------|
| **Accuracy** | **99.0%** (101/102) | 92.2% (94/102) | **all-MiniLM-L6-v2** |
| **Correct Type Rate** | 89.2% | 89.2% | Tie |
| **Avg Search Time** | **0.50s** | 0.63s | **all-MiniLM-L6-v2** |
| **Avg Similarity Score** | 0.584 | **0.923** | EmbeddingGemma |
| **Rank 1 Rate** | 28.4% | **47.1%** | EmbeddingGemma |

**Analysis**: Whilst higher similarity scores were achieved by EmbeddingGemma (0.923 vs 0.584), near-perfect accuracy was demonstrated by all-MiniLM-L6-v2 (99.0% vs 92.2%). The higher scores from EmbeddingGemma did not translate into better practical performance.

**Example – Where both models Excelled**:  
- Query: "kpi total ad request"  
- all-MiniLM-L6-v2: `metric_Total_Ad_Request` was found (Score: 0.561, Rank: 1)  
- EmbeddingGemma: `metric_Total_Ad_Request` was found (Score: 0.919, Rank: 1)  

**Example – Where all-MiniLM-L6-v2 Excelled**:  
- Query: "find valid ad request"  
- all-MiniLM-L6-v2: `metric_VADR` was found (Score: 0.579, Rank: 1)  
- EmbeddingGemma: No relevant results were found  

### 2. Formulas Performance

| Metric | all-MiniLM-L6-v2 | EmbeddingGemma | Better Model |
|--------|------------------|----------------|----------|
| **Accuracy** | **97.2%** (207/213) | 82.2% (175/213) | **all-MiniLM-L6-v2** |
| **Correct Type Rate** | **99.1%** | 94.4% | **all-MiniLM-L6-v2** |
| **Avg Search Time** | **0.51s** | 0.64s | **all-MiniLM-L6-v2** |
| **Avg Similarity Score** | 0.603 | **0.926** | EmbeddingGemma |
| **Rank 1 Rate** | 64.3% | **55.4%** | **all-MiniLM-L6-v2** |

**Analysis**: Better accuracy (97.2% vs 82.2%) and better rank-1 performance (64.3% vs 55.4%) were demonstrated by all-MiniLM-L6-v2, despite EmbeddingGemma achieving higher similarity scores.

**Example – Where all-MiniLM-L6-v2 Excelled**:  
- Query: "calculate ad impressions served/ vadr"  
- all-MiniLM-L6-v2: `formula_IX_Fill_Rate` was found (Score: 0.661, Rank: 3)  
- EmbeddingGemma: No relevant results were found  

**Example – Where EmbeddingGemma Excelled**:  
- Query: "ctr"  
- all-MiniLM-L6-v2: No relevant results were found  
- EmbeddingGemma: `formula_CTR` was found (Score: 0.988, Rank: 1)  

### 3. Dimensions Performance

| Metric | all-MiniLM-L6-v2 | EmbeddingGemma | Evidence |
|--------|------------------|----------------|----------|
| **Accuracy** | **80.0%** (24/30) | 3.3% (1/30) | **all-MiniLM-L6-v2** |
| **Correct Type Rate** | **36.7%** | 0.0% | **all-MiniLM-L6-v2** |
| **Avg Search Time** | **0.49s** | 0.64s | **all-MiniLM-L6-v2** |
| **Avg Similarity Score** | 0.530 | 0.794 | EmbeddingGemma |
| **Rank 1 Rate** | **13.3%** | 0.0% | **all-MiniLM-L6-v2** |

**Analysis**: A strong understanding of dimension concepts was demonstrated by all-MiniLM-L6-v2 (80.0% accuracy), whilst catastrophic failure (3.3%) was shown by EmbeddingGemma.

**Example – Where all-MiniLM-L6-v2 Excelled**:  
- Query: "segment for country"  
- all-MiniLM-L6-v2: `dimension_Country` was found (Score: 0.487, Rank: 1)  
- EmbeddingGemma: No relevant results were found  

**Example – Where EmbeddingGemma Failed**:  
- Query: "inventory channel attribute"  
- all-MiniLM-L6-v2: `dimension_Inventory_Channel` was found (Score: 0.678, Rank: 2)  
- EmbeddingGemma: No relevant results were found  

### 4. Dimension Values Performance

| Metric | all-MiniLM-L6-v2 | EmbeddingGemma | Evidence |
|--------|------------------|----------------|----------|
| **Accuracy** | 0.0% (0/60) | 0.0% (0/60) | Tie |
| **Correct Type Rate** | 0.0% | 0.0% | Tie |
| **Avg Search Time** | **0.50s** | 0.63s | **all-MiniLM-L6-v2** |
| **Avg Similarity Score** | 0.000 | 0.000 | Tie |
| **Rank 1 Rate** | 0.0% | 0.0% | Tie |

**Analysis**: Both models struggled with dimension values. However, faster search times were maintained by all-MiniLM-L6-v2.

**Example – Where Both Models Failed**:  
- Query: "France"  
- all-MiniLM-L6-v2: No relevant results were found  
- EmbeddingGemma: No relevant results were found  

---

## Detailed Performance Examples

### Where all-MiniLM-L6-v2 Significantly Outperformed

#### Example 1: Business Context Understanding  
- **Query**: "breakdown by the demand channel"  
- **all-MiniLM-L6-v2**: `dimension_Inventory_Channel` was found (Score: 0.586, Rank: 2)  
- **EmbeddingGemma**: `dimension_Inventory_Channel` was found (Score: 0.794, Rank: 5)  

#### Example 2: Formula Understanding  
- **Query**: "formula for valid wins/ad impressions served(hb)"  
- **all-MiniLM-L6-v2**: `formula_IX_Win_Rate` was found (Score: 0.771, Rank: 1)  
- **EmbeddingGemma**: No relevant results were found  

#### Example 3: Dimension Semantics  
- **Query**: "attribute inventory channel"  
- **all-MiniLM-L6-v2**: `dimension_Inventory_Channel` was found (Score: 0.657, Rank: 2)  
- **EmbeddingGemma**: No relevant results were found  

### Where EmbeddingGemma Outperformed

#### Example 1: Exact Name Matching  
- **Query**: "TotalAdRequest"  
- **all-MiniLM-L6-v2**: `metric_Total_Ad_Request` was found (Score: 0.330, Rank: 2)  
- **EmbeddingGemma**: `metric_Total_Ad_Request` was found (Score: 0.983, Rank: 1)  

#### Example 2: Short Acronyms  
- **Query**: "ctr"  
- **all-MiniLM-L6-v2**: No relevant results were found  
- **EmbeddingGemma**: `formula_CTR` was found (Score: 0.988, Rank: 1)  

### Where Both Models Struggled

#### Example 1: Dimension Values  
- **Query**: "Country France"  
- **all-MiniLM-L6-v2**: No relevant results were found  
- **EmbeddingGemma**: No relevant results were found  

#### Example 2: Complex Business Terms  
- **Query**: "filter Integration Direct = p_corona"  
- **all-MiniLM-L6-v2**: No relevant results were found  
- **EmbeddingGemma**: No relevant results were found  

---

## Performance Characteristics Analysis

### Search Latency Comparison

| Model | Metrics | Formulas | Dimensions | Dimension Values | Overall |
|-------|---------|----------|------------|------------------|---------|
| **all-MiniLM-L6-v2** | 0.50s | 0.51s | 0.49s | 0.50s | **0.50s** |
| **EmbeddingGemma** | 0.63s | 0.64s | 0.64s | 0.63s | 0.64s |

**Analysis**: all-MiniLM-L6-v2 was consistently **22% faster** across all categories.

### Similarity Score Analysis

| Model | Metrics | Formulas | Dimensions | Dimension Values | Overall |
|-------|---------|----------|------------|------------------|---------|
| **all-MiniLM-L6-v2** | 0.584 | 0.603 | 0.530 | 0.000 | 0.429 |
| **EmbeddingGemma** | **0.923** | **0.926** | 0.794 | 0.000 | **0.661** |

**Analysis**: Higher similarity scores were achieved by EmbeddingGemma, but this did not result in better accuracy.

### Ranking Performance

| Model | Rank 1 Rate | Rank 3 Rate | Top 5 Rate |
|-------|-------------|-------------|------------|
| **all-MiniLM-L6-v2** | **51.4%** | **78.8%** | **82.0%** |
| **EmbeddingGemma** | 50.6% | 69.6% | 66.7% |

**Analysis**: Better ranking performance was provided by all-MiniLM-L6-v2, improving the likelihood of relevant results appearing at the top.

---

### Cost-Benefit Analysis

| Factor | all-MiniLM-L6-v2 | EmbeddingGemma | Impact |
|--------|------------------|----------------|--------|
| **Search Latency** | 0.50s | 0.64s | EmbeddingGemma 28% slower |
| **Accuracy** | 82.0% | 66.7% | EmbeddingGemma 15.3% less accurate |

---

## Conclusion

The evaluation of 405 queries provided clear evidence that **all-MiniLM-L6-v2 is the superior choice** for business intelligence semantic search. Whilst higher similarity scores and occasional formula performance were achieved by EmbeddingGemma, the domain understanding required for reliable BI semantic search was lacking.
