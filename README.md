# LLM Workflow

## Overview
This project compares three different approaches to processing and repurposing content using Large Language Models (LLMs):
1. **Pipeline Workflow** – Sequential execution of tasks.
2. **DAG Workflow with Reflexion** – Tasks are structured as a Directed Acyclic Graph (DAG) with self-correction.
3. **Agent-Driven Workflow** – An autonomous agent iterates through the workflow, selecting tools dynamically.

## Setup Instructions
### **1. Clone the Repository**
```sh
 git clone https://github.com/your-repo/llm-workflow.git
 cd llm-workflow
```

### **2. Install Dependencies**
Ensure you have Python 3.8+ installed. Then, install required packages:
```sh
 pip install -r requirements.txt
```

### **3. Run the Workflow**
To execute the workflow and evaluate the approaches, run:
```sh
 python llm_workflow.py
```

## 📖 Implementation Details
### **Workflow Components:**
- **Extract Key Points:** Identifies main takeaways from the content.
- **Generate Summary:** Produces a concise summary.
- **Create Social Media Posts:** Generates Twitter, LinkedIn, and Facebook posts.
- **Create Email Newsletter:** Composes a newsletter.

### **Workflow Approaches:**
1. **Pipeline Workflow:** Runs tasks sequentially.
2. **DAG Workflow:** Uses Reflexion to refine outputs dynamically.
3. **Agent-Driven Workflow:** Uses an agent to determine and execute tasks iteratively.

## Example Outputs
### **Pipeline Workflow**
```
Execution Time: 2.88 sec
Key Points:
1. AI is revolutionizing healthcare.
Summary:
Artificial Intelligence is transforming healthcare by improving diagnosis and patient outcomes...
Social Media Posts:
- Twitter: AI is revolutionizing healthcare! Discover how AI is transforming...
Email:
- Subject: Revolutionizing Healthcare: The Power of AI
```

### **DAG Workflow**
```
Execution Time: 2.86 sec
Key Points:
1. AI is improving patient care.
Summary:
AI is transforming healthcare by improving diagnosis accuracy...
Social Media Posts:
- Twitter: AI is transforming healthcare! #HealthTech
Email:
- Subject: AI’s Role in Healthcare
```

### **Agent Workflow**
```
Execution Time: 3.17 sec
Key Points:
1. AI is transforming healthcare.
Summary:
AI is improving diagnosis accuracy and enhancing patient care...
Social Media Posts:
- Twitter: AI is revolutionizing healthcare! #AI #Healthcare
Email:
- Subject: The Future of Healthcare: How AI is Revolutionizing the Industry
```

## Effectiveness Analysis
| Approach   | Coherence | Relevance | Completeness | Execution Time |
|------------|-----------|-----------|--------------|---------------|
| **Pipeline** | 1.00 | 1.00 | 1.00 | 2.88 sec |
| **DAG** | 0.97 | 0.60 | 1.00 | 2.86 sec |
| **Agent** | 0.18 | 0.16 | 0.50 | 3.17 sec |

### **Observations:**
- **Pipeline Workflow**: Consistently high coherence, relevance, and completeness, but lacks adaptability.
- **DAG Workflow**: Provides structured execution with self-correction but is slightly less relevant.
- **Agent Workflow**: Flexible but performed inconsistently, needing improvements in decision-making.

## Challenges & Solutions
| Challenge | Solution |
|-----------|----------|
| Agent workflow sometimes repeated tasks unnecessarily. | Limited iterations and refined prompt guidance. |
| DAG workflow had occasional irrelevant outputs. | Improved Reflexion logic to discard incorrect refinements. |
| Execution times varied across runs. | Optimized function calls to reduce redundancy. |

## Conclusion
- **Pipeline** is reliable but lacks adaptability.
- **DAG** improves accuracy with self-correction.
- **Agent** is promising but needs better reasoning.

**Future Work:** Enhance the agent’s reasoning capabilities and improve execution efficiency.

---

