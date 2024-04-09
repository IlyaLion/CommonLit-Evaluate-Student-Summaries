## CommonLit - Evaluate Student Summaries (113th Place)

This repository contains the code for training models that achieved 113th place (bronze medal) in a competition CommonLit - Evaluate Student Summaries
https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries

### Models Used
- DeBERTa-v3-small
- DeBERTa-v3-base
- DeBERTa-v3-large

### Pooling Techniques Utilized
- MeanMaxPooling
- LSTMPooling
- WeightedLayerPooling
- AttentionPooling
- GeMTextPooling

### Best Submission Strategy
The best submission was achieved by taking the weighted average of predictions from the top 10 performing models.
