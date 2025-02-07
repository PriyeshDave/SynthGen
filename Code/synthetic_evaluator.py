import pandas as pd
from scipy.spatial.distance import jensenshannon

class SyntheticEvaluator:
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        self.real_data = real_data
        self.synthetic_data = synthetic_data

    def evaluate(self):
        """Evaluate synthetic data using Jensen-Shannon Distance."""
        jsd_scores = {}

        for col in self.real_data.select_dtypes(include=['object', 'category']).columns:
            real_counts = self.real_data[col].value_counts(normalize=True)
            synth_counts = self.synthetic_data[col].value_counts(normalize=True)

            # Reindex to ensure same categories in both
            all_categories = set(real_counts.index).union(set(synth_counts.index))
            real_counts = real_counts.reindex(all_categories, fill_value=0)
            synth_counts = synth_counts.reindex(all_categories, fill_value=0)

            # Compute JSD
            jsd_score = jensenshannon(real_counts, synth_counts)
            jsd_scores[col] = jsd_score

        print("Jensen-Shannon Distances:", jsd_scores)
